"""
=============================================================================
模块7: 推理引擎优化 (Inference Engine Optimization)
=============================================================================

核心知识点:
-----------

1. KV Cache（键值缓存）
   - 在自回归生成中，每生成一个新 token 都需要对之前所有 token 做 attention
   - 朴素方法：每一步都重新计算所有 token 的 K、V，计算量为 O(n^2)
   - KV Cache：把之前算过的 K、V 缓存下来，新 token 只算自己的 Q/K/V，
     然后和缓存拼接，计算量降为 O(n)
   - 显存开销：每层每个头都要存 K 和 V，形状为 [batch, heads, seq_len, head_dim]

2. Continuous Batching（连续批处理）
   - 传统 static batching：一个 batch 里所有请求必须同时开始、同时结束
   - 短请求被长请求拖慢，GPU 利用率低
   - Continuous batching：请求可以随时加入和离开 batch
   - 某个请求生成完毕后，立即用新请求填充该 slot，最大化吞吐

3. PagedAttention（分页注意力）—— vLLM 的核心创新
   - 传统 KV Cache 需要连续内存，导致内存碎片和浪费
   - PagedAttention 把 KV Cache 按 block 分页管理，类似操作系统的虚拟内存
   - 好处：减少内存浪费、支持更大 batch、支持 beam search 时共享 KV 页
   - 内存利用率从 ~50% 提升到 ~95%

4. Speculative Decoding（推测性解码）
   - 用小模型快速 "猜" 多个 token，再用大模型一次性验证
   - 如果猜对了就跳过多步，猜错了只用到错误点之前的结果
   - 不改变输出分布（数学上等价），但可以加速 2-3 倍
   - 关键：小模型足够快 + 与大模型分布足够接近

5. 实际推理引擎对比
   - vLLM：PagedAttention，连续批处理，高吞吐
   - TensorRT-LLM：NVIDIA 官方，算子融合 + 量化，低延迟
   - DeepSpeed-Inference：与 DeepSpeed 训练框架配合
   - MLC-LLM：多端部署（手机、浏览器）

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import dataclasses
from typing import Optional, List, Dict, Tuple


# =============================================================================
# 第一部分：定义 NanoGPT 模型（内联定义，不依赖外部模块）
# =============================================================================

@dataclasses.dataclass
class NanoGPTConfig:
    """NanoGPT 配置，小模型用于演示推理优化"""
    n_layers: int = 4          # Transformer 层数
    n_dim: int = 64            # 隐藏层维度
    n_heads: int = 4           # 注意力头数
    vocab_size: int = 256      # 词表大小（字节级）
    max_seq_len: int = 128     # 最大序列长度


class CausalSelfAttention(nn.Module):
    """因果自注意力层，支持可选的 KV Cache"""

    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        assert config.n_dim % config.n_heads == 0, "维度必须能被头数整除"
        self.n_heads = config.n_heads
        self.head_dim = config.n_dim // config.n_heads
        self.n_dim = config.n_dim

        # Q/K/V 投影，合并为一个线性层以提高效率
        self.qkv_proj = nn.Linear(config.n_dim, 3 * config.n_dim, bias=False)
        # 输出投影
        self.out_proj = nn.Linear(config.n_dim, config.n_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        参数:
            x: 输入张量 [batch, seq_len, dim]
            kv_cache: 之前缓存的 (K, V)，形状各为 [batch, heads, cached_len, head_dim]
            use_cache: 是否返回更新后的 KV Cache
        返回:
            output: 注意力输出 [batch, seq_len, dim]
            new_kv_cache: 更新后的 (K, V) 或 None
        """
        B, T, C = x.shape

        # 计算 Q/K/V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_dim, dim=-1)

        # 重塑为多头形式: [batch, heads, seq_len, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 如果有 KV Cache，拼接历史的 K/V
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)  # 在 seq_len 维度拼接
            v = torch.cat([cached_v, v], dim=2)

        # 保存新的 KV Cache（包含当前步的 K/V）
        new_kv_cache = (k, v) if use_cache else None

        # 计算注意力分数
        # q: [B, heads, T_q, head_dim], k: [B, heads, T_kv, head_dim]
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, heads, T_q, T_kv]

        # 因果掩码：只关注当前及之前的位置
        T_kv = k.size(2)
        # 对于 KV Cache 模式，q 的长度可能是 1（只有新 token）
        # 因果掩码需要确保新 token 能看到所有之前的 token
        causal_mask = torch.tril(torch.ones(T, T_kv, device=x.device, dtype=torch.bool))
        # 在 KV Cache 模式下，偏移掩码使新 token 能看到缓存内容
        if kv_cache is not None:
            # 新 token 可以看到所有之前的 token，所以掩码全为 True
            causal_mask = torch.ones(T, T_kv, device=x.device, dtype=torch.bool)
        attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # [B, heads, T_q, head_dim]

        # 合并多头
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out, new_kv_cache


class TransformerBlock(nn.Module):
    """Transformer 块：注意力 + FFN，带残差连接和 LayerNorm"""

    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_dim)
        # FFN: 扩展 4 倍再缩回
        self.ffn = nn.Sequential(
            nn.Linear(config.n_dim, 4 * config.n_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_dim, config.n_dim, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 注意力子层（Pre-Norm 结构）
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        # FFN 子层
        x = x + self.ffn(self.ln2(x))
        return x, new_kv_cache


class NanoGPT(nn.Module):
    """
    微型 GPT 模型
    用于演示推理优化技术，随机权重即可
    """

    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        self.config = config

        # Token Embedding 和位置编码
        self.token_emb = nn.Embedding(config.vocab_size, config.n_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.n_dim)

        # Transformer 层
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # 输出层
        self.ln_f = nn.LayerNorm(config.n_dim)
        self.lm_head = nn.Linear(config.n_dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        前向传播
        参数:
            input_ids: token 索引 [batch, seq_len]
            kv_caches: 每层的 KV Cache 列表
            use_cache: 是否启用 KV Cache
        返回:
            logits: 预测分布 [batch, seq_len, vocab_size]
            new_kv_caches: 更新后的 KV Cache 列表
        """
        B, T = input_ids.shape

        # 计算位置索引（KV Cache 模式下需要偏移）
        if kv_caches is not None and kv_caches[0] is not None:
            # 从缓存长度推断当前位置偏移
            past_len = kv_caches[0][0].size(2)  # cached_k 的 seq_len
            positions = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0)
        else:
            positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)

        # Embedding
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # 逐层通过 Transformer
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, kv_cache=layer_cache, use_cache=use_cache)
            new_kv_caches.append(new_cache)

        # 输出
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, new_kv_caches
        return logits, None


# =============================================================================
# 第二部分：推理引擎 —— KV Cache、连续批处理、吞吐统计
# =============================================================================

@dataclasses.dataclass
class GenerationRequest:
    """单个生成请求"""
    request_id: int                     # 请求唯一标识
    prompt_ids: List[int]               # 输入 prompt 的 token id 列表
    max_new_tokens: int = 20            # 最多生成的新 token 数
    generated_ids: List[int] = dataclasses.field(default_factory=list)  # 已生成的 token
    is_finished: bool = False           # 是否已完成生成
    start_time: float = 0.0            # 开始时间
    end_time: float = 0.0              # 结束时间


class InferenceEngine:
    """
    推理引擎
    实现 KV Cache 增量推理、朴素推理对比、连续批处理、吞吐统计
    """

    def __init__(self, config: NanoGPTConfig):
        self.config = config
        self.model = NanoGPT(config)
        self.model.eval()  # 推理模式，关闭 dropout 等

        # 统计数据
        self.total_tokens_generated = 0
        self.total_time_s = 0.0

    # -------------------------------------------------------------------------
    # 方法1：朴素推理（无 KV Cache，每步重新计算全部序列）
    # -------------------------------------------------------------------------
    def generate_naive(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 20,
    ) -> Tuple[List[int], float]:
        """
        朴素自回归生成：每一步都把完整序列送入模型重新计算
        这是最简单但最慢的方式，计算量随序列增长呈 O(n^2)
        """
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        generated = list(prompt_ids)

        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 每步都送入完整序列（不使用缓存）
                seq = torch.tensor([generated], dtype=torch.long)
                # 截断到最大长度
                if seq.size(1) > self.config.max_seq_len:
                    seq = seq[:, -self.config.max_seq_len:]
                logits, _ = self.model(seq, use_cache=False)
                # 取最后一个位置的 logits，贪心解码
                next_token = logits[:, -1, :].argmax(dim=-1).item()
                generated.append(next_token)

        elapsed = time.perf_counter() - start_time
        new_tokens = generated[len(prompt_ids):]
        return new_tokens, elapsed

    # -------------------------------------------------------------------------
    # 方法2：KV Cache 增量推理（只计算新 token）
    # -------------------------------------------------------------------------
    def generate_with_kv_cache(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 20,
    ) -> Tuple[List[int], float]:
        """
        KV Cache 增量推理：
        - Prefill 阶段：一次性处理整个 prompt，建立 KV Cache
        - Decode 阶段：每步只送入 1 个新 token，复用缓存的 K/V
        计算量从 O(n^2) 降为 O(n)
        """
        generated = list(prompt_ids)

        start_time = time.perf_counter()

        with torch.no_grad():
            # === Prefill 阶段 ===
            # 一次性处理 prompt，构建初始 KV Cache
            input_ids = torch.tensor([prompt_ids], dtype=torch.long)
            logits, kv_caches = self.model(input_ids, use_cache=True)
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            generated.append(next_token)

            # === Decode 阶段 ===
            # 每步只送入新生成的 1 个 token
            for _ in range(max_new_tokens - 1):
                # 只送入最后一个 token
                input_ids = torch.tensor([[next_token]], dtype=torch.long)
                logits, kv_caches = self.model(input_ids, kv_caches=kv_caches, use_cache=True)
                next_token = logits[:, -1, :].argmax(dim=-1).item()
                generated.append(next_token)

        elapsed = time.perf_counter() - start_time
        new_tokens = generated[len(prompt_ids):]
        return new_tokens, elapsed

    # -------------------------------------------------------------------------
    # 方法3：连续批处理（Continuous Batching）
    # -------------------------------------------------------------------------
    def continuous_batching_generate(
        self,
        requests: List[GenerationRequest],
    ) -> Dict[int, Dict]:
        """
        简单的连续批处理实现：
        - 维护一个活跃请求池
        - 每步将所有活跃请求 batch 在一起（用 padding 对齐长度）
        - 完成的请求移出池，统计吞吐
        注意：真实系统用 PagedAttention 管理内存，这里简化为朴素 padding 方案
        """
        # 初始化每个请求的状态
        active_requests: List[GenerationRequest] = []
        # 每个请求的 KV Cache，按 request_id 索引
        request_kv_caches: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        results: Dict[int, Dict] = {}

        # 把所有请求加入活跃池
        for req in requests:
            req.start_time = time.perf_counter()
            req.generated_ids = []
            req.is_finished = False
            active_requests.append(req)

        overall_start = time.perf_counter()
        total_tokens = 0
        step = 0

        with torch.no_grad():
            # === Prefill 阶段：逐个请求处理 prompt，建立各自的 KV Cache ===
            for req in active_requests:
                input_ids = torch.tensor([req.prompt_ids], dtype=torch.long)
                logits, kv_caches = self.model(input_ids, use_cache=True)
                next_token = logits[:, -1, :].argmax(dim=-1).item()
                req.generated_ids.append(next_token)
                request_kv_caches[req.request_id] = kv_caches
                total_tokens += 1

            # === Decode 阶段：批量处理所有活跃请求 ===
            while active_requests:
                step += 1

                # 收集每个活跃请求最后生成的 token
                batch_tokens = []
                batch_req_indices = []
                for i, req in enumerate(active_requests):
                    if req.is_finished:
                        continue
                    last_token = req.generated_ids[-1]
                    batch_tokens.append(last_token)
                    batch_req_indices.append(i)

                if not batch_tokens:
                    break

                # 逐个请求做增量推理（简化版 batching）
                # 真实系统会用 padding/packing 做真正的 batch 推理
                # 这里为了演示连续批处理的调度逻辑，逐个处理但统一调度
                tokens_this_step = 0
                finished_indices = []

                for idx, req_idx in enumerate(batch_req_indices):
                    req = active_requests[req_idx]
                    input_ids = torch.tensor([[batch_tokens[idx]]], dtype=torch.long)
                    kv_caches = request_kv_caches[req.request_id]

                    logits, new_kv_caches = self.model(
                        input_ids, kv_caches=kv_caches, use_cache=True
                    )
                    next_token = logits[:, -1, :].argmax(dim=-1).item()
                    req.generated_ids.append(next_token)
                    request_kv_caches[req.request_id] = new_kv_caches
                    tokens_this_step += 1

                    # 检查是否达到最大生成长度
                    if len(req.generated_ids) >= req.max_new_tokens:
                        req.is_finished = True
                        req.end_time = time.perf_counter()
                        finished_indices.append(req_idx)

                total_tokens += tokens_this_step

                # 移除已完成的请求（从后往前删以避免索引问题）
                for idx in sorted(finished_indices, reverse=True):
                    finished_req = active_requests.pop(idx)
                    elapsed = finished_req.end_time - finished_req.start_time
                    n_tokens = len(finished_req.generated_ids)
                    results[finished_req.request_id] = {
                        "request_id": finished_req.request_id,
                        "prompt_len": len(finished_req.prompt_ids),
                        "generated_tokens": n_tokens,
                        "latency_s": elapsed,
                        "tokens_per_s": n_tokens / elapsed if elapsed > 0 else 0,
                        "generated_ids": finished_req.generated_ids,
                    }
                    # 释放 KV Cache 内存
                    del request_kv_caches[finished_req.request_id]

        overall_elapsed = time.perf_counter() - overall_start

        # 汇总统计
        results["_summary"] = {
            "total_requests": len(requests),
            "total_tokens_generated": total_tokens,
            "total_time_s": overall_elapsed,
            "overall_throughput_tok_per_s": total_tokens / overall_elapsed if overall_elapsed > 0 else 0,
        }

        return results


# =============================================================================
# 第三部分：运行演示
# =============================================================================

def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_kv_cache_comparison(engine: InferenceEngine):
    """演示 KV Cache vs 朴素推理的速度对比"""
    print_separator("实验1: KV Cache vs 朴素推理 速度对比")

    # 不同 prompt 长度的测试
    prompt_lengths = [8, 16, 32, 64]
    max_new_tokens = 20

    print(f"\n生成 token 数: {max_new_tokens}")
    print(f"{'Prompt长度':>12} | {'朴素推理(ms)':>14} | {'KV Cache(ms)':>14} | {'加速比':>8}")
    print("-" * 60)

    for prompt_len in prompt_lengths:
        # 随机 prompt
        prompt = list(range(1, prompt_len + 1))

        # 朴素推理（预热一次）
        engine.generate_naive(prompt, max_new_tokens=2)
        _, naive_time = engine.generate_naive(prompt, max_new_tokens=max_new_tokens)

        # KV Cache 推理（预热一次）
        engine.generate_with_kv_cache(prompt, max_new_tokens=2)
        _, cache_time = engine.generate_with_kv_cache(prompt, max_new_tokens=max_new_tokens)

        speedup = naive_time / cache_time if cache_time > 0 else float('inf')

        print(
            f"{prompt_len:>12d} | "
            f"{naive_time * 1000:>12.2f}ms | "
            f"{cache_time * 1000:>12.2f}ms | "
            f"{speedup:>7.2f}x"
        )

    print("\n结论: KV Cache 通过缓存历史 K/V 避免重复计算，随序列增长加速越明显")
    print("  - 朴素方法: 每步重算全序列 attention，复杂度 O(n^2)")
    print("  - KV Cache: 每步只算新 token，复杂度 O(n)")


def demo_continuous_batching(engine: InferenceEngine):
    """演示连续批处理"""
    print_separator("实验2: 连续批处理 (Continuous Batching)")

    # 创建多个不同长度的请求
    requests = [
        GenerationRequest(
            request_id=0,
            prompt_ids=list(range(1, 6)),       # prompt 长度 5
            max_new_tokens=15,
        ),
        GenerationRequest(
            request_id=1,
            prompt_ids=list(range(1, 17)),      # prompt 长度 16
            max_new_tokens=10,
        ),
        GenerationRequest(
            request_id=2,
            prompt_ids=list(range(1, 33)),      # prompt 长度 32
            max_new_tokens=20,
        ),
        GenerationRequest(
            request_id=3,
            prompt_ids=list(range(1, 9)),       # prompt 长度 8
            max_new_tokens=12,
        ),
    ]

    print(f"\n提交 {len(requests)} 个生成请求:")
    for req in requests:
        print(f"  请求 {req.request_id}: prompt 长度={len(req.prompt_ids)}, "
              f"最大生成={req.max_new_tokens} tokens")

    # 运行连续批处理
    results = engine.continuous_batching_generate(requests)

    # 打印每个请求的结果
    print(f"\n{'请求ID':>6} | {'Prompt长度':>10} | {'生成tokens':>10} | {'延迟(ms)':>10} | {'吞吐(tok/s)':>12}")
    print("-" * 65)

    for req in requests:
        r = results[req.request_id]
        print(
            f"{r['request_id']:>6d} | "
            f"{r['prompt_len']:>10d} | "
            f"{r['generated_tokens']:>10d} | "
            f"{r['latency_s'] * 1000:>9.2f}ms | "
            f"{r['tokens_per_s']:>11.1f}"
        )

    # 汇总
    summary = results["_summary"]
    print(f"\n汇总统计:")
    print(f"  总请求数:       {summary['total_requests']}")
    print(f"  总生成 token:   {summary['total_tokens_generated']}")
    print(f"  总耗时:         {summary['total_time_s'] * 1000:.2f} ms")
    print(f"  整体吞吐:       {summary['overall_throughput_tok_per_s']:.1f} tokens/s")

    print("\n说明: 连续批处理的核心优势在于:")
    print("  - 短请求完成后立即释放资源，新请求可以填入")
    print("  - 不会因为 batch 中最长的请求拖慢整体")
    print("  - 配合 PagedAttention 可进一步减少内存碎片")


def demo_throughput_scaling(engine: InferenceEngine):
    """演示吞吐量随 batch 规模的变化"""
    print_separator("实验3: 吞吐量 vs 并发请求数")

    batch_sizes = [1, 2, 4, 8]
    max_new_tokens = 10

    print(f"\n每请求生成 {max_new_tokens} tokens")
    print(f"{'并发数':>8} | {'总tokens':>10} | {'总耗时(ms)':>12} | {'吞吐(tok/s)':>12}")
    print("-" * 55)

    for n_requests in batch_sizes:
        requests = [
            GenerationRequest(
                request_id=i,
                prompt_ids=list(range(1, 11)),  # 统一 prompt 长度为 10
                max_new_tokens=max_new_tokens,
            )
            for i in range(n_requests)
        ]

        results = engine.continuous_batching_generate(requests)
        summary = results["_summary"]

        print(
            f"{n_requests:>8d} | "
            f"{summary['total_tokens_generated']:>10d} | "
            f"{summary['total_time_s'] * 1000:>11.2f}ms | "
            f"{summary['overall_throughput_tok_per_s']:>11.1f}"
        )

    print("\n说明: 随着并发请求增加，整体吞吐量提升，")
    print("  因为计算资源得到了更充分的利用（batch 效应）")


def print_real_infra_comparison():
    """打印与真实推理基础设施的对比"""
    print_separator("与真实推理引擎的对比")

    print("""
本教程演示了推理优化的核心思想。以下是真实系统的对比:

┌─────────────────┬──────────────────────────────────────────────────────────┐
│ 特性            │ 本教程 (NanoGPT)          vs  真实系统                  │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ KV Cache        │ 简单 Python 字典存储      vs  GPU 显存 + 分页管理       │
│ 连续批处理      │ 逐请求调度模拟            vs  iteration-level 调度      │
│ 内存管理        │ PyTorch 自动管理          vs  PagedAttention 分页管理   │
│ 量化            │ FP32 全精度               vs  FP16/INT8/INT4 量化       │
│ 算子融合        │ 无                        vs  FlashAttention/融合FFN    │
│ 推测性解码      │ 未实现                    vs  小模型草稿 + 大模型验证   │
└─────────────────┴──────────────────────────────────────────────────────────┘

真实推理引擎详细对比:

1. vLLM (UC Berkeley)
   - 核心创新: PagedAttention，把 KV Cache 按页管理
   - 吞吐量比 HuggingFace 高 14-24 倍
   - 支持连续批处理、Tensor 并行
   - 适合: 高吞吐在线服务

2. TensorRT-LLM (NVIDIA)
   - 核心优势: 算子融合 + INT8/FP8 量化 + CUDA kernel 优化
   - 延迟最低，尤其在 NVIDIA GPU 上
   - 支持 Inflight Batching (连续批处理)
   - 适合: 低延迟、NVIDIA GPU 专用场景

3. DeepSpeed-Inference (Microsoft)
   - 核心优势: 与 DeepSpeed 训练框架无缝衔接
   - 自动 Tensor 并行，支持超大模型
   - 适合: 从训练到推理的一体化流程

4. MLC-LLM
   - 核心优势: 跨平台编译部署（手机、浏览器、嵌入式）
   - 基于 Apache TVM 编译优化
   - 适合: 端侧部署

5. SGLang (UC Berkeley)
   - 核心创新: RadixAttention，利用前缀共享加速多轮对话
   - 结构化生成优化（JSON、正则约束）
   - 适合: 多轮对话、复杂 prompt 编排

性能量级参考 (以 Llama-2-7B 为例，单张 A100 GPU):
  - HuggingFace 原生:    ~30 tokens/s
  - vLLM:                ~600 tokens/s (高并发)
  - TensorRT-LLM:        ~800 tokens/s (高并发)
  - 本教程 NanoGPT (CPU): 仅用于理解原理，不做性能对标
""")


def main():
    """主函数：运行所有推理优化演示"""
    print("=" * 70)
    print("  LLM 推理引擎优化教程")
    print("  模型: NanoGPT (4层, 64维, 4头, 词表256)")
    print("  设备: CPU (纯教学演示)")
    print("=" * 70)

    # 初始化配置和引擎
    config = NanoGPTConfig()
    engine = InferenceEngine(config)

    # 打印模型参数量
    n_params = sum(p.numel() for p in engine.model.parameters())
    print(f"\n模型参数量: {n_params:,} ({n_params / 1e3:.1f}K)")

    # 实验1: KV Cache vs 朴素推理
    demo_kv_cache_comparison(engine)

    # 实验2: 连续批处理
    demo_continuous_batching(engine)

    # 实验3: 吞吐量缩放
    demo_throughput_scaling(engine)

    # 与真实系统对比
    print_real_infra_comparison()

    print("\n" + "=" * 70)
    print("  教程完成！")
    print("  下一步建议: 阅读 vLLM 论文或源码，理解 PagedAttention 的实现细节")
    print("=" * 70)


if __name__ == "__main__":
    main()
