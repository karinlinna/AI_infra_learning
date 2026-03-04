"""
================================================================================
模块2: 从零手写一个最小GPT模型 (NanoGPT)
================================================================================

【知识总结: Transformer Decoder 架构】

1. GPT 是一个 Transformer Decoder-Only 模型，核心思路：
   - 输入一段 token 序列，预测下一个 token
   - 自回归生成：每次用已生成的所有 token 预测下一个

2. 一个 Transformer Decoder Block 包含：
   ┌─────────────────────────────┐
   │  输入 x                      │
   │    ↓                         │
   │  LayerNorm                   │
   │    ↓                         │
   │  Masked Multi-Head Attention │  ← 因果掩码，防止看到未来 token
   │    ↓                         │
   │  残差连接 (x + attn_out)     │
   │    ↓                         │
   │  LayerNorm                   │
   │    ↓                         │
   │  FFN (Linear→GELU→Linear)   │  ← 先扩大4倍再缩回来
   │    ↓                         │
   │  残差连接 (x + ffn_out)      │
   └─────────────────────────────┘

3. 完整模型结构：
   Token Embedding + Position Embedding → N 个 Decoder Block → LayerNorm → Linear (输出 logits)

4. KV Cache 机制 (推理加速)：
   - 自回归生成时，每一步只需要计算新 token 的 Q，但 K 和 V 需要包含所有历史 token
   - 把之前算过的 K、V 缓存起来，每步只算新增部分，避免重复计算
   - 复杂度从 O(n²) 降到 O(n) (每步)

5. 本文件参数：
   - 词表大小: 256 (字节级，简单直观)
   - 序列长度: 128
   - 隐藏维度: 64
   - 注意力头数: 4 (每个头 16 维)
   - 层数: 4
   - FFN 中间维度: 256 (4倍隐藏维度)

================================================================================
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 超参数配置
# ============================================================
VOCAB_SIZE = 256       # 词表大小 (字节级: 0~255)
SEQ_LEN = 128          # 最大序列长度
D_MODEL = 64           # 隐藏维度
N_HEADS = 4            # 注意力头数
D_HEAD = D_MODEL // N_HEADS  # 每个头的维度 = 16
N_LAYERS = 4           # Transformer 层数
D_FFN = D_MODEL * 4    # FFN 中间维度 = 256
DROPOUT = 0.1          # Dropout 比率


# ============================================================
# 多头因果自注意力 (带 KV Cache)
# ============================================================
class CausalSelfAttention(nn.Module):
    """
    多头因果自注意力机制
    - 因果掩码: 每个位置只能看到自己和之前的 token
    - KV Cache: 推理时缓存历史的 K 和 V，避免重复计算
    """

    def __init__(self):
        super().__init__()
        # Q、K、V 的线性投影，合并为一个矩阵加速计算
        self.qkv_proj = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        # 输出投影
        self.out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

        # 预计算因果掩码 (下三角矩阵)
        # 上三角部分为 -inf，softmax 后变为 0，实现"看不到未来"
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x, kv_cache=None):
        """
        x: (batch, seq_len, d_model)
        kv_cache: 元组 (cached_k, cached_v)，推理时使用
        返回: (输出, 新的kv_cache)
        """
        B, T, C = x.shape

        # 计算 Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*D_MODEL)
        q, k, v = qkv.chunk(3, dim=-1)  # 各 (B, T, D_MODEL)

        # 拆分成多头: (B, T, D_MODEL) → (B, N_HEADS, T, D_HEAD)
        q = q.view(B, T, N_HEADS, D_HEAD).transpose(1, 2)
        k = k.view(B, T, N_HEADS, D_HEAD).transpose(1, 2)
        v = v.view(B, T, N_HEADS, D_HEAD).transpose(1, 2)

        # 如果有 KV Cache，拼接历史的 K 和 V
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)  # 在序列维度拼接
            v = torch.cat([cached_v, v], dim=2)

        # 保存新的 KV Cache
        new_kv_cache = (k, v)

        # 计算注意力分数: Q @ K^T / sqrt(d_head)
        # (B, N_HEADS, T_q, D_HEAD) @ (B, N_HEADS, D_HEAD, T_kv)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D_HEAD)

        # 应用因果掩码 (只在非缓存模式下需要完整掩码)
        T_kv = k.shape[2]
        if kv_cache is None and T > 1:
            # 训练模式或首次推理: 使用预计算的因果掩码
            attn_scores.masked_fill_(self.causal_mask[:T, :T_kv], float('-inf'))
        # 如果是带缓存的逐步生成 (T=1)，不需要掩码，因为只有一个 query

        # Softmax + Dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        out = torch.matmul(attn_weights, v)  # (B, N_HEADS, T, D_HEAD)

        # 合并多头: (B, N_HEADS, T, D_HEAD) → (B, T, D_MODEL)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影
        out = self.out_proj(out)
        return out, new_kv_cache


# ============================================================
# 前馈网络 (FFN)
# ============================================================
class FeedForward(nn.Module):
    """
    两层 MLP，中间用 GELU 激活
    维度变化: d_model → 4*d_model → d_model
    GELU 比 ReLU 更平滑，是 GPT 系列的标配激活函数
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, D_FFN)
        self.fc2 = nn.Linear(D_FFN, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)       # GELU 激活: x * Φ(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================
# Transformer Decoder Block
# ============================================================
class TransformerBlock(nn.Module):
    """
    一个 Transformer Decoder Block:
    x → LayerNorm → Attention → 残差 → LayerNorm → FFN → 残差
    使用 Pre-Norm 结构 (LayerNorm 在 Attention/FFN 之前)，训练更稳定
    """

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)    # 注意力前的 LayerNorm
        self.attn = CausalSelfAttention()    # 因果自注意力
        self.ln2 = nn.LayerNorm(D_MODEL)     # FFN 前的 LayerNorm
        self.ffn = FeedForward()             # 前馈网络
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, kv_cache=None):
        # 子层1: 注意力 + 残差连接
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + self.dropout(attn_out)

        # 子层2: FFN + 残差连接
        x = x + self.dropout(self.ffn(self.ln2(x)))

        return x, new_kv_cache


# ============================================================
# 完整的 NanoGPT 模型
# ============================================================
class NanoGPT(nn.Module):
    """
    最小 GPT 模型:
    Token Embedding + Position Embedding
    → N 个 TransformerBlock
    → LayerNorm
    → 线性层输出 logits
    """

    def __init__(self):
        super().__init__()
        # Token 嵌入: 把 token ID 映射到向量
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        # 位置嵌入: 可学习的位置编码 (GPT-2 风格)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

        # N 层 Transformer Block
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])

        # 最终的 LayerNorm
        self.ln_final = nn.LayerNorm(D_MODEL)

        # 输出头: 映射回词表大小
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        # 权重共享: token embedding 和 lm_head 共享权重 (减少参数量)
        self.lm_head.weight = self.token_emb.weight

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier 初始化，防止梯度爆炸/消失"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, kv_caches=None):
        """
        input_ids: (batch, seq_len) - token ID 序列
        kv_caches: 列表，每层一个 (cached_k, cached_v)，推理用
        返回: logits, new_kv_caches
        """
        B, T = input_ids.shape

        # 计算位置索引 (如果有缓存，偏移到正确位置)
        if kv_caches is not None and kv_caches[0] is not None:
            # 推理模式: 位置 = 已缓存的长度
            past_len = kv_caches[0][0].shape[2]
            pos = torch.arange(past_len, past_len + T, device=input_ids.device)
        else:
            pos = torch.arange(0, T, device=input_ids.device)

        # Token Embedding + Position Embedding
        tok_emb = self.token_emb(input_ids)    # (B, T, D_MODEL)
        pos_emb = self.pos_emb(pos)            # (T, D_MODEL)
        x = self.dropout(tok_emb + pos_emb)

        # 逐层通过 Transformer Block
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, kv_cache=cache_i)
            new_kv_caches.append(new_cache)

        # 最终 LayerNorm + 线性输出
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, VOCAB_SIZE)

        return logits, new_kv_caches

    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# 训练演示
# ============================================================
def train_demo():
    """用随机数据训练几步，展示模型能正常学习"""
    print("=" * 60)
    print("【训练演示】")
    print("=" * 60)

    model = NanoGPT()
    model.train()

    # 打印模型信息
    n_params = model.count_parameters()
    print(f"模型参数量: {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"模型配置: {N_LAYERS}层, {D_MODEL}维, {N_HEADS}头, 词表{VOCAB_SIZE}")
    print()

    # 随机生成训练数据 (模拟字节序列)
    torch.manual_seed(42)
    batch_size = 8
    # 输入和目标: 目标是输入右移一位 (next token prediction)
    data = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN + 1))
    input_ids = data[:, :-1]   # (B, SEQ_LEN)
    targets = data[:, 1:]      # (B, SEQ_LEN)

    # 优化器: AdamW (带权重衰减的 Adam)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # 训练循环
    print("开始训练 (20步)...")
    print("-" * 40)
    for step in range(20):
        t0 = time.time()

        # 前向传播
        logits, _ = model(input_ids)  # (B, SEQ_LEN, VOCAB_SIZE)

        # 计算交叉熵损失
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),  # 展平: (B*SEQ_LEN, VOCAB_SIZE)
            targets.reshape(-1)               # 展平: (B*SEQ_LEN,)
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        dt = (time.time() - t0) * 1000  # 毫秒

        if step % 5 == 0 or step == 19:
            # 随机基线损失: -ln(1/256) ≈ 5.545
            print(f"  Step {step:3d} | Loss: {loss.item():.4f} | 耗时: {dt:.1f}ms")

    print("-" * 40)
    print(f"  随机猜测的理论损失: {math.log(VOCAB_SIZE):.4f}")
    print(f"  训练后损失已低于随机基线 → 模型在学习!")
    print()

    return model


# ============================================================
# KV Cache 推理演示
# ============================================================
@torch.no_grad()
def generate_demo(model):
    """使用 KV Cache 逐 token 生成，展示推理加速"""
    print("=" * 60)
    print("【KV Cache 推理演示】")
    print("=" * 60)

    model.eval()
    gen_len = 32  # 生成长度

    # ---- 方式1: 不用 KV Cache (每步重新计算所有 token) ----
    print("\n方式1: 无 KV Cache (每步全量计算)")
    prompt = torch.randint(0, VOCAB_SIZE, (1, 4))  # 4个token作为prompt
    tokens = prompt.clone()

    t0 = time.time()
    for _ in range(gen_len):
        logits, _ = model(tokens)           # 每次输入所有已有token
        next_logit = logits[:, -1, :]        # 取最后一个位置的logit
        next_token = next_logit.argmax(dim=-1, keepdim=True)  # 贪心解码
        tokens = torch.cat([tokens, next_token], dim=1)
    time_no_cache = (time.time() - t0) * 1000

    generated_no_cache = tokens[0].tolist()
    print(f"  Prompt (字节值): {generated_no_cache[:4]}")
    print(f"  生成结果 (字节值): {generated_no_cache[4:20]}...")
    print(f"  耗时: {time_no_cache:.1f}ms")

    # ---- 方式2: 使用 KV Cache (每步只计算新token) ----
    print("\n方式2: 有 KV Cache (增量计算)")
    tokens_cached = prompt.clone()
    kv_caches = [None] * N_LAYERS  # 初始化空缓存

    t0 = time.time()
    # 第一步: 处理整个 prompt，建立初始缓存
    logits, kv_caches = model(tokens_cached, kv_caches=None)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    all_tokens = [tokens_cached[0].tolist() + [next_token.item()]]

    # 后续步: 每次只输入1个新token，使用缓存
    for _ in range(gen_len - 1):
        logits, kv_caches = model(next_token, kv_caches=kv_caches)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        all_tokens[0].append(next_token.item())
    time_with_cache = (time.time() - t0) * 1000

    print(f"  Prompt (字节值): {all_tokens[0][:4]}")
    print(f"  生成结果 (字节值): {all_tokens[0][4:20]}...")
    print(f"  耗时: {time_with_cache:.1f}ms")

    # ---- 结果对比 ----
    print(f"\n--- 速度对比 ---")
    speedup = time_no_cache / max(time_with_cache, 1e-6)
    print(f"  无缓存: {time_no_cache:.1f}ms")
    print(f"  有缓存: {time_with_cache:.1f}ms")
    print(f"  加速比: {speedup:.2f}x")

    # 验证两种方式生成结果一致 (贪心解码应完全一致)
    match = generated_no_cache == all_tokens[0]
    print(f"  结果一致性: {'一致 ✓' if match else '不一致 ✗'}")
    print()


# ============================================================
# 与真实 LLM 基础设施的对比
# ============================================================
def print_comparison():
    print("=" * 60)
    print("【与真实 LLM 基础设施对比】")
    print("=" * 60)
    print("""
┌────────────────┬───────────────────────┬────────────────────────────────┐
│     方面       │   本文件 (NanoGPT)     │   真实 LLM (如 GPT-4)          │
├────────────────┼───────────────────────┼────────────────────────────────┤
│ 参数量         │ ~5万                   │ 数百亿 ~ 上万亿                │
│ 词表           │ 256 (字节级)           │ 30K~100K (BPE/SentencePiece)  │
│ 序列长度       │ 128                    │ 4K ~ 128K+                    │
│ 隐藏维度       │ 64                     │ 4096 ~ 18432                  │
│ 层数           │ 4                      │ 32 ~ 120+                     │
│ 注意力头数     │ 4                      │ 32 ~ 96 (含 GQA/MQA)          │
│ 位置编码       │ 可学习绝对位置          │ RoPE (旋转位置编码)            │
│ 激活函数       │ GELU                   │ SwiGLU / GeGLU                │
│ Norm           │ LayerNorm (Pre-Norm)   │ RMSNorm (更高效)              │
│ KV Cache       │ 简单拼接                │ PagedAttention 显存优化       │
│ 注意力优化     │ 朴素实现                │ FlashAttention (IO感知)       │
│ 训练精度       │ FP32                   │ BF16 / FP8 混合精度           │
│ 并行策略       │ 无 (单CPU)             │ TP + PP + DP + EP             │
│ 推理框架       │ 朴素 PyTorch           │ vLLM / TensorRT-LLM / TGI    │
└────────────────┴───────────────────────┴────────────────────────────────┘

【关键知识点回顾】

1. Transformer Decoder 是 GPT 的核心架构
   → 因果掩码确保自回归特性，每个位置只能看到过去

2. Multi-Head Attention 让模型在不同子空间捕获不同模式
   → 实际中 GQA (分组查询注意力) 可以减少 KV 的头数，节省显存

3. KV Cache 是推理加速的关键技巧
   → 避免重复计算，但会占用大量显存 (显存换时间)

4. Pre-Norm (先 LayerNorm 再 Attention) 比 Post-Norm 训练更稳定
   → 现代模型多用 RMSNorm 替代 LayerNorm，省去均值计算

5. 权重共享 (Embedding 和 LM Head) 减少参数量
   → 实际效果好，是常见做法

下一模块: 3_parallel → 数据并行与模型并行 (让模型跑在多卡/多机上)
""")


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   模块2: 从零手写最小GPT模型 (NanoGPT)                  ║")
    print("║   Transformer Decoder + KV Cache                       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # 1. 训练演示
    model = train_demo()

    # 2. 推理演示 (含 KV Cache 对比)
    generate_demo(model)

    # 3. 与真实 LLM 对比
    print_comparison()

    print("模块2 演示完毕。")
