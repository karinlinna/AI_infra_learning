"""第 8 课：优化与性能调优 —— 让 Agent 又快、又省、又稳

这是把 demo 推向生产的关键一课。围绕三个维度系统讲解可落地的调优手段：

  【延迟 Latency】
    1) 流式输出：降低"首字延迟(TTFT)"，改善体感；
    2) 并发：多个独立请求并行发，吞吐量翻倍。

  【成本 Cost】
    3) 精确缓存：相同请求直接复用，省一次调用；
    4) 语义缓存：近似相同的问题也复用；
    5) 模型路由：简单任务用小/便宜模型，难任务才上大模型；
    6) 上下文裁剪：只带必要的历史，token 越少越省越快。

  【稳定 Robustness】
    7) 重试 + 指数退避 + jitter：从容应对 429/5xx/网络抖动。

全部离线可跑：用带"模拟延迟"的 Mock 直观展示并发/缓存带来的墙钟时间收益。
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmkit import Message, Usage, wrap_with_cache, with_retry
from llmkit.providers.mock import MockProvider
from llmkit.rag import HashingEmbedder, cosine


# ---------------- 1. 流式：降低首字延迟 ----------------
def demo_streaming_ttft():
    print("\n=== 1. 流式输出：改善首字延迟(TTFT) ===")
    llm = MockProvider(latency=0.3)
    t0 = time.time()
    first_token_at = None
    for ev in llm.stream([Message.user("介绍一下性能优化")]):
        if ev.type == "text" and first_token_at is None:
            first_token_at = time.time() - t0
            break
    print(f"  首字到达用时 ~{first_token_at:.2f}s（非流式则要等完整生成才有反馈）")


# ---------------- 2. 并发：提升吞吐 ----------------
def demo_concurrency():
    print("\n=== 2. 并发：多个独立请求并行 ===")
    llm = MockProvider(latency=0.2)
    questions = [f"问题{i}" for i in range(8)]

    t0 = time.time()
    for q in questions:  # 串行
        llm.chat([Message.user(q)])
    serial = time.time() - t0

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:  # 并发
        list(pool.map(lambda q: llm.chat([Message.user(q)]), questions))
    parallel = time.time() - t0

    print(f"  串行 {len(questions)} 个请求: {serial:.2f}s")
    print(f"  并发 {len(questions)} 个请求: {parallel:.2f}s  (~{serial/parallel:.1f}x 提速)")


# ---------------- 3. 精确缓存：省钱省时 ----------------
def demo_exact_cache():
    print("\n=== 3. 精确缓存：相同请求直接复用 ===")
    llm = wrap_with_cache(MockProvider(latency=0.2))
    msgs = [Message.user("什么是缓存？")]

    t0 = time.time(); llm.chat(msgs); miss = time.time() - t0
    t0 = time.time(); llm.chat(msgs); hit = time.time() - t0  # 第二次命中缓存

    print(f"  首次(未命中): {miss:.2f}s | 再次(命中): {hit:.3f}s")
    print(f"  命中率: {llm.hit_rate:.0%}（命中即省下一次真实模型调用）")


# ---------------- 4. 语义缓存：近似问题也复用 ----------------
class SemanticCache:
    """当新问题与历史问题足够相似时，直接返回历史答案。"""

    def __init__(self, threshold: float = 0.85):
        self.embedder = HashingEmbedder()
        self.threshold = threshold
        self.entries: List[tuple] = []  # (vector, question, answer)

    def get(self, question: str):
        qv = self.embedder.embed(question)
        for vec, q, ans in self.entries:
            if cosine(qv, vec) >= self.threshold:
                return ans, q
        return None, None

    def put(self, question: str, answer: str):
        self.entries.append((self.embedder.embed(question), question, answer))


def demo_semantic_cache():
    print("\n=== 4. 语义缓存：措辞不同、意思相近也能复用 ===")
    llm = MockProvider()
    cache = SemanticCache(threshold=0.7)

    def ask(q: str):
        ans, matched = cache.get(q)
        if ans is not None:
            print(f"  「{q}」-> 命中语义缓存（相似于「{matched}」）")
            return ans
        ans = llm.chat([Message.user(q)]).content
        cache.put(q, ans)
        print(f"  「{q}」-> 未命中，调用模型")
        return ans

    ask("如何优化 Agent 性能？")
    ask("怎样优化 Agent 性能呢")  # 措辞略变，应命中语义缓存


# ---------------- 5. 模型路由：难易分流控成本 ----------------
def demo_model_routing():
    print("\n=== 5. 模型路由：简单任务走便宜模型 ===")
    cheap = MockProvider(model="haiku-便宜", latency=0.05)
    strong = MockProvider(model="opus-强大", latency=0.3)

    def route(question: str) -> MockProvider:
        # 启发式：短问题/分类任务 -> 便宜模型；长/复杂 -> 强模型
        hard = len(question) > 20 or any(k in question for k in ["设计", "推导", "方案", "架构"])
        return strong if hard else cheap

    for q in ["今天几号？", "帮我设计一个高可用的分布式限流方案并推导其容量模型"]:
        m = route(q)
        print(f"  「{q[:15]}...」-> 路由到 [{m.model}]")


# ---------------- 6. 上下文裁剪：token 即成本 ----------------
def demo_context_trim():
    print("\n=== 6. 上下文裁剪：只带必要历史 ===")
    llm = MockProvider()
    long_history = [Message.system("你是助手")] + [
        Message.user(f"历史消息 {i}" * 10) for i in range(20)
    ]
    trimmed = [long_history[0]] + long_history[-4:]  # 保留 system + 最近 4 条
    print(f"  裁剪前约 {llm.count_tokens(long_history)} tokens"
          f" -> 裁剪后约 {llm.count_tokens(trimmed)} tokens")
    print("  （提示缓存另一利器：把稳定前缀放最前，命中缓存可再省 ~90% 输入成本）")


# ---------------- 7. 重试 + 指数退避 ----------------
def demo_retry():
    print("\n=== 7. 重试 + 指数退避：应对瞬时故障 ===")
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:  # 前两次故意失败
            raise ConnectionError("模拟 429/网络抖动")
        return "成功"

    result = with_retry(flaky, max_retries=5, base_delay=0.01, sleep=lambda s: None)
    print(f"  第 {calls['n']} 次尝试后 -> {result}")


if __name__ == "__main__":
    print("性能调优全景演示（离线 Mock）")
    demo_streaming_ttft()
    demo_concurrency()
    demo_exact_cache()
    demo_semantic_cache()
    demo_model_routing()
    demo_context_trim()
    demo_retry()
    print("\n小结：延迟靠[流式+并发]，成本靠[缓存+路由+裁剪]，稳定靠[重试退避]。")
