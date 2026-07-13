"""第 9 课：可观测性、评估与成本 —— 把 Agent 当"生产系统"来经营

知识点：
  - 可观测性(Observability)：给每次调用埋点，记录延迟、token、工具调用、成败。
    没有观测就无法优化——你无法改进你看不见的东西。
  - 成本核算：按 token 单价估算每次/每天花费，设预算护栏。
  - 评估(Evaluation)：用一组"测试用例 + 期望"给 Agent 打分，防止迭代时悄悄退化。
    这就是 Agent 版的"单元测试/回归测试"。

本课演示：一个轻量 Tracer（追踪器）、成本估算、一个最小评估框架。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmkit import Message, Usage, get_llm
from llmkit.base import LLMProvider


# ---------------- 可观测性：Tracer ----------------
@dataclass
class Span:
    name: str
    duration_ms: float
    usage: Usage
    ok: bool


class Tracer(LLMProvider):
    """代理任意 provider，为每次 chat 记录一条 span（延迟/用量/成败）。"""

    #: 各模型的粗略单价（美元 / 百万 token），仅供演示
    PRICES = {"mock-1": (0.0, 0.0), "claude-opus-4-8": (5.0, 25.0), "gpt-4o-mini": (0.15, 0.6)}

    def __init__(self, inner: LLMProvider):
        self.inner = inner
        self.name = f"traced({inner.name})"
        self.model = inner.model
        self.spans: List[Span] = []

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=1024):
        t0 = time.time()
        ok = True
        try:
            resp = self.inner.chat(messages, tools, temperature, max_tokens)
            return resp
        except Exception:
            ok = False
            raise
        finally:
            dur = (time.time() - t0) * 1000
            usage = resp.usage if ok else Usage()
            self.spans.append(Span("chat", dur, usage, ok))

    def report(self):
        total = sum((s.usage for s in self.spans), Usage())
        avg_ms = sum(s.duration_ms for s in self.spans) / max(1, len(self.spans))
        pin, pout = self.PRICES.get(self.model, (0.0, 0.0))
        cost = total.input_tokens / 1e6 * pin + total.output_tokens / 1e6 * pout
        print(f"  调用次数: {len(self.spans)} | 平均延迟: {avg_ms:.1f}ms")
        print(f"  累计 token: in={total.input_tokens} out={total.output_tokens}")
        print(f"  估算成本: ${cost:.6f}（按 {self.model} 单价）")


def demo_tracing(llm):
    print("\n=== 1. 可观测性 + 成本核算 ===")
    traced = Tracer(llm)
    for q in ["你好", "介绍一下 Agent", "什么是 RAG"]:
        traced.chat([Message.user(q)])
    traced.report()


# ---------------- 评估：最小回归测试框架 ----------------
@dataclass
class Case:
    question: str
    must_include: List[str] = field(default_factory=list)  # 答案必须包含的关键词
    scorer: Callable[[str], bool] = None  # 或自定义打分


def evaluate(llm, cases: List[Case]):
    print("\n=== 2. 评估：给 Agent 跑回归测试 ===")
    passed = 0
    for i, c in enumerate(cases, 1):
        answer = llm.chat([Message.user(c.question)]).content
        if c.scorer:
            ok = c.scorer(answer)
        else:
            ok = all(k in answer for k in c.must_include)
        passed += ok
        print(f"  用例{i} [{'通过' if ok else '失败'}] {c.question} -> {answer[:40]}")
    print(f"  通过率: {passed}/{len(cases)} = {passed/len(cases):.0%}")


def demo_eval(llm):
    cases = [
        # Mock 会回显问题内容，因此关键词校验可离线通过；接真实模型时改成真实期望
        Case("请在回答中包含'你好'两个字", must_include=["你好"]),
        Case("任意回答", scorer=lambda ans: len(ans) > 0),
    ]
    evaluate(llm, cases)


if __name__ == "__main__":
    llm = get_llm()
    print(f"当前后端: {llm.name} / {llm.model}")
    demo_tracing(llm)
    demo_eval(llm)
    print("\n小结：可观测(埋点) + 可度量(成本/延迟) + 可评估(回归) = 可持续迭代的 Agent。")
