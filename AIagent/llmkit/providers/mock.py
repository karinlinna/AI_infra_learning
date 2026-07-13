"""MockProvider —— 离线、确定性、零依赖的"假模型"。

教学价值：
1. 学员无需 API Key、无需联网即可跑通所有课程；
2. 输出确定，便于讲解和复现；
3. 通过一个可插拔的 responder 回调，我们能精确编排模型行为
   （比如"先调用天气工具，再根据结果作答"），从而演示 Agent 循环、工具调用等，
   而不依赖真实模型的随机性。

它不是玩具：它展示了"如何为 Agent 系统做可测试的替身（test double）"，
这是工程化 Agent 的重要一环——你不会想在每次单测里都真的烧 token。
"""

from __future__ import annotations

import re
import time
from typing import Callable, Iterator, List, Optional

from ..base import LLMProvider
from ..types import (
    LLMResponse,
    Message,
    StreamEvent,
    ToolCall,
    ToolSpec,
    Usage,
)

# responder 签名：给定完整上下文，返回一个 LLMResponse
Responder = Callable[[List[Message], Optional[List[ToolSpec]]], LLMResponse]


def default_responder(messages: List[Message], tools: Optional[List[ToolSpec]]) -> LLMResponse:
    """默认的启发式行为，足够驱动前几节课的演示：

    - 若上一条是工具结果 → 基于工具结果给出"最终答复"；
    - 若有可用工具且用户问题命中工具关键词 → 发起一次工具调用；
    - 否则 → 回显式作答（把用户问题复述并给一个模板回答）。
    """
    last = messages[-1] if messages else Message.user("")

    # 1) 刚拿到工具结果 → 收尾作答
    if last.role == "tool":
        return LLMResponse(
            content=f"根据工具「{last.name}」的结果：{last.content}，我的回答是：已为你处理完成。",
            stop_reason="end",
            usage=Usage(input_tokens=_est(messages), output_tokens=20),
        )

    # 2) 尝试触发工具调用
    user_text = _last_user_text(messages)
    if tools:
        for spec in tools:
            # 简单启发式：工具名或其关键词出现在问题里就调用
            keywords = [spec.name] + spec.name.split("_")
            if any(k and k in user_text for k in keywords) or _mentions(user_text, spec):
                args = _guess_args(spec, user_text)
                return LLMResponse(
                    tool_calls=[ToolCall(id=f"call_{spec.name}", name=spec.name, arguments=args)],
                    stop_reason="tool_use",
                    usage=Usage(input_tokens=_est(messages), output_tokens=15),
                )

    # 3) 兜底：模板作答
    return LLMResponse(
        content=f"[mock] 我已理解你的问题：「{user_text}」。这是一个离线演示回答。",
        stop_reason="end",
        usage=Usage(input_tokens=_est(messages), output_tokens=18),
    )


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, responder: Optional[Responder] = None, model: str = "mock-1", latency: float = 0.0):
        self.responder = responder or default_responder
        self.model = model
        self._latency = latency  # 模拟网络/推理延迟，用于优化课演示并发收益

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=1024) -> LLMResponse:
        if self._latency:
            time.sleep(self._latency)
        resp = self.responder(list(messages), tools)
        resp.model = self.model
        return resp

    def stream(self, messages, tools=None, temperature=0.7, max_tokens=1024) -> Iterator[StreamEvent]:
        # 逐字符流式，直观展示"打字机效果"从何而来
        resp = self.chat(messages, tools, temperature, max_tokens)
        for ch in resp.content:
            if self._latency:
                time.sleep(self._latency / max(1, len(resp.content)))
            yield StreamEvent(type="text", text=ch)
        for tc in resp.tool_calls:
            yield StreamEvent(type="tool", tool_call=tc)
        yield StreamEvent(type="done", usage=resp.usage)


# ---------- 内部小工具 ----------

def _est(messages: List[Message]) -> int:
    return max(1, sum(len(m.content or "") for m in messages) // 4)


def _last_user_text(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _mentions(text: str, spec: ToolSpec) -> bool:
    # 用 description 里中文的 2-gram 做弱匹配，如"天气""计算""搜索"
    for run in re.findall(r"[一-龥]{2,}", spec.description):
        for i in range(len(run) - 1):
            if run[i:i + 2] in text:
                return True
    return False


def _guess_args(spec: ToolSpec, user_text: str) -> dict:
    """从用户问题里"猜"出工具参数。教学演示够用，真实模型会自己抽取。"""
    props = spec.parameters.get("properties", {})
    args = {}
    for pname, pdef in props.items():
        # 抽取数字给 number/integer 类型
        if pdef.get("type") in ("number", "integer"):
            nums = re.findall(r"-?\d+(?:\.\d+)?", user_text)
            if nums:
                args[pname] = float(nums[0]) if pdef["type"] == "number" else int(float(nums[0]))
        else:
            # 字符串参数：优先抽取"天气/市"前面的地名，否则取首个中文词，再否则整句
            m = (re.search(r"([一-龥]{2,4})(?=的?天气|市)", user_text)
                 or re.search(r"[一-龥]{2,4}", user_text))
            args[pname] = m.group(0) if m else user_text
    return args
