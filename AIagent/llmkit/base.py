"""LLMProvider 抽象基类 —— 整个框架的"接缝"。

任何模型后端只要实现这几个方法，上层的 Agent / RAG / 多智能体代码就能原样复用。
这正是"可切换 LLM"的关键：**依赖倒置** —— 上层依赖抽象，具体 provider 依赖抽象。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from .types import LLMResponse, Message, StreamEvent, ToolSpec


class LLMProvider(ABC):
    """所有模型后端的统一接口。"""

    #: 便于日志与成本核算的标识
    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """一次完整（阻塞）的对话补全。"""
        raise NotImplementedError

    def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[StreamEvent]:
        """流式输出。默认实现：退化为一次 chat 后再切成单个事件。

        真正的 provider 应覆写它以获得逐 token 体验（见优化课）。
        """
        resp = self.chat(messages, tools, temperature, max_tokens)
        if resp.content:
            yield StreamEvent(type="text", text=resp.content)
        for tc in resp.tool_calls:
            yield StreamEvent(type="tool", tool_call=tc)
        yield StreamEvent(type="done", usage=resp.usage)

    def count_tokens(self, messages: List[Message]) -> int:
        """估算输入 token 数。默认用"字符数/4"的粗略近似。

        生产中应使用官方 tokenizer（Anthropic 用 count_tokens 接口，
        OpenAI 用 tiktoken）。这里保持零依赖，便于教学。
        """
        chars = sum(len(m.content or "") for m in messages)
        return max(1, chars // 4)
