"""Anthropic Claude 适配器。

把 llmkit 的统一类型翻译成 Anthropic Messages API，再把响应翻译回来。
默认模型 claude-opus-4-8（当前最强 Opus 档），并采用自适应思考（adaptive thinking）。

只有当 LLM_PROVIDER=anthropic 时才会 import anthropic，因此不装该 SDK 也能跑 Mock 课程。
"""

from __future__ import annotations

import json
import os
from typing import Iterator, List, Optional

from ..base import LLMProvider
from ..types import LLMResponse, Message, StreamEvent, ToolCall, ToolSpec, Usage


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-8", api_key: Optional[str] = None):
        try:
            import anthropic  # 延迟导入
        except ImportError as e:
            raise RuntimeError("请先 `pip install anthropic` 才能使用 Anthropic provider") from e
        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    # ---- 类型转换：llmkit -> Anthropic ----
    def _to_anthropic(self, messages: List[Message]):
        system_parts, converted = [], []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "tool":
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content,
                    }],
                })
            elif m.role == "assistant" and m.tool_calls:
                blocks = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({
                        "type": "tool_use", "id": tc.id,
                        "name": tc.name, "input": tc.arguments,
                    })
                converted.append({"role": "assistant", "content": blocks})
            else:
                converted.append({"role": m.role, "content": m.content})
        return "\n".join(system_parts), converted

    def _tools(self, tools: Optional[List[ToolSpec]]):
        if not tools:
            return None
        return [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=1024) -> LLMResponse:
        system, msgs = self._to_anthropic(messages)
        kwargs = dict(model=self.model, max_tokens=max_tokens, messages=msgs,
                      thinking={"type": "adaptive"})
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._tools(tools)
        resp = self._client.messages.create(**kwargs)

        text, tool_calls = "", []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        return LLMResponse(
            content=text,
            tool_calls=tool_calls,
            usage=Usage(resp.usage.input_tokens, resp.usage.output_tokens),
            stop_reason="tool_use" if resp.stop_reason == "tool_use" else "end",
            model=self.model,
            raw=resp,
        )

    def stream(self, messages, tools=None, temperature=0.7, max_tokens=1024) -> Iterator[StreamEvent]:
        system, msgs = self._to_anthropic(messages)
        kwargs = dict(model=self.model, max_tokens=max_tokens, messages=msgs)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._tools(tools)
        with self._client.messages.stream(**kwargs) as s:
            for text in s.text_stream:
                yield StreamEvent(type="text", text=text)
            final = s.get_final_message()
            for block in final.content:
                if block.type == "tool_use":
                    yield StreamEvent(type="tool",
                                      tool_call=ToolCall(block.id, block.name, dict(block.input)))
            yield StreamEvent(type="done",
                              usage=Usage(final.usage.input_tokens, final.usage.output_tokens))

    def count_tokens(self, messages: List[Message]) -> int:
        system, msgs = self._to_anthropic(messages)
        kwargs = dict(model=self.model, messages=msgs)
        if system:
            kwargs["system"] = system
        return self._client.messages.count_tokens(**kwargs).input_tokens
