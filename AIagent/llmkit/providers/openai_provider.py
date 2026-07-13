"""OpenAI 兼容适配器。

用 openai SDK 的 chat.completions 接口。因为国内很多模型服务（DeepSeek、通义、
Kimi、本地 vLLM/Ollama 等）都提供 OpenAI 兼容端点，所以这个适配器同时也是
"接入国产/本地模型"的通用入口——只需改 base_url 与 model。
"""

from __future__ import annotations

import json
import os
from typing import Iterator, List, Optional

from ..base import LLMProvider
from ..types import LLMResponse, Message, StreamEvent, ToolCall, ToolSpec, Usage


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        try:
            from openai import OpenAI  # 延迟导入
        except ImportError as e:
            raise RuntimeError("请先 `pip install openai` 才能使用 OpenAI provider") from e
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),  # 指向本地/国产端点即可
        )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def _to_openai(self, messages: List[Message]):
        out = []
        for m in messages:
            if m.role == "assistant" and m.tool_calls:
                out.append({
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [{
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    } for tc in m.tool_calls],
                })
            elif m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    def _tools(self, tools: Optional[List[ToolSpec]]):
        if not tools:
            return None
        return [{"type": "function",
                 "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools]

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=1024) -> LLMResponse:
        kwargs = dict(model=self.model, messages=self._to_openai(messages),
                      temperature=temperature, max_tokens=max_tokens)
        if tools:
            kwargs["tools"] = self._tools(tools)
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        tool_calls = []
        for tc in (choice.message.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = resp.usage
        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            usage=Usage(usage.prompt_tokens, usage.completion_tokens) if usage else Usage(),
            stop_reason="tool_use" if tool_calls else "end",
            model=self.model,
            raw=resp,
        )

    def stream(self, messages, tools=None, temperature=0.7, max_tokens=1024) -> Iterator[StreamEvent]:
        kwargs = dict(model=self.model, messages=self._to_openai(messages),
                      temperature=temperature, max_tokens=max_tokens, stream=True,
                      stream_options={"include_usage": True})
        if tools:
            kwargs["tools"] = self._tools(tools)
        usage = Usage()
        for chunk in self._client.chat.completions.create(**kwargs):
            if chunk.usage:
                usage = Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield StreamEvent(type="text", text=delta.content)
        yield StreamEvent(type="done", usage=usage)
