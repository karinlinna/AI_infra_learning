"""缓存工具，供"优化与性能调优"课使用。

包含两种缓存思路：
  1. ExactCache：精确缓存——相同请求直接返回上次结果（key = 消息内容哈希）。
     命中率取决于请求是否逐字节相同，适合确定性、重复的调用。
  2. 语义缓存的思想在 lesson_08 里用向量相似度演示（此处只放精确缓存这一简单可靠的基线）。

另外提供 wrap_with_cache()，把任意 provider 透明地套上缓存，
体现"横切关注点用装饰器/代理实现，不污染业务代码"的工程原则。
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional

from .base import LLMProvider
from .types import LLMResponse, Message, ToolSpec


def _key(messages: List[Message], tools: Optional[List[ToolSpec]], model: str) -> str:
    payload = {
        "model": model,
        "messages": [(m.role, m.content) for m in messages],
        "tools": [t.name for t in (tools or [])],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CachingProvider(LLMProvider):
    """代理模式：包住一个真实 provider，对 chat 做精确缓存。"""

    def __init__(self, inner: LLMProvider):
        self.inner = inner
        self.name = f"cached({inner.name})"
        self.model = inner.model
        self._store: Dict[str, LLMResponse] = {}
        self.hits = 0
        self.misses = 0

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=1024) -> LLMResponse:
        k = _key(messages, tools, self.model)
        if k in self._store:
            self.hits += 1
            return self._store[k]
        self.misses += 1
        resp = self.inner.chat(messages, tools, temperature, max_tokens)
        self._store[k] = resp
        return resp

    def stream(self, *a, **kw):
        return self.inner.stream(*a, **kw)

    def count_tokens(self, messages):
        return self.inner.count_tokens(messages)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def wrap_with_cache(inner: LLMProvider) -> CachingProvider:
    return CachingProvider(inner)
