"""get_llm() —— 一行代码切换后端。

课程里所有地方都用 get_llm() 拿模型，读取环境变量 LLM_PROVIDER：
    - mock（默认，离线）
    - anthropic
    - openai（也可指向国产/本地兼容端点）

这就是"可切换 LLM"落到实处的样子：切换成本 = 改一个环境变量。
"""

from __future__ import annotations

import os
from typing import Optional

from .base import LLMProvider


def get_llm(provider: Optional[str] = None, **kwargs) -> LLMProvider:
    provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()

    if provider == "mock":
        from .providers.mock import MockProvider
        return MockProvider(**kwargs)
    if provider == "anthropic":
        from .providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(**kwargs)
    if provider in ("openai", "compatible"):
        from .providers.openai_provider import OpenAIProvider
        return OpenAIProvider(**kwargs)

    raise ValueError(f"未知的 LLM_PROVIDER: {provider!r}（可选 mock/anthropic/openai）")
