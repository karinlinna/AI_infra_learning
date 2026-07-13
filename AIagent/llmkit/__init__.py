"""llmkit —— 一个用于教学的、可切换后端的极简 Agent 基础库。

对外导出最常用的符号，让课程 `from llmkit import ...` 一步到位。
"""

from .base import LLMProvider
from .cache import CachingProvider, wrap_with_cache
from .factory import get_llm
from .retry import with_retry
from .tools import Tool, ToolRegistry, tool
from .types import (
    LLMResponse,
    Message,
    Role,
    StreamEvent,
    ToolCall,
    ToolSpec,
    Usage,
)

__all__ = [
    "LLMProvider",
    "get_llm",
    "Message",
    "Role",
    "ToolCall",
    "ToolSpec",
    "LLMResponse",
    "StreamEvent",
    "Usage",
    "tool",
    "Tool",
    "ToolRegistry",
    "wrap_with_cache",
    "CachingProvider",
    "with_retry",
]
