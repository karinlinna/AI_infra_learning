"""llmkit 的核心数据类型。

设计目标：用一组"provider 无关"的数据结构，把不同厂商（OpenAI / Anthropic / 本地 Mock）
的差异挡在抽象层之外。上层课程只依赖这些类型，永远不直接碰某个 SDK 的原始对象。

这是 Agent 开发的第一个重要工程实践：**面向接口编程，而非面向具体模型**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

# 一条消息的角色。system=系统指令，user=用户，assistant=模型，tool=工具执行结果
Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """模型发起的一次工具调用请求。

    注意 arguments 是已解析好的 dict —— 不同厂商在传输层是 JSON 字符串，
    我们在 provider 内部统一解析，避免上层反复 json.loads()。
    """

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Message:
    """统一的对话消息。"""

    role: Role
    content: str = ""
    # 当 role == "assistant" 且模型决定调用工具时填充
    tool_calls: List[ToolCall] = field(default_factory=list)
    # 当 role == "tool" 时，标明这是对哪一次 tool_call 的响应
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # 工具名，便于日志/追踪

    # --- 便捷构造器：让课程代码读起来像自然语言 ---
    @staticmethod
    def system(content: str) -> "Message":
        return Message(role="system", content=content)

    @staticmethod
    def user(content: str) -> "Message":
        return Message(role="user", content=content)

    @staticmethod
    def assistant(content: str = "", tool_calls: Optional[List[ToolCall]] = None) -> "Message":
        return Message(role="assistant", content=content, tool_calls=tool_calls or [])

    @staticmethod
    def tool(content: str, tool_call_id: str, name: str = "") -> "Message":
        return Message(role="tool", content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class ToolSpec:
    """暴露给模型的工具描述（即 function calling 的 schema）。

    parameters 使用 JSON Schema。好的 description 直接决定模型是否/何时正确调用，
    这是提示工程在工具层面的体现。
    """

    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class Usage:
    """token 用量，用于成本核算与性能观测。"""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass
class LLMResponse:
    """一次（非流式）模型调用的完整结果。"""

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    # 停止原因：end（正常结束）/ tool_use（要求调用工具）/ length（触达 max_tokens）
    stop_reason: str = "end"
    model: str = ""
    raw: Any = None  # 保留原始响应，方便调试，但上层不应依赖它

    @property
    def wants_tools(self) -> bool:
        return self.stop_reason == "tool_use" or bool(self.tool_calls)


@dataclass
class StreamEvent:
    """流式输出中的一个增量事件。

    type: text（正文增量）/ tool（工具调用已确定）/ done（本次结束，携带最终 usage）
    """

    type: Literal["text", "tool", "done"]
    text: str = ""
    tool_call: Optional[ToolCall] = None
    usage: Optional[Usage] = None
