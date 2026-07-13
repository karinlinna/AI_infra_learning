"""一个可复用的 Agent 循环（ReAct 风格：思考 → 行动 → 观察 → 再思考）。

Agent 的本质，就是把"模型调用 + 工具执行"放进一个循环里，
直到模型不再请求工具、给出最终答复为止。所有花哨的 Agent 框架，内核都是这个循环。

关键的工程约束（生产必备）：
  - max_steps：硬性步数上限，防止模型陷入无限工具调用；
  - 每一步都把 assistant(含 tool_use) 与 tool 结果按顺序追加进历史；
  - 可选 on_event 回调，用于可观测性（打印/埋点/记账）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .base import LLMProvider
from .tools import ToolRegistry
from .types import Message, Usage


@dataclass
class AgentResult:
    answer: str
    steps: int
    usage: Usage
    history: List[Message] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        llm: LLMProvider,
        tools: Optional[ToolRegistry] = None,
        system: str = "你是一个会使用工具解决问题的智能助手。",
        max_steps: int = 6,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ):
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.system = system
        self.max_steps = max_steps
        self.on_event = on_event or (lambda ev, data: None)

    def run(self, user_input: str, history: Optional[List[Message]] = None) -> AgentResult:
        messages: List[Message] = list(history) if history else [Message.system(self.system)]
        messages.append(Message.user(user_input))
        total = Usage()
        specs = self.tools.specs() or None

        for step in range(1, self.max_steps + 1):
            resp = self.llm.chat(messages, tools=specs)
            total = total + resp.usage
            self.on_event("model", {"step": step, "content": resp.content,
                                     "tool_calls": [t.name for t in resp.tool_calls]})

            # 模型不再要工具 → 收尾
            if not resp.wants_tools:
                messages.append(Message.assistant(resp.content))
                return AgentResult(resp.content, step, total, messages)

            # 把模型的工具调用意图记进历史
            messages.append(Message.assistant(resp.content, tool_calls=resp.tool_calls))

            # 逐个执行工具，结果按 tool_call_id 回传
            for tc in resp.tool_calls:
                result = self.tools.call(tc.name, tc.arguments)
                self.on_event("tool", {"step": step, "name": tc.name,
                                       "args": tc.arguments, "result": result})
                messages.append(Message.tool(result, tc.id, tc.name))

        # 触达步数上限，安全退出（生产中应告警）
        self.on_event("halt", {"reason": "max_steps", "steps": self.max_steps})
        return AgentResult("(已达到最大步数，未能得出最终答案)", self.max_steps, total, messages)
