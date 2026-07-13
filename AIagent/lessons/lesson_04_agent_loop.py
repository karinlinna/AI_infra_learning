"""第 4 课：Agent 循环 —— 从"单次工具调用"到"自主多步"

知识点：
  - Agent = 模型 + 工具 + 一个"思考→行动→观察"的循环；
  - 为什么需要循环：一个任务可能要连续调用多个工具（先查汇率，再算总价…）；
  - 生产必备的护栏：max_steps 步数上限、工具异常回传、事件回调（可观测性）；
  - ReAct 思想：模型交替产出"推理"和"行动"，用工具观察结果再决定下一步。

本课直接复用 llmkit.agent.Agent（前几课我们手写握手，现在把它沉淀成可复用组件）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmkit import ToolRegistry, get_llm, tool
from llmkit.agent import Agent


@tool
def search_flight_price(city: str) -> str:
    """查询到某城市的机票价格（元）"""
    prices = {"东京": "3200", "巴黎": "5800", "纽约": "6500"}
    return prices.get(city, "4000")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算错误: {e}"


def pretty_event(ev: str, data: dict):
    """把 Agent 内部事件打印出来，直观展示循环的每一步。"""
    if ev == "model":
        tc = data["tool_calls"]
        note = f"决定调用工具 {tc}" if tc else f"给出答复：{data['content'][:40]}"
        print(f"  [第{data['step']}步·思考] {note}")
    elif ev == "tool":
        print(f"  [第{data['step']}步·行动] {data['name']}({data['args']}) -> {data['result']}")
    elif ev == "halt":
        print(f"  [终止] {data['reason']}")


if __name__ == "__main__":
    llm = get_llm()
    print(f"当前后端: {llm.name} / {llm.model}\n")

    registry = ToolRegistry(search_flight_price, calculator)
    agent = Agent(
        llm,
        tools=registry,
        system="你是旅行助手，可用工具查机票价、做计算。请一步步完成用户请求。",
        max_steps=6,
        on_event=pretty_event,
    )

    print("=== Agent 自主解决多步任务 ===")
    result = agent.run("帮我查一下去东京的机票价格。")
    print(f"\n最终答案: {result.answer}")
    print(f"共用 {result.steps} 步，累计 token: {result.usage.total_tokens}")
