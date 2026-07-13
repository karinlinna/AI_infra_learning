"""第 3 课：工具调用（Function Calling）

知识点：
  - 工具 = 把模型的"意图"翻译成对真实世界的操作（查天气、算数、查库…）；
  - 完整的一次工具调用握手：
      1) 我们把工具 schema 连同问题给模型；
      2) 模型返回 tool_use（要调用哪个工具、参数是什么）；
      3) 我们执行工具，把结果作为 role=tool 的消息回传；
      4) 模型基于工具结果给出最终答复。
  - @tool 装饰器如何从函数自动生成 schema。

这一课是"从聊天机器人迈向 Agent"的关键一步。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmkit import Message, ToolRegistry, get_llm, tool


# ---- 定义几个工具：普通函数 + @tool 即可 ----
@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气"""
    fake = {"北京": "晴 26℃", "上海": "多云 24℃", "广州": "小雨 29℃"}
    return fake.get(city, f"{city} 天气未知")


@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式，例如 '2+3*4'"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # 教学演示；生产勿直接 eval
    except Exception as e:
        return f"计算错误: {e}"


def demo_tool_schema():
    print("\n=== 1. @tool 自动生成的 schema ===")
    import json
    print(json.dumps(get_weather.spec.__dict__, ensure_ascii=False, indent=2))


def demo_single_tool_handshake(llm):
    print("\n=== 2. 一次完整的工具调用握手 ===")
    registry = ToolRegistry(get_weather, calculator)
    messages = [
        Message.system("你可以使用工具来回答问题。"),
        Message.user("北京现在天气怎么样？"),
    ]

    # 第 1 步：模型决定调用工具
    resp = llm.chat(messages, tools=registry.specs())
    if not resp.wants_tools:
        print("模型直接作答:", resp.content)
        return
    tc = resp.tool_calls[0]
    print(f"模型请求调用工具: {tc.name}({tc.arguments})")

    # 第 2 步：我们执行工具
    result = registry.call(tc.name, tc.arguments)
    print(f"工具执行结果: {result}")

    # 第 3 步：把工具结果回传，模型给最终答复
    messages.append(Message.assistant(tool_calls=[tc]))
    messages.append(Message.tool(result, tc.id, tc.name))
    final = llm.chat(messages, tools=registry.specs())
    print(f"模型最终答复: {final.content}")


if __name__ == "__main__":
    llm = get_llm()
    print(f"当前后端: {llm.name} / {llm.model}")
    demo_tool_schema()
    demo_single_tool_handshake(llm)
