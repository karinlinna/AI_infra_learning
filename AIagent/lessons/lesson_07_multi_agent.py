"""第 7 课：多智能体（Multi-Agent）—— 分工协作

知识点：
  - 为什么要多个 Agent：单个 Agent 塞太多职责会"样样通样样松"。
    把复杂任务拆给各有专长、各有工具的 Agent，更可控、可测、可复用。
  - 两种经典编排：
        * 流水线(Pipeline)：Agent A 的输出喂给 Agent B（研究 -> 写作 -> 审校）；
        * 协调者(Coordinator/Router)：一个"主管"Agent 根据任务把活派给合适的下属。
  - 通信方式：最朴素也最稳的，就是"用自然语言/结构化文本传递中间产物"。

本课演示一个"研究员 -> 作家 -> 审校"的流水线，全部复用 llmkit.agent.Agent。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmkit import ToolRegistry, get_llm, tool
from llmkit.agent import Agent


@tool
def search_web(query: str) -> str:
    """联网搜索资料（此处为离线假数据）"""
    return f"关于「{query}」的三条要点：1) 起源于2017年；2) 核心是注意力机制；3) 已成为主流。"


def build_pipeline(llm):
    """三个专职 Agent，各自 system 提示不同、职责不同。"""
    researcher = Agent(
        llm,
        tools=ToolRegistry(search_web),
        system="你是研究员。使用搜索工具收集资料，只输出要点清单，不做润色。",
        max_steps=3,
    )
    writer = Agent(
        llm,
        system="你是作家。把给你的要点扩写成一段通顺连贯的中文短文。",
        max_steps=1,
    )
    reviewer = Agent(
        llm,
        system="你是审校。检查文章是否通顺、有无明显问题，输出最终定稿。",
        max_steps=1,
    )
    return researcher, writer, reviewer


def run_pipeline(researcher, writer, reviewer, topic: str):
    print(f"\n=== 流水线协作：主题「{topic}」 ===")

    print("\n[研究员] 收集资料中...")
    notes = researcher.run(f"搜集关于「{topic}」的资料要点。").answer
    print("  产出:", notes)

    print("\n[作家] 扩写成文...")
    draft = writer.run(f"请根据以下要点写一段短文：\n{notes}").answer
    print("  产出:", draft)

    print("\n[审校] 定稿...")
    final = reviewer.run(f"请审校并定稿：\n{draft}").answer
    print("  最终定稿:", final)
    return final


def demo_coordinator(llm):
    """协调者模式：主管把不同类型的问题路由给不同下属。"""
    print("\n=== 协调者模式（Router）===")

    def route(question: str) -> str:
        # 真实项目里由"主管 Agent"用模型判断路由；这里用关键词演示路由骨架
        if any(k in question for k in ["天气", "温度"]):
            return "天气专家"
        if any(k in question for k in ["计算", "多少", "加", "乘"]):
            return "数学专家"
        return "通用助手"

    for q in ["北京天气如何？", "3 乘以 7 等于多少？", "给我讲个笑话"]:
        print(f"  问题「{q}」-> 路由到【{route(q)}】")


if __name__ == "__main__":
    llm = get_llm()
    print(f"当前后端: {llm.name} / {llm.model}")
    r, w, v = build_pipeline(llm)
    run_pipeline(r, w, v, "Transformer 架构")
    demo_coordinator(llm)
