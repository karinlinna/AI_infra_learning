"""第 1 课：裸调用 LLM —— 一切的起点

知识点：
  - LLM 本质是"给定对话，预测下一段文本"的函数；
  - system / user / assistant 三种角色的含义；
  - 无状态：API 不记得上一轮，历史要每次自己带上；
  - 流式输出（打字机效果）从何而来。

运行：
  python lessons/lesson_01_bare_call.py
默认用离线 Mock。想接真实模型：设置 LLM_PROVIDER=anthropic 或 openai。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 让脚本能 import llmkit

from llmkit import Message, get_llm


def demo_single_turn(llm):
    print("\n=== 1. 最简单的一次调用 ===")
    messages = [
        Message.system("你是一个简洁的中文助手。"),
        Message.user("用一句话解释什么是大语言模型。"),
    ]
    resp = llm.chat(messages, max_tokens=200)
    print("模型回复:", resp.content)
    print("用量:", resp.usage, "| 停止原因:", resp.stop_reason)


def demo_multi_turn(llm):
    """多轮对话：API 无状态，所以要手动维护 history 列表。"""
    print("\n=== 2. 多轮对话（自己维护历史） ===")
    history = [Message.system("你是助手，请记住用户告诉你的信息。")]

    def ask(text: str):
        history.append(Message.user(text))
        resp = llm.chat(history)
        history.append(Message.assistant(resp.content))  # 把模型回复也存回历史
        print(f"用户: {text}\n助手: {resp.content}\n")

    ask("我叫李雷。")
    ask("我叫什么名字？")  # 只有把第一轮历史带上，模型才可能答对


def demo_streaming(llm):
    print("=== 3. 流式输出（打字机效果） ===")
    print("助手: ", end="", flush=True)
    for ev in llm.stream([Message.user("讲讲流式输出的好处。")]):
        if ev.type == "text":
            print(ev.text, end="", flush=True)
        elif ev.type == "done":
            print(f"\n[流结束，用量: {ev.usage}]")


if __name__ == "__main__":
    llm = get_llm()
    print(f"当前后端: {llm.name} / {llm.model}")
    demo_single_turn(llm)
    demo_multi_turn(llm)
    demo_streaming(llm)
