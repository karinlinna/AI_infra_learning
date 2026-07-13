"""第 5 课：记忆（Memory）—— 让 Agent 记住东西

知识点：
  - 短期记忆：就是对话历史(messages)本身。但上下文窗口有限、且越长越贵越慢；
  - 上下文管理：当历史过长时的两类策略
        * 截断/滑动窗口（保留最近 N 轮）——简单，可能丢信息；
        * 摘要压缩（把旧对话让模型总结成一段）——省 token，保留要点。
  - 长期记忆：跨会话持久化（写入文件/DB/向量库），下次会话再取回。

本课演示：滑动窗口、摘要压缩、一个极简的键值长期记忆。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmkit import Message, get_llm
from llmkit.base import LLMProvider


# ---------- 短期记忆：滑动窗口 ----------
def sliding_window(history: List[Message], keep_last: int = 4) -> List[Message]:
    """保留 system 提示 + 最近 keep_last 条消息。"""
    system = [m for m in history if m.role == "system"][:1]
    rest = [m for m in history if m.role != "system"]
    return system + rest[-keep_last:]


# ---------- 短期记忆：摘要压缩 ----------
def summarize_history(llm: LLMProvider, history: List[Message]) -> Message:
    """把一段历史压成一条 system 摘要消息，替换掉冗长原文。"""
    convo = "\n".join(f"{m.role}: {m.content}" for m in history if m.role != "system")
    resp = llm.chat([
        Message.system("把下面的对话压缩成不超过 50 字的要点，只保留关键事实。"),
        Message.user(convo),
    ])
    return Message.system(f"[历史摘要] {resp.content}")


# ---------- 长期记忆：跨会话持久化 ----------
class LongTermMemory:
    """最简单的长期记忆：一个 JSON 文件当键值库。真实项目可换成向量库/数据库。"""

    def __init__(self, path: str = ".memory.json"):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def remember(self, key: str, value: str):
        self.data[key] = value
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2))

    def recall(self, key: str) -> str:
        return self.data.get(key, "")


def demo_short_term(llm):
    print("\n=== 1. 短期记忆：滑动窗口 vs 摘要压缩 ===")
    history = [Message.system("你是助手。")]
    for i in range(1, 7):
        history.append(Message.user(f"这是第 {i} 条用户消息"))
        history.append(Message.assistant(f"收到第 {i} 条"))

    windowed = sliding_window(history, keep_last=4)
    print(f"原始 {len(history)} 条 -> 滑动窗口保留 {len(windowed)} 条：",
          [f"{m.role}:{m.content}" for m in windowed])

    summary = summarize_history(llm, history)
    print("摘要压缩后的单条记忆:", summary.content)


def demo_long_term(llm):
    print("\n=== 2. 长期记忆：跨会话记住用户偏好 ===")
    mem = LongTermMemory()
    mem.remember("用户偏好语言", "中文")
    mem.remember("用户所在城市", "杭州")

    # 新会话开始：把长期记忆注入 system 提示
    profile = "；".join(f"{k}={v}" for k, v in mem.data.items())
    resp = llm.chat([
        Message.system(f"已知用户档案：{profile}。据此个性化回答。"),
        Message.user("推荐个周末去处。"),
    ])
    print("注入长期记忆后的回答:", resp.content)
    print(f"（记忆已持久化到 {mem.path}）")


if __name__ == "__main__":
    llm = get_llm()
    print(f"当前后端: {llm.name} / {llm.model}")
    demo_short_term(llm)
    demo_long_term(llm)
