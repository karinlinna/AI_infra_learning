"""第 6 课：RAG（检索增强生成）—— 让 Agent 拥有外部知识

知识点：
  - 为什么需要 RAG：模型的知识有截止日期、不知道你的私有资料、且会"幻觉"。
    RAG 把"相关资料"检索出来塞进提示，让模型"开卷答题"。
  - RAG 五步：切块(chunk) -> 向量化(embed) -> 建库(index) -> 检索(retrieve) -> 拼接生成(augment)。
  - 检索质量决定上限：切块大小、重叠、top-k、embedding 质量都是可调旋钮。

本课用离线哈希 embedding 演示完整链路；换真实 embedding + 向量库即为生产形态。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmkit import Message, get_llm
from llmkit.rag import VectorStore, chunk_text

# 假装这是公司内部知识库（模型训练时不可能见过）
KNOWLEDGE = """
llmkit 是本课程自研的教学用 Agent 框架。它的核心是 LLMProvider 抽象，支持一键切换后端。
llmkit 的默认后端是离线 Mock，无需 API Key 即可运行所有课程示例。
llmkit 的 Agent 类实现了 ReAct 循环，通过 max_steps 参数防止无限工具调用。
llmkit 的缓存通过 CachingProvider 代理实现，命中相同请求可省去一次模型调用。
llmkit 的 RAG 模块用哈希 embedding 做离线演示，生产中可替换为真实向量库。
"""


def build_index() -> VectorStore:
    store = VectorStore()
    for chunk in chunk_text(KNOWLEDGE, size=60, overlap=10):
        store.add(chunk)
    return store


def rag_answer(llm, store: VectorStore, question: str):
    print(f"\n问题: {question}")
    # 1) 检索
    hits = store.search(question, k=2)
    print("  检索到的相关片段:")
    for doc, score in hits:
        print(f"    ({score:.3f}) {doc.text.strip()}")

    # 2) 拼接：把检索结果作为"上下文"注入提示
    context = "\n".join(d.text.strip() for d, _ in hits)
    messages = [
        Message.system("只根据下面提供的资料回答，资料没有就说不知道，不要编造。\n\n资料:\n" + context),
        Message.user(question),
    ]
    # 3) 生成
    resp = llm.chat(messages)
    print("  最终回答:", resp.content)


def demo_without_vs_with_rag(llm, store):
    print("\n=== 无 RAG（模型不知道私有知识） vs 有 RAG ===")
    q = "llmkit 用什么防止无限工具调用？"
    print("【无 RAG】", llm.chat([Message.user(q)]).content)
    print("【有 RAG】", end="")
    rag_answer(llm, store, q)


if __name__ == "__main__":
    llm = get_llm()
    print(f"当前后端: {llm.name} / {llm.model}")
    store = build_index()
    print(f"知识库已建立，共 {len(store.docs)} 个片段。")

    rag_answer(llm, store, "llmkit 的默认后端是什么？")
    rag_answer(llm, store, "llmkit 的缓存是怎么实现的？")
    demo_without_vs_with_rag(llm, store)
