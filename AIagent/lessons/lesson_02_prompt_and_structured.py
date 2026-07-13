"""第 2 课：提示工程 与 结构化输出

知识点：
  - 提示工程的几根支柱：角色设定、清晰指令、少样本示例(few-shot)、约束输出格式；
  - 为什么要"结构化输出"：Agent 系统里，模型的输出往往要喂给下游程序，
    自由文本难以解析，JSON / 固定 schema 才可靠；
  - 一个朴素但通用的做法：要求模型只输出 JSON，再解析 + 校验 + 出错重试。

（真实生产中，Anthropic/OpenAI 都有原生"结构化输出/严格模式"，比让模型自觉输出 JSON 更稳；
  这里演示通用原理，任何模型后端都适用。）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmkit import Message, get_llm
from llmkit.base import LLMProvider


def demo_prompt_techniques(llm):
    print("\n=== 1. 提示工程对比：模糊指令 vs 结构化指令 ===")
    vague = [Message.user("分析这句话的情感：这家餐厅太棒了，我还会再来！")]
    print("模糊提问 ->", llm.chat(vague).content)

    # 好提示 = 角色 + 明确任务 + 输出约束 + 少样本
    good = [
        Message.system("你是情感分析引擎。只输出 positive / negative / neutral 之一，不要解释。"),
        Message.user("示例：'糟透了' -> negative\n示例：'还行吧' -> neutral\n\n这家餐厅太棒了，我还会再来！"),
    ]
    print("结构化提问 ->", llm.chat(good).content)


def extract_json(text: str) -> dict:
    """从模型输出里稳健地抠出 JSON —— 模型常会包裹 ```json ``` 或前后加话。"""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"输出中找不到 JSON: {text!r}")
    return json.loads(text[start:end + 1])


def structured_call(llm: LLMProvider, user_text: str, schema_hint: str, max_retries: int = 2) -> dict:
    """要求模型按 schema 输出 JSON，解析失败则把错误回灌给模型重试。

    这是"自我修复循环(self-healing loop)"的最小形态，Agent 工程里非常常用。
    """
    messages = [
        Message.system(f"你是抽取引擎。只输出严格符合下述结构的 JSON，不要任何多余文字。\n结构：{schema_hint}"),
        Message.user(user_text),
    ]
    for attempt in range(max_retries + 1):
        resp = llm.chat(messages)
        try:
            return extract_json(resp.content)
        except (ValueError, json.JSONDecodeError) as e:
            if attempt == max_retries:
                raise
            # 把上一次的错误输出 + 报错信息告诉模型，让它纠正
            messages.append(Message.assistant(resp.content))
            messages.append(Message.user(f"上面的输出无法解析为 JSON（{e}）。请只输出合法 JSON。"))
    raise RuntimeError("unreachable")


def demo_structured(llm):
    print("\n=== 2. 结构化输出 + 解析 + 出错重试 ===")
    schema = '{"name": string, "age": number, "city": string}'
    text = "我叫韩梅梅，今年 28 岁，住在上海。"
    try:
        data = structured_call(llm, text, schema)
        print("解析结果:", data, "| 类型:", type(data).__name__)
    except Exception as e:
        # Mock 不会真输出 JSON，这里演示失败也被优雅处理
        print(f"(离线 Mock 不产出真实 JSON，触发重试后仍失败：{e})")
        print("=> 接真实模型（LLM_PROVIDER=anthropic/openai）即可看到成功解析。")


if __name__ == "__main__":
    llm = get_llm()
    print(f"当前后端: {llm.name} / {llm.model}")
    demo_prompt_techniques(llm)
    demo_structured(llm)
