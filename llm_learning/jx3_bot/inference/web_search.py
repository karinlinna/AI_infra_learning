"""
联网搜索模块

提供实时搜索能力，用于回答涉及最新游戏内容的问题。

特性：
- 使用 DuckDuckGo 免费搜索 API（无需 API Key）
- 自动判断是否需要联网搜索
- 提取搜索结果摘要作为上下文

使用方式：
    from web_search import should_search, search_and_summarize
"""

from __future__ import annotations

import re


# 触发联网搜索的关键词
SEARCH_KEYWORDS = [
    "最新", "更新", "新版本", "新活动", "维护", "公告",
    "赛季", "新赛季", "什么时候", "开服", "合服",
    "新门派", "新副本", "新地图", "周年庆",
    "现在", "目前", "当前", "今天",
]


def should_search(question: str) -> bool:
    """判断问题是否需要联网搜索"""
    return any(kw in question for kw in SEARCH_KEYWORDS)


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    使用 DuckDuckGo 搜索

    返回:
        [{"title": "...", "body": "...", "href": "..."}, ...]
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region="cn-zh", max_results=max_results))
            return results
    except ImportError:
        print("未安装 duckduckgo-search，请运行: pip install duckduckgo-search")
        return []
    except Exception as e:
        print(f"搜索失败: {e}")
        return []


def extract_context(results: list[dict], max_length: int = 1500) -> str:
    """从搜索结果中提取上下文"""
    if not results:
        return ""

    parts = []
    total_len = 0
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        text = f"【{title}】{body}"
        if total_len + len(text) > max_length:
            break
        parts.append(text)
        total_len += len(text)

    return "\n".join(parts)


def search_and_summarize(question: str) -> str | None:
    """
    搜索并返回可用于模型 prompt 的上下文

    返回 None 表示不需要搜索或搜索无结果
    """
    if not should_search(question):
        return None

    query = f"剑网三 {question}"
    # 去掉一些无用的修饰词
    query = re.sub(r"(请问|你好|谢谢|帮我|告诉我)", "", query)

    results = web_search(query)
    context = extract_context(results)

    if not context:
        return None

    return context
