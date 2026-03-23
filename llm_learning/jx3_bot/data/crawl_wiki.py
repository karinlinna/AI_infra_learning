"""
剑网三百科/攻略站数据爬取

爬取来源：
1. 剑网三官方百科（jx3.xoyo.com）
2. 各门派技能、装备、副本等结构化数据

使用方式：
    python crawl_wiki.py --output data/raw/wiki_data.json
"""

import json
import time
import argparse
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# 剑网三门派列表
SCHOOLS = [
    "天策", "万花", "纯阳", "七秀", "少林",
    "藏剑", "五毒", "唐门", "明教", "丐帮",
    "苍云", "长歌", "霸刀", "蓬莱", "凌雪",
    "衍天", "药宗", "刀宗", "万灵", "北天药宗",
]

# 萌新常见主题
TOPICS = [
    "新手入门", "门派选择", "装备获取", "副本攻略",
    "PVP入门", "日常任务", "赚钱方法", "宠物奇遇",
    "阵营任务", "插件推荐", "宏设置", "配装推荐",
]


def fetch_page(url: str, retry: int = 3) -> str | None:
    """获取网页内容，带重试机制"""
    for attempt in range(retry):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.encoding = resp.apparent_encoding
            if resp.status_code == 200:
                return resp.text
            print(f"  HTTP {resp.status_code}: {url}")
        except requests.RequestException as e:
            print(f"  请求失败 (尝试 {attempt + 1}/{retry}): {e}")
            time.sleep(2)
    return None


def parse_article(html: str, url: str) -> dict | None:
    """从 HTML 中提取文章标题和正文"""
    soup = BeautifulSoup(html, "lxml")

    # 提取标题
    title_tag = soup.find("h1") or soup.find("title")
    if not title_tag:
        return None
    title = title_tag.get_text(strip=True)

    # 提取正文段落
    content_parts = []
    for tag in soup.find_all(["p", "li", "td", "h2", "h3", "h4"]):
        text = tag.get_text(strip=True)
        if len(text) > 10:  # 过滤太短的内容
            content_parts.append(text)

    if not content_parts:
        return None

    content = "\n".join(content_parts)

    # 过滤太短的页面
    if len(content) < 100:
        return None

    return {
        "title": title,
        "content": content,
        "url": url,
        "source": "wiki",
    }


def search_baidu_for_jx3(query: str, max_results: int = 5) -> list[str]:
    """
    通过百度搜索获取剑网三相关页面链接
    注意：实际使用时可能需要处理反爬
    """
    search_url = "https://www.baidu.com/s"
    params = {"wd": f"剑网三 {query}", "rn": max_results}
    html = fetch_page(f"{search_url}?wd={params['wd']}&rn={params['rn']}")
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    urls = []
    for a_tag in soup.select("h3.t a"):
        href = a_tag.get("href")
        if href:
            urls.append(href)
    return urls[:max_results]


def crawl_search_results(queries: list[str], max_per_query: int = 5) -> list[dict]:
    """基于搜索查询爬取数据"""
    all_data = []
    seen_urls = set()

    for query in queries:
        print(f"搜索: 剑网三 {query}")
        urls = search_baidu_for_jx3(query, max_per_query)

        for url in urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)

            html = fetch_page(url)
            if not html:
                continue

            article = parse_article(html, url)
            if article:
                article["query"] = query
                all_data.append(article)
                print(f"  ✓ {article['title'][:40]}")

            time.sleep(1)  # 礼貌爬取

    return all_data


def build_school_queries() -> list[str]:
    """生成门派相关的搜索查询"""
    queries = []
    for school in SCHOOLS:
        queries.extend([
            f"{school} 新手入门",
            f"{school} 配装推荐",
            f"{school} PVE手法",
            f"{school} 宏推荐",
        ])
    return queries


def build_topic_queries() -> list[str]:
    """生成通用主题搜索查询"""
    queries = []
    for topic in TOPICS:
        queries.append(f"{topic} 2024 2025")
    return queries


def main():
    parser = argparse.ArgumentParser(description="爬取剑网三百科数据")
    parser.add_argument(
        "--output", type=str, default="data/raw/wiki_data.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--max-per-query", type=int, default=5,
        help="每个查询最多爬取的页面数"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 构建搜索查询
    queries = build_school_queries() + build_topic_queries()
    print(f"共 {len(queries)} 个搜索查询")

    # 爬取
    data = crawl_search_results(queries, args.max_per_query)

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n完成！共爬取 {len(data)} 篇文章，保存到 {output_path}")


if __name__ == "__main__":
    main()
