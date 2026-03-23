"""
剑网三社区数据爬取（NGA / 百度贴吧）

爬取来源：
1. NGA 剑网三版块（萌新提问帖）
2. 百度贴吧剑网三吧

使用方式：
    python crawl_community.py --output data/raw/community_data.json
"""

import json
import time
import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_page(url: str, retry: int = 3) -> str | None:
    """获取网页内容"""
    for attempt in range(retry):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.encoding = resp.apparent_encoding
            if resp.status_code == 200:
                return resp.text
        except requests.RequestException as e:
            print(f"  请求失败 (尝试 {attempt + 1}/{retry}): {e}")
            time.sleep(2)
    return None


# ============================================================
# NGA 爬取
# ============================================================

NGA_BASE = "https://bbs.nga.cn"
NGA_JX3_FID = 650  # 剑网三版块 ID（可能需要确认）


def crawl_nga_thread_list(page: int = 1) -> list[dict]:
    """爬取 NGA 剑网三版块帖子列表"""
    url = f"{NGA_BASE}/thread.php?fid={NGA_JX3_FID}&page={page}"
    html = fetch_page(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    threads = []

    for title_tag in soup.select("a.topic"):
        title = title_tag.get_text(strip=True)
        href = title_tag.get("href", "")

        # 过滤：只要包含萌新关键词的帖子
        keywords = ["新手", "萌新", "入门", "求助", "怎么", "如何", "推荐", "选什么"]
        if any(kw in title for kw in keywords):
            threads.append({
                "title": title,
                "url": href if href.startswith("http") else NGA_BASE + href,
            })

    return threads


def crawl_nga_thread_content(url: str) -> dict | None:
    """爬取单个 NGA 帖子的内容"""
    html = fetch_page(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")

    # 提取主楼内容
    post_content = soup.select_one(".postcontent")
    if not post_content:
        return None

    content = post_content.get_text(strip=True)
    if len(content) < 50:
        return None

    # 提取回复（取前几个高质量回复）
    replies = []
    for reply in soup.select(".postcontent")[1:6]:  # 取前 5 个回复
        text = reply.get_text(strip=True)
        if len(text) > 30:
            replies.append(text)

    return {
        "question": content,
        "replies": replies,
        "url": url,
        "source": "nga",
    }


def crawl_nga(max_pages: int = 5) -> list[dict]:
    """爬取 NGA 剑网三萌新帖子"""
    all_data = []
    for page in range(1, max_pages + 1):
        print(f"NGA 第 {page} 页...")
        threads = crawl_nga_thread_list(page)
        print(f"  找到 {len(threads)} 个相关帖子")

        for thread in threads:
            data = crawl_nga_thread_content(thread["url"])
            if data:
                data["title"] = thread["title"]
                all_data.append(data)
                print(f"  ✓ {thread['title'][:40]}")
            time.sleep(1)

    return all_data


# ============================================================
# 百度贴吧爬取
# ============================================================

TIEBA_BASE = "https://tieba.baidu.com"


def crawl_tieba_list(kw: str = "剑网3", page: int = 0) -> list[dict]:
    """爬取贴吧帖子列表"""
    url = f"{TIEBA_BASE}/f?kw={kw}&pn={page * 50}"
    html = fetch_page(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    threads = []

    for item in soup.select(".j_thread_list"):
        title_tag = item.select_one(".threadlist_title a")
        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)
        href = title_tag.get("href", "")

        keywords = ["新手", "萌新", "入门", "求助", "怎么", "如何", "推荐", "选什么", "求教"]
        if any(kw in title for kw in keywords):
            threads.append({
                "title": title,
                "url": TIEBA_BASE + href,
            })

    return threads


def crawl_tieba_thread(url: str) -> dict | None:
    """爬取单个贴吧帖子"""
    html = fetch_page(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")

    posts = []
    for post in soup.select(".d_post_content"):
        text = post.get_text(strip=True)
        if len(text) > 20:
            posts.append(text)

    if len(posts) < 2:  # 至少要有问题和一个回复
        return None

    return {
        "question": posts[0],
        "replies": posts[1:6],  # 取前 5 个回复
        "url": url,
        "source": "tieba",
    }


def crawl_tieba(max_pages: int = 5) -> list[dict]:
    """爬取百度贴吧剑网三吧"""
    all_data = []
    for page in range(max_pages):
        print(f"贴吧 第 {page + 1} 页...")
        threads = crawl_tieba_list("剑网3", page)
        print(f"  找到 {len(threads)} 个相关帖子")

        for thread in threads:
            data = crawl_tieba_thread(thread["url"])
            if data:
                data["title"] = thread["title"]
                all_data.append(data)
                print(f"  ✓ {thread['title'][:40]}")
            time.sleep(1)

    return all_data


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="爬取剑网三社区数据")
    parser.add_argument(
        "--output", type=str, default="data/raw/community_data.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--max-pages", type=int, default=5,
        help="每个来源最多爬取的页数"
    )
    parser.add_argument(
        "--source", type=str, choices=["all", "nga", "tieba"], default="all",
        help="爬取来源"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_data = []

    if args.source in ("all", "nga"):
        print("=" * 50)
        print("爬取 NGA 剑网三版块")
        print("=" * 50)
        all_data.extend(crawl_nga(args.max_pages))

    if args.source in ("all", "tieba"):
        print("=" * 50)
        print("爬取 百度贴吧 剑网三吧")
        print("=" * 50)
        all_data.extend(crawl_tieba(args.max_pages))

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n完成！共爬取 {len(all_data)} 个问答帖，保存到 {output_path}")


if __name__ == "__main__":
    main()
