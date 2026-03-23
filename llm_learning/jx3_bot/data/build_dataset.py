"""
剑网三训练数据构建

功能：
1. 读取爬取的原始数据（wiki + community）
2. 清洗和去重
3. 调用 AI（Claude API）批量生成高质量 Q&A 训练对
4. 输出 Qwen Chat 格式的 JSONL 数据集

使用方式：
    # 从原始数据生成训练集
    python build_dataset.py --wiki data/raw/wiki_data.json --community data/raw/community_data.json --output data/train.jsonl

    # 仅从手动编写的种子数据生成（不需要爬虫数据）
    python build_dataset.py --seed-only --output data/train.jsonl
"""

import json
import hashlib
import argparse
from pathlib import Path

# ============================================================
# 系统 Prompt（定义机器人人设）
# ============================================================

SYSTEM_PROMPT = (
    "你是「剑三小助手」，一个专门帮助剑网三萌新的游戏助手。"
    "你熟悉剑网三的所有门派、技能、装备、副本、PVP、日常玩法。"
    "回答要求：\n"
    "1. 语气友好、耐心，像老玩家带萌新\n"
    "2. 回答准确，基于游戏实际机制\n"
    "3. 给出具体可操作的建议\n"
    "4. 适当使用游戏术语但要解释含义\n"
    "5. 如果不确定，诚实说明并建议查阅官方资料"
)


# ============================================================
# 种子数据（手动编写的高质量示例）
# ============================================================

SEED_QA_PAIRS = [
    {
        "question": "我是新手，应该选什么门派？",
        "answer": (
            "欢迎入坑剑三！门派选择主要看你喜欢什么玩法：\n\n"
            "**远程输出（推荐新手）**：\n"
            "- 气纯（纯阳·气宗）：操作简单，输出稳定，新手友好\n"
            "- 花间（万花·花间游）：有控制有输出，团队需求大\n\n"
            "**近战输出**：\n"
            "- 藏剑：手感流畅，输出不错\n"
            "- 霸刀：操作简单粗暴，适合喜欢近战的新手\n\n"
            "**治疗（排队快）**：\n"
            "- 离经（万花·离经易道）：最受欢迎的奶妈\n"
            "- 云裳（七秀·云裳心经）：操作相对简单\n\n"
            "**坦克（排队最快）**：\n"
            "- 铁牛（天策·铁骨衣）：最传统的坦克\n"
            "- 明尊（明教·明尊琉璃体）：攻防一体\n\n"
            "建议先选一个远程输出练手，熟悉游戏后再尝试其他定位。"
            "最重要的是选你觉得帅/美的门派，毕竟这是个时装游戏😄"
        ),
    },
    {
        "question": "剑网三怎么赚钱？",
        "answer": (
            "萌新赚金的几个主要途径：\n\n"
            "**日常稳定收入**：\n"
            "1. 每日/每周任务：日常、周常副本掉落装备分解\n"
            "2. 战场日常：完成后有金币奖励\n"
            "3. 生活技能：采集/制造，前期投入后期收益稳定\n\n"
            "**中期赚钱**：\n"
            "1. 刷副本：高级副本掉落值钱的装备和材料\n"
            "2. 奇遇：部分奇遇奖励可以交易\n"
            "3. 交易行倒卖：了解市场行情后低买高卖\n\n"
            "**小技巧**：\n"
            "- 刚入游戏先别乱花钱，攒着\n"
            "- 加入帮会，帮会有各种福利\n"
            "- 关注游戏活动，活动期间赚钱机会多\n"
            "- 多个角色可以多倍日常收益"
        ),
    },
    {
        "question": "PVE 和 PVP 是什么意思？",
        "answer": (
            "这是 MMO 游戏里的两个核心玩法：\n\n"
            "**PVE**（Player vs Environment，玩家对抗环境）：\n"
            "- 就是打副本、做任务这些\n"
            "- 和队友一起挑战 BOSS\n"
            "- 剑三里包括：5人本、10人本、25人本等\n"
            "- 重视输出循环、走位、团队配合\n\n"
            "**PVP**（Player vs Player，玩家对抗玩家）：\n"
            "- 和其他玩家战斗\n"
            "- 剑三里包括：竞技场（2v2/3v3）、战场、大攻防等\n"
            "- 重视技能释放时机、控制链、临场反应\n\n"
            "新手建议先从 PVE 入手，熟悉技能后再去 PVP。"
            "PVE 装备和 PVP 装备是分开的体系哦。"
        ),
    },
    {
        "question": "什么是奇遇？怎么触发？",
        "answer": (
            "奇遇是剑三的特色系统，类似隐藏任务，完成后有丰厚奖励。\n\n"
            "**奇遇类型**：\n"
            "- 宠物奇遇：获得稀有宠物\n"
            "- 外观奇遇：获得特殊外观/挂件\n"
            "- 金钱奇遇：获得大量金币\n\n"
            "**触发方式**：\n"
            "- 大部分奇遇需要满足特定条件（地点、时间、天气等）\n"
            "- 有些需要特定物品触发\n"
            "- 有些是完全随机的（看脸）\n\n"
            "**萌新建议**：\n"
            "1. 安装「剑三奇遇助手」插件\n"
            "2. 关注世界频道的奇遇通报\n"
            "3. 加入帮会，老玩家会带你触发\n"
            "4. 平时多在地图上跑跑，有些奇遇是路上碰到的"
        ),
    },
    {
        "question": "插件怎么安装？推荐哪些插件？",
        "answer": (
            "剑三支持插件系统，可以大幅提升游戏体验。\n\n"
            "**安装方法**：\n"
            "1. 下载「海鳗插件管理器」（最主流的插件平台）\n"
            "2. 安装后运行，会自动检测剑三安装路径\n"
            "3. 在管理器里搜索插件，一键安装\n\n"
            "**萌新必装插件**：\n"
            "- **茗伊插件集**：最全面的综合插件，包含团队框架、目标监控等\n"
            "- **DBM（副本助手）**：副本 BOSS 技能提醒，新手救命神器\n"
            "- **大脚插件**：综合型插件，功能丰富\n"
            "- **奇遇助手**：追踪奇遇触发条件\n"
            "- **DPS统计**：查看自己的输出数据\n\n"
            "**注意**：只从官方渠道下载插件，不要用来路不明的插件，防止盗号。"
        ),
    },
    {
        "question": "怎么提升装备等级？",
        "answer": (
            "剑三的装备提升主要有以下途径：\n\n"
            "**PVE 装备**：\n"
            "1. **副本掉落**：难度越高的副本，装备等级越高\n"
            "   - 普通本 → 英雄本 → 25人本\n"
            "2. **秘境币兑换**：刷副本攒币，在NPC处兑换\n"
            "3. **精炼**：用精炼石提升已有装备的品质\n"
            "4. **附魔**：给装备附加额外属性\n\n"
            "**萌新升装路线**：\n"
            "1. 满级后先做主线送的装备\n"
            "2. 打普通五人本积累基础装备\n"
            "3. 装等够了去打英雄五人本\n"
            "4. 逐步挑战 10 人 / 25 人副本\n\n"
            "**小贴士**：\n"
            "- 先把武器升到最好，武器对输出影响最大\n"
            "- 找帮会大佬带你刷本效率最高\n"
            "- 关注每周重置时间，及时刷CD"
        ),
    },
]


# ============================================================
# 数据清洗
# ============================================================

def clean_text(text: str) -> str:
    """清洗文本"""
    import re
    # 去除多余空白
    text = re.sub(r"\s+", " ", text).strip()
    # 去除 HTML 残留
    text = re.sub(r"<[^>]+>", "", text)
    # 去除特殊字符
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def dedup_by_content(items: list[dict], key: str = "content") -> list[dict]:
    """基于内容哈希去重"""
    seen = set()
    result = []
    for item in items:
        text = item.get(key, "")
        h = hashlib.md5(text.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(item)
    return result


# ============================================================
# AI 生成 Q&A 对
# ============================================================

def generate_qa_from_wiki(articles: list[dict], api_key: str | None = None) -> list[dict]:
    """
    基于 wiki 文章用 AI 生成 Q&A 训练对

    如果没有 API key，返回基于规则的简单 Q&A
    """
    qa_pairs = []

    if api_key:
        # 使用 Claude API 生成高质量 Q&A
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            for article in articles:
                prompt = (
                    f"基于以下剑网三游戏资料，生成 5 个萌新玩家可能会问的问题和详细回答。\n"
                    f"要求：\n"
                    f"1. 问题要自然口语化\n"
                    f"2. 回答要准确、友好、有具体建议\n"
                    f"3. 回答长度适中（100-300字）\n"
                    f"4. 输出 JSON 数组格式：[{{\"question\": \"...\", \"answer\": \"...\"}}]\n\n"
                    f"资料标题：{article['title']}\n"
                    f"资料内容：{article['content'][:2000]}"
                )

                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text

                # 解析 JSON
                try:
                    # 尝试从回复中提取 JSON
                    import re
                    json_match = re.search(r"\[.*\]", text, re.DOTALL)
                    if json_match:
                        pairs = json.loads(json_match.group())
                        qa_pairs.extend(pairs)
                        print(f"  ✓ 从「{article['title'][:30]}」生成 {len(pairs)} 个 Q&A")
                except json.JSONDecodeError:
                    print(f"  ✗ JSON 解析失败: {article['title'][:30]}")

        except ImportError:
            print("未安装 anthropic 库，使用规则生成")
            api_key = None

    if not api_key:
        # 基于规则的简单生成
        for article in articles:
            qa_pairs.append({
                "question": f"介绍一下{article['title']}",
                "answer": article["content"][:500],
            })

    return qa_pairs


def generate_qa_from_community(posts: list[dict]) -> list[dict]:
    """将社区问答帖转换为 Q&A 训练对"""
    qa_pairs = []
    for post in posts:
        question = clean_text(post.get("question", ""))
        replies = post.get("replies", [])

        if not question or not replies:
            continue

        # 取最长的回复作为答案（通常质量较高）
        best_reply = max(replies, key=len)
        answer = clean_text(best_reply)

        if len(question) > 10 and len(answer) > 30:
            qa_pairs.append({
                "question": question[:500],
                "answer": answer[:1000],
            })

    return qa_pairs


# ============================================================
# 输出格式化
# ============================================================

def format_as_qwen_chat(qa_pairs: list[dict]) -> list[dict]:
    """将 Q&A 对转换为 Qwen Chat 格式"""
    formatted = []
    for pair in qa_pairs:
        formatted.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": pair["question"]},
                {"role": "assistant", "content": pair["answer"]},
            ]
        })
    return formatted


def split_train_val(data: list[dict], val_ratio: float = 0.1) -> tuple[list, list]:
    """划分训练集和验证集"""
    import random
    random.shuffle(data)
    val_size = max(1, int(len(data) * val_ratio))
    return data[val_size:], data[:val_size]


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="构建剑网三 Q&A 训练数据")
    parser.add_argument("--wiki", type=str, default="data/raw/wiki_data.json")
    parser.add_argument("--community", type=str, default="data/raw/community_data.json")
    parser.add_argument("--output", type=str, default="data/train.jsonl")
    parser.add_argument("--api-key", type=str, default=None, help="Claude API Key (可选)")
    parser.add_argument("--seed-only", action="store_true", help="仅使用种子数据")
    args = parser.parse_args()

    all_qa_pairs = []

    # 1. 添加种子数据（始终包含）
    all_qa_pairs.extend(SEED_QA_PAIRS)
    print(f"种子数据: {len(SEED_QA_PAIRS)} 条")

    if not args.seed_only:
        # 2. 从 wiki 数据生成 Q&A
        wiki_path = Path(args.wiki)
        if wiki_path.exists():
            with open(wiki_path, encoding="utf-8") as f:
                wiki_data = json.load(f)
            wiki_data = dedup_by_content(wiki_data)
            wiki_qa = generate_qa_from_wiki(wiki_data, args.api_key)
            all_qa_pairs.extend(wiki_qa)
            print(f"Wiki Q&A: {len(wiki_qa)} 条")
        else:
            print(f"Wiki 数据文件不存在: {wiki_path}，跳过")

        # 3. 从社区数据生成 Q&A
        community_path = Path(args.community)
        if community_path.exists():
            with open(community_path, encoding="utf-8") as f:
                community_data = json.load(f)
            community_qa = generate_qa_from_community(community_data)
            all_qa_pairs.extend(community_qa)
            print(f"社区 Q&A: {len(community_qa)} 条")
        else:
            print(f"社区数据文件不存在: {community_path}，跳过")

    # 4. 去重
    seen = set()
    unique_pairs = []
    for pair in all_qa_pairs:
        key = pair["question"][:50]
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)
    print(f"去重后: {len(unique_pairs)} 条")

    # 5. 格式化
    formatted = format_as_qwen_chat(unique_pairs)

    # 6. 划分训练/验证集
    train_data, val_data = split_train_val(formatted)

    # 7. 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    val_path = output_path.with_name("val.jsonl")
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n训练集: {len(train_data)} 条 → {output_path}")
    print(f"验证集: {len(val_data)} 条 → {val_path}")


if __name__ == "__main__":
    main()
