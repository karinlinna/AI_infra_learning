"""
data/build_sft_data.py  ——  合成「工具调用」SFT 数据 + 评测 benchmark
========================================================================
【JD-2 后训练数据体系】数据采集(合成) / 标注 / 分布与多样性控制

为什么用合成数据：工具调用样本可以由「工具 schema + 参数取值」程序化生成，
天然带 gold 标签（该调哪个工具、参数是什么），无需人工标注即可规模化，
且能精确控制 **数据分布** 与 **难度分层** —— 这正是后训练数据体系的核心诉求。

产物：
  - data/sft_train.jsonl     SFT 训练集（assistant 带正确 tool_call）
  - eval/benchmark.jsonl     held-out 评测集（只含 gold，不含答案，供 eval_harness 打分）

数据 schema（每行一个样本）：
  {
    "id": "...",
    "category": "single_tool | multi_turn | no_tool | missing_param",
    "difficulty": "easy | medium | hard",
    "tools":   [ {工具定义...} ],           # 本轮可用工具（system 里会展开）
    "messages":[ {"role": "user", ...}, ...],
    "gold":    {"name": ..., "arguments": {...}} | null,   # null = 不该调工具
    "assistant": "<think>...</think>\n<tool_call>...</tool_call>"   # SFT 目标（benchmark 不含）
  }

工具调用格式约定（业界常见写法，便于规则解析）：
  <tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>
  不需要调工具时，直接自然语言回答，不出现 <tool_call>。
"""

import argparse
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# --------------------------------------------------------------------------
# 1. 工具库（tool registry）—— 每个工具有 name / 描述 / 参数 schema / 取值池
#    取值池用于程序化采样出多样的参数组合，控制数据分布。
# --------------------------------------------------------------------------
TOOLS = {
    "get_weather": {
        "description": "查询指定城市的天气",
        "parameters": {
            "city": {"type": "string", "required": True},
            "date": {"type": "string", "required": False},
        },
        "samples": {
            "city": ["北京", "上海", "深圳", "杭州", "成都", "东京", "纽约", "伦敦"],
            "date": ["今天", "明天", "周末", "后天"],
        },
    },
    "calculator": {
        "description": "进行数学计算，输入一个表达式",
        "parameters": {"expression": {"type": "string", "required": True}},
        "samples": {"expression": ["23*47", "1024/8", "156+289", "99-37", "(12+8)*5"]},
    },
    "web_search": {
        "description": "在互联网上搜索信息",
        "parameters": {
            "query": {"type": "string", "required": True},
            "top_k": {"type": "integer", "required": False},
        },
        "samples": {
            "query": ["2024 诺贝尔物理学奖", "最新的 GPT 模型", "杭州亚运会金牌榜",
                      "Python 3.12 新特性", "如何做红烧肉"],
            "top_k": [3, 5, 10],
        },
    },
    "send_email": {
        "description": "发送邮件",
        "parameters": {
            "to": {"type": "string", "required": True},
            "subject": {"type": "string", "required": True},
            "body": {"type": "string", "required": False},
        },
        "samples": {
            "to": ["boss@corp.com", "alice@x.com", "team@corp.com"],
            "subject": ["周报", "会议纪要", "请假申请", "项目进展"],
            "body": ["详见附件", "请查收", "下午三点开会"],
        },
    },
    "create_calendar_event": {
        "description": "创建日历日程",
        "parameters": {
            "title": {"type": "string", "required": True},
            "time": {"type": "string", "required": True},
        },
        "samples": {
            "title": ["产品评审", "牙医预约", "团队聚餐", "面试"],
            "time": ["明天上午10点", "周五下午3点", "下周一9点"],
        },
    },
    "currency_convert": {
        "description": "货币汇率换算",
        "parameters": {
            "amount": {"type": "number", "required": True},
            "from_currency": {"type": "string", "required": True},
            "to_currency": {"type": "string", "required": True},
        },
        "samples": {
            "amount": [100, 250, 1000, 5000],
            "from_currency": ["USD", "CNY", "EUR", "JPY"],
            "to_currency": ["CNY", "USD", "EUR", "GBP"],
        },
    },
}

# 用户问法模板：{slots} 会被采样出的参数值填充。每个工具多套模板 → 提升语言多样性。
QUERY_TEMPLATES = {
    "get_weather": [
        "{city}{date}天气怎么样？",
        "帮我查一下{city}的天气",
        "我{date}要去{city}，那边天气如何",
    ],
    "calculator": [
        "帮我算一下 {expression}",
        "{expression} 等于多少？",
        "计算 {expression}",
    ],
    "web_search": [
        "帮我搜一下{query}",
        "查查{query}的相关信息",
        "我想了解{query}",
    ],
    "send_email": [
        "给{to}发一封主题为「{subject}」的邮件",
        "帮我给{to}写个邮件，主题是{subject}",
    ],
    "create_calendar_event": [
        "在{time}帮我建个「{title}」的日程",
        "帮我把{title}加到日历，时间是{time}",
    ],
    "currency_convert": [
        "{amount}{from_currency}等于多少{to_currency}？",
        "帮我把{amount}{from_currency}换算成{to_currency}",
    ],
}

# no_tool 类：闲聊 / 常识问答，模型应直接回答，**不应**调用任何工具（反幻觉）。
NO_TOOL_QUERIES = [
    "你好呀，今天心情怎么样？",
    "给我讲个冷笑话",
    "1+1 为什么等于 2？用一句话解释",
    "你觉得人工智能会有意识吗？",
    "推荐一句励志的话",
    "谢谢你的帮助！",
    "什么是机器学习？简单说说",
    "周末适合做点什么放松？",
]


def _fmt_think(tool_name, args):
    """生成一段简短 CoT，体现「先判断意图→选工具→填参数」的推理链。"""
    return (f"用户的意图需要调用工具 `{tool_name}`。"
            f"根据请求提取参数：{json.dumps(args, ensure_ascii=False)}。")


def _tool_call_str(name, args):
    return f'<tool_call>{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}</tool_call>'


def _sample_args(tool_name, rng, drop_required=False):
    """从取值池采样一组参数。drop_required=True 时故意漏掉一个必填参数（missing_param 类）。"""
    spec = TOOLS[tool_name]
    args, dropped = {}, None
    required = [p for p, m in spec["parameters"].items() if m["required"]]
    if drop_required and required:
        dropped = rng.choice(required)
    for p, meta in spec["parameters"].items():
        if p == dropped:
            continue
        # 必填参数一定给；可选参数 50% 概率给，制造分布多样性
        if meta["required"] or rng.random() < 0.5:
            args[p] = rng.choice(spec["samples"][p])
    return args, dropped


def _tool_defs_for(tool_name, rng, n_distractors=2):
    """构造本轮工具列表：目标工具 + 若干干扰工具（提升难度：模型要选对）。"""
    others = [t for t in TOOLS if t != tool_name]
    picked = [tool_name] + rng.sample(others, min(n_distractors, len(others)))
    rng.shuffle(picked)
    return [{"name": t, "description": TOOLS[t]["description"],
             "parameters": TOOLS[t]["parameters"]} for t in picked]


def _difficulty(n_distractors, n_optional_filled):
    """难度分层：干扰工具越多、可选参数越多 → 越难。用于数据分层与评测分桶。"""
    score = n_distractors + n_optional_filled
    return "easy" if score <= 1 else ("medium" if score <= 3 else "hard")


def build_single_tool(rng, idx):
    tool = rng.choice(list(TOOLS.keys()))
    n_distractors = rng.randint(0, 3)
    args, _ = _sample_args(tool, rng)
    template = rng.choice(QUERY_TEMPLATES[tool])
    # 模板槽位可能多于采样到的参数（可选参数没给时），用取值池兜底填充问法文本
    slot_vals = {**{k: rng.choice(v) for k, v in TOOLS[tool]["samples"].items()}, **args}
    query = template.format(**slot_vals)
    n_optional = len([p for p in args if not TOOLS[tool]["parameters"][p]["required"]])
    return {
        "id": f"single-{idx}",
        "category": "single_tool",
        "difficulty": _difficulty(n_distractors, n_optional),
        "tools": _tool_defs_for(tool, rng, n_distractors),
        "messages": [{"role": "user", "content": query}],
        "gold": {"name": tool, "arguments": args},
        "assistant": _fmt_think(tool, args) + "\n" + _tool_call_str(tool, args),
    }


def build_no_tool(rng, idx):
    query = rng.choice(NO_TOOL_QUERIES)
    tool = rng.choice(list(TOOLS.keys()))
    return {
        "id": f"notool-{idx}",
        "category": "no_tool",
        "difficulty": "medium",  # 抑制幻觉调用有一定难度
        "tools": _tool_defs_for(tool, rng, n_distractors=2),
        "messages": [{"role": "user", "content": query}],
        "gold": None,  # 不该调任何工具
        "assistant": "这个问题我可以直接回答，不需要调用工具。",
    }


def build_missing_param(rng, idx):
    """参数缺失类：用户没给全必填参数，正确行为是**追问澄清**而非瞎编（反幻觉）。"""
    tool = rng.choice(list(TOOLS.keys()))
    args, dropped = _sample_args(tool, rng, drop_required=True)
    if dropped is None:  # 该工具没有可丢的必填参数，退化成 single_tool
        return build_single_tool(rng, idx)
    template = rng.choice(QUERY_TEMPLATES[tool])
    slot_vals = {**{k: rng.choice(v) for k, v in TOOLS[tool]["samples"].items()}, **args}
    slot_vals[dropped] = ""  # 问法里也不出现该参数
    query = template.format(**slot_vals).replace("  ", " ").strip()
    return {
        "id": f"missing-{idx}",
        "category": "missing_param",
        "difficulty": "hard",
        "tools": _tool_defs_for(tool, rng, n_distractors=1),
        "messages": [{"role": "user", "content": query}],
        # gold=null：正确行为是追问，不产出 tool_call；eval 判「是否克制未瞎调」
        "gold": None,
        "assistant": f"为了调用 `{tool}`，我还需要你补充 `{dropped}`，方便告诉我吗？",
    }


def build_multi_turn(rng, idx):
    """多轮：第一轮已调工具并返回结果，考察模型基于 tool 结果的后续动作一致性。"""
    tool = rng.choice(["get_weather", "web_search", "calculator"])
    args, _ = _sample_args(tool, rng)
    template = rng.choice(QUERY_TEMPLATES[tool])
    slot_vals = {**{k: rng.choice(v) for k, v in TOOLS[tool]["samples"].items()}, **args}
    q1 = template.format(**slot_vals)
    tool_result = {"get_weather": "晴，26℃", "web_search": "已找到 3 条相关结果",
                   "calculator": "结果为 1081"}[tool]
    follow_up = "谢谢，那再帮我查下明天的呢？" if tool == "get_weather" else "好的，谢谢！"
    # 第二轮：get_weather 需再次调工具（日期改明天）；其余为收尾不再调工具
    if tool == "get_weather":
        args2 = {**args, "date": "明天"}
        gold = {"name": tool, "arguments": args2}
        assistant = _tool_call_str(tool, args2)
    else:
        gold = None
        assistant = "不客气，还有什么可以帮你的吗？"
    return {
        "id": f"multi-{idx}",
        "category": "multi_turn",
        "difficulty": "hard",
        "tools": _tool_defs_for(tool, rng, n_distractors=1),
        "messages": [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": _tool_call_str(tool, args)},
            {"role": "tool", "content": tool_result},
            {"role": "user", "content": follow_up},
        ],
        "gold": gold,
        "assistant": assistant,
    }


BUILDERS = {
    "single_tool": build_single_tool,
    "no_tool": build_no_tool,
    "missing_param": build_missing_param,
    "multi_turn": build_multi_turn,
}

# 类别分布（可调）：控制数据集的 **分布与多样性**
DEFAULT_MIX = {
    "single_tool": 0.50,
    "no_tool": 0.20,
    "missing_param": 0.15,
    "multi_turn": 0.15,
}


def generate(n, seed, mix=DEFAULT_MIX):
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        cat = rng.choices(list(mix.keys()), weights=list(mix.values()))[0]
        samples.append(BUILDERS[cat](rng, i))
    return samples


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _stats(rows):
    from collections import Counter
    cat = Counter(r["category"] for r in rows)
    dif = Counter(r["difficulty"] for r in rows)
    return dict(cat), dict(dif)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=800, help="训练样本数")
    ap.add_argument("--n_eval", type=int, default=200, help="评测样本数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train = generate(args.n_train, seed=args.seed)
    # 评测集用不同 seed，保证与训练集不重叠（held-out）
    eval_rows = generate(args.n_eval, seed=args.seed + 10000)
    # benchmark 去掉 assistant 答案，只保留 gold 供打分
    bench = [{k: v for k, v in r.items() if k != "assistant"} for r in eval_rows]

    _write_jsonl(ROOT / "data" / "sft_train.jsonl", train)
    _write_jsonl(ROOT / "eval" / "benchmark.jsonl", bench)

    print(f"[OK] 训练集 -> data/sft_train.jsonl        ({len(train)} 条)")
    print(f"[OK] 评测集 -> eval/benchmark.jsonl        ({len(bench)} 条)")
    tc, td = _stats(train)
    ec, ed = _stats(bench)
    print(f"     训练分布  类别={tc}")
    print(f"               难度={td}")
    print(f"     评测分布  类别={ec}")
    print(f"               难度={ed}")
    print("\n样例（训练集第 1 条）：")
    print(json.dumps(train[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
