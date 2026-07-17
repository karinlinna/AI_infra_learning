"""
eval/eval_harness.py  ——  工具调用能力评测 harness
========================================================================
【JD-4 训-评-迭代闭环】+【JD-5 评测基准】+【JD-3 反幻觉度量】

一把「尺子」，对任意模型输出做规则化、可复现的打分。核心指标：

  1. tool_selection_acc  工具选择准确率
       - 该调工具的样本(gold≠null)：是否选对了工具名
       - 不该调工具的样本(gold=null)：是否正确地「克制未调用」(abstain)
  2. arg_accuracy        参数准确率（仅在工具选对的前提下，参数是否完全正确）
  3. hallucination_rate  幻觉调用率（gold=null 却硬调了工具的比例）—— 越低越好
  4. task_success        端到端成功率（选对工具 且 参数对；或该弃调时正确弃调）

并按 category / difficulty 分桶，暴露模型在「多工具干扰」「参数缺失」「多轮」上的短板。

三种预测来源（解耦「推理」与「打分」，工程上更灵活）：
  --mock [--skill 0.7]   规则模拟模型，本地秒级跑通闭环、演示报表（无需 GPU）
  --pred predictions.jsonl   读已有模型输出打分（每行 {"id":..., "output":"...模型原文..."}）
  --model outputs/sft        直接加载 HF 模型现场推理并打分（需 transformers，CPU 慢）

用法：
  python eval/eval_harness.py --mock                 # base 水平
  python eval/eval_harness.py --mock --skill 0.9     # 模拟 DPO 后
  python eval/eval_harness.py --pred outputs/sft_preds.jsonl
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


# ======================================================================
# 1. 解析：从模型原文里抽出 tool_call（做到容错，真实模型输出常有噪声）
# ======================================================================
def parse_tool_call(text):
    """返回 {"name":..., "arguments":{...}} 或 None（表示没调工具）。解析失败视为没调。"""
    if not text:
        return None
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    obj.setdefault("arguments", {})
    if not isinstance(obj["arguments"], dict):
        return None
    return obj


# ======================================================================
# 2. 打分：单条样本 -> 各项判定
# ======================================================================
def _norm(v):
    """参数值归一化比较：数字 100 与 "100" 视为相等，字符串去空格。"""
    if isinstance(v, (int, float)):
        return str(v)
    return str(v).strip()


def score_one(sample, pred):
    """
    返回 dict:
      selected_ok  工具选择是否正确（含 abstain 判定）
      args_ok      参数是否完全正确（仅工具选对时有意义，否则 None）
      halluc       是否发生幻觉调用（gold=null 却调了工具）
      success      端到端成功
    """
    gold = sample["gold"]
    called = pred is not None
    base = {"gold_null": gold is None}

    # --- gold 为 null：不该调工具 ---
    if gold is None:
        halluc = called
        selected_ok = not called          # 正确弃调
        return {**base, "selected_ok": selected_ok, "args_ok": None,
                "halluc": halluc, "success": selected_ok}

    # --- gold 非 null：应调用指定工具 ---
    if not called:
        # 该调却没调（漏调）
        return {**base, "selected_ok": False, "args_ok": None, "halluc": False, "success": False}

    selected_ok = pred["name"] == gold["name"]
    if not selected_ok:
        return {**base, "selected_ok": False, "args_ok": None, "halluc": False, "success": False}

    # 工具选对 → 比参数（必填必须对；这里要求所有 gold 参数键值完全匹配）
    gargs, pargs = gold["arguments"], pred.get("arguments", {})
    args_ok = all(k in pargs and _norm(pargs[k]) == _norm(v) for k, v in gargs.items()) \
        and len(pargs) == len(gargs)
    return {**base, "selected_ok": True, "args_ok": args_ok, "halluc": False, "success": args_ok}


# ======================================================================
# 3. 汇总：按整体 / category / difficulty 聚合
# ======================================================================
def aggregate(samples, preds):
    buckets = defaultdict(lambda: defaultdict(list))  # dim -> key -> list[score]

    def add(dim, key, s):
        buckets[dim][key].append(s)

    for sample in samples:
        pred = preds.get(sample["id"])
        s = score_one(sample, pred)
        add("overall", "ALL", s)
        add("category", sample["category"], s)
        add("difficulty", sample["difficulty"], s)

    def summarize(rows):
        n = len(rows)
        sel = sum(r["selected_ok"] for r in rows) / n
        succ = sum(r["success"] for r in rows) / n
        arg_rows = [r for r in rows if r["args_ok"] is not None]
        argacc = (sum(r["args_ok"] for r in arg_rows) / len(arg_rows)) if arg_rows else float("nan")
        # 幻觉率：分母 = 该弃调的样本(gold_null)，分子 = 其中硬调了工具的
        abstain_rows = [r for r in rows if r["gold_null"]]
        halluc_rate = (sum(r["halluc"] for r in abstain_rows) / len(abstain_rows)) \
            if abstain_rows else float("nan")
        return {"n": n, "selection": sel, "arg_acc": argacc,
                "halluc_rate": halluc_rate, "success": succ}

    out = {}
    for dim, keys in buckets.items():
        out[dim] = {k: summarize(v) for k, v in keys.items()}
    return out


def print_report(report, title="EVAL REPORT"):
    def pct(x):
        return "  n/a " if x != x else f"{x*100:5.1f}%"  # x!=x -> NaN

    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)
    o = report["overall"]["ALL"]
    print(f"  样本数            {o['n']}")
    print(f"  工具选择准确率     {pct(o['selection'])}")
    print(f"  参数准确率         {pct(o['arg_acc'])}   (仅工具选对的样本)")
    print(f"  幻觉调用率         {pct(o['halluc_rate'])}   (越低越好)")
    print(f"  端到端成功率       {pct(o['success'])}   ★ 主指标")

    for dim, label in [("category", "按类别"), ("difficulty", "按难度")]:
        print(f"\n  —— {label} 分桶（端到端成功率 / 工具选择）——")
        order = (["single_tool", "multi_turn", "missing_param", "no_tool"]
                 if dim == "category" else ["easy", "medium", "hard"])
        for k in order:
            if k in report[dim]:
                r = report[dim][k]
                print(f"    {k:<14} success={pct(r['success'])}  selection={pct(r['selection'])}  (n={r['n']})")
    print("=" * 62)
    return o


# ======================================================================
# 4. 预测来源
# ======================================================================
def load_benchmark(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def preds_from_file(path):
    """每行 {"id":..., "output":"模型原文"}。"""
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            preds[row["id"]] = parse_tool_call(row["output"])
    return preds


def preds_from_mock(samples, skill, seed=0):
    """
    规则模拟「不同水平的模型」，用于本地跑通闭环 & 演示 base→SFT→DPO 收益曲线。
    skill ∈ [0,1]：越高越像训练充分的模型。真实失败模式建模：
      - 选错工具（有干扰工具时更易错）
      - 参数漏填/填错
      - 该弃调时幻觉调用（低 skill 尤其明显）
    """
    rng = random.Random(seed)
    preds = {}
    for s in samples:
        gold = s["gold"]
        # 弃调场景：低 skill 更容易幻觉硬调工具
        if gold is None:
            if rng.random() < (1 - skill) * 0.8:      # 幻觉概率随 skill 上升而下降
                tool = rng.choice(s["tools"])
                fake_args = {p: "?" for p, m in tool["parameters"].items() if m["required"]}
                preds[s["id"]] = {"name": tool["name"], "arguments": fake_args}
            else:
                preds[s["id"]] = None
            continue
        # 应调场景：以 skill 概率选对工具
        if rng.random() < skill:
            name = gold["name"]
        else:
            others = [t["name"] for t in s["tools"] if t["name"] != gold["name"]]
            name = rng.choice(others) if others else gold["name"]
        if name != gold["name"]:
            preds[s["id"]] = {"name": name, "arguments": {}}
            continue
        # 工具选对 → 参数以 skill^0.5 概率全对（参数比选工具更难）
        if rng.random() < skill ** 0.5:
            preds[s["id"]] = {"name": name, "arguments": dict(gold["arguments"])}
        else:
            bad = dict(gold["arguments"])
            if bad:  # 破坏一个参数
                k = rng.choice(list(bad.keys()))
                bad[k] = "WRONG"
            preds[s["id"]] = {"name": name, "arguments": bad}
    return preds


def preds_from_model(samples, model_path, max_new_tokens=128):
    """加载 HF 模型现场推理（可选，需 transformers；CPU 上慢）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.eval()
    preds = {}
    for s in samples:
        sys = _system_prompt(s["tools"])
        msgs = [{"role": "system", "content": sys}] + s["messages"]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        gen = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        preds[s["id"]] = parse_tool_call(gen)
    return preds


def _system_prompt(tools):
    """把工具列表展开进 system prompt（训练/推理需与此保持一致）。"""
    lines = ["你是一个会使用工具的智能助手。可用工具如下（JSON schema）：", ""]
    for t in tools:
        lines.append(json.dumps(t, ensure_ascii=False))
    lines += ["",
              "需要调工具时，输出：<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>",
              "不需要工具时，直接自然语言回答。参数不全时，先追问而不要编造。"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=str(ROOT / "eval" / "benchmark.jsonl"))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mock", action="store_true", help="用规则模拟模型")
    src.add_argument("--pred", help="读预测文件 {id, output}")
    src.add_argument("--model", help="加载 HF 模型现场推理")
    ap.add_argument("--skill", type=float, default=0.4, help="[--mock] 模型水平 0~1")
    ap.add_argument("--tag", default=None, help="报表标题标签")
    ap.add_argument("--dump", default=None, help="把打分明细写到 jsonl")
    args = ap.parse_args()

    samples = load_benchmark(args.benchmark)

    if args.mock:
        preds = preds_from_mock(samples, skill=args.skill)
        tag = args.tag or f"MOCK (skill={args.skill})"
    elif args.pred:
        preds = preds_from_file(args.pred)
        tag = args.tag or f"PRED {Path(args.pred).name}"
    else:
        preds = preds_from_model(samples, args.model)
        tag = args.tag or f"MODEL {args.model}"

    report = aggregate(samples, preds)
    print_report(report, title=tag)

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(
                    {"id": s["id"], "gold": s["gold"],
                     "pred": preds.get(s["id"]), **score_one(s, preds.get(s["id"]))},
                    ensure_ascii=False) + "\n")
        print(f"\n[dump] 打分明细 -> {args.dump}")


if __name__ == "__main__":
    main()
