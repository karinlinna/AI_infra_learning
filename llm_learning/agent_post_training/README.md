# Agent 后训练流水线 (Agent Post-Training Pipeline)

一个**可运行、可复现**的 Agent 后训练 mini-infra：把一个基座小模型，通过
`数据构造 → SFT → 偏好数据 → DPO → 评测` 的完整闭环，
系统性提升它的 **工具调用 (tool calling)** 能力，并用同一套 eval harness 量化每一步的收益。

> 载体选「工具调用」的原因：贴近 Agent 场景、数据可自动合成、评测客观可自动打分
> （工具名 / 参数 / 多轮一致性都能规则化判分），因此适合作为完整后训练链路的最小验证载体。

---

## 主线故事

```
                          ┌──────────────────────────────────────────────┐
                          │              eval/  评测 harness               │
                          │   工具名准确率 / 参数准确率 / 多轮一致性        │
                          └───────────────▲──────────────────────────────┘
                                          │ 每一步都用同一把尺子量化收益
   ┌──────────┐   SFT    ┌──────────┐  采样+打分  ┌──────────┐   DPO   ┌──────────┐
   │  基座模型 │ ───────▶ │ SFT 模型  │ ─────────▶ │ 偏好数据  │ ──────▶ │ DPO 模型 │
   │  Base    │          │ (会用工具) │            │chosen/rej │         │ (用得更对)│
   └──────────┘          └──────────┘            └──────────┘         └──────────┘
        │  40%                 │  ~70%                                      │  80%+
        └──────────────────────┴──────────── 工具调用准确率 ───────────────┘
```

面试一句话：**「我从基座模型出发，用合成数据做 SFT 让它学会调用工具，再用自采样 +
规则/RM 打分构造偏好对做 DPO 纠正错误调用，全程用自建 benchmark 量化每一步收益，
最终工具调用准确率从 ~40% 提升到 80%+。训练侧支持 LoRA + FSDP 多卡。」**

---

## 目录结构

```
agent_post_training/
├── README.md                     # 本文件：主线故事 + 架构 + JD 对照
├── requirements.txt              # 依赖分两档：base(可本地跑) / train(需GPU)
│
├── configs/                      # 配置化，训练参数与代码解耦
│   ├── sft_lora.yaml
│   ├── dpo.yaml
│   └── fsdp.yaml                 # accelerate FSDP 多卡配置
│
├── data/                         # 【JD-2 数据体系】
│   ├── build_sft_data.py         # 合成工具调用 SFT 数据 (tool_call 格式 + CoT)
│   ├── build_pref_data.py        # SFT 模型采样 → 打分 → 造 chosen/rejected
│   └── quality_filter.py         # 去重 / 格式校验 / 难度分层
│
├── train/                        # 【JD-1 核心算法】+【JD-4 分布式】
│   ├── train_sft.py              # LoRA + accelerate/FSDP
│   ├── train_rm.py               # 奖励模型 (Bradley-Terry pairwise)
│   └── train_dpo.py              # DPO (主推)，注释对照 PPO/GRPO
│
├── eval/                         # 【JD-4 评测闭环】+【JD-5 评测基准】
│   ├── eval_harness.py           # 工具调用准确率 / 参数 / 多轮一致性
│   ├── benchmark.jsonl           # 自建评测集（由 build_sft_data.py 生成）
│   └── report.py                 # base vs SFT vs DPO 对比报表
│
├── pipeline.py                   # 一键跑通闭环 + 收益曲线
└── requirements.txt
```

---

## 与岗位 JD 的对照

| JD 能力点 | 本项目模块 | 说明 |
|---|---|---|
| 1. SFT / RM / RLHF 算法 | `train/train_sft.py` `train_rm.py` `train_dpo.py` | DPO 为主线；DPO loss 手写版与 trl 版对照，体现懂原理 |
| 2. 后训练数据体系 | `data/build_sft_data.py` `build_pref_data.py` `quality_filter.py` | 采集(合成)→清洗→分层→偏好对构造 |
| 3. Agent 能力优化 | `eval/` 难度分层 + 反幻觉判分 | 多轮一致性、参数幻觉检测（部分 demo，部分写思路） |
| 4. 工程化 / 分布式 | `configs/fsdp.yaml` + accelerate | LoRA + FSDP 多卡；`pipeline.py` 打通训-评-迭代闭环 |
| 5. 技术沉淀 / 评测基准 | `eval/benchmark.jsonl` + 本 README | 可复用的评测基准与技术文档 |

> 姊妹项目 [`../mini_infra`](../mini_infra) 覆盖训练/推理**系统 infra**（算力/并行/通信/存储/容错）。
> 两者组合 = 「后训练算法链路」+「训练系统底座」的完整视图。

---

## 快速开始

### 本地（Mac / CPU，验证数据与评测闭环）

```bash
pip install -r requirements.txt          # 只装 base 档依赖即可

# 1. 合成 SFT 训练集 + 评测 benchmark
python data/build_sft_data.py

# 2. 用「规则模拟模型」跑通评测闭环（无需真实推理，秒级）
python eval/eval_harness.py --mock

# 输出：工具名准确率 / 参数准确率 / 多轮一致性 的量化报表
```

### GPU 机器（A100 多卡，真训练）

```bash
pip install -r requirements.txt trl peft accelerate datasets   # train 档

# 单机多卡 FSDP SFT
accelerate launch --config_file configs/fsdp.yaml train/train_sft.py --config configs/sft_lora.yaml

# 造偏好数据 → DPO
python data/build_pref_data.py --model outputs/sft
accelerate launch --config_file configs/fsdp.yaml train/train_dpo.py --config configs/dpo.yaml

# 全链路对比评测
python eval/report.py
```

---

## 技术选型

| 组件 | 选型 | 理由 |
|---|---|---|
| 基座 | `Qwen2.5-1.5B-Instruct` | 工具调用基础好、1.5B 多卡训练快、社区支持全 |
| SFT | `peft` LoRA | 显存友好、可快速迭代；代码保留全参微调开关 |
| 偏好优化 | `trl` DPO | 比 PPO 稳定、无需 online rollout，最适合 demo；注释对照 PPO/GRPO |
| 分布式 | `accelerate` + FSDP | 工业界主流，配置化多卡 |
| 评测 | 自建 harness | 规则化打分，客观可复现 |

## 设计原则

- **闭环优先**：任何时刻都能一键跑通 `数据 → 训练 → 评测`，产出收益曲线
- **本地可验证**：数据构造与评测不依赖 GPU，Mac 上秒级跑通
- **配置与代码解耦**：训练超参走 yaml，便于扫参与复现
- **懂原理不只调库**：关键算法（DPO loss、RM loss）保留手写对照版
