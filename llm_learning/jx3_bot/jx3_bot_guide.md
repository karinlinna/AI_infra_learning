# 剑网三萌新问答机器人

基于 **Qwen2.5-7B-Instruct** 微调的剑网三游戏问答助手，支持联网搜索，可部署为 QQ 机器人。

## 架构概览

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│   QQ 群聊     │────▶│  NoneBot2 机器人  │────▶│  FastAPI 推理 │
│  @剑三小助手   │◀────│  (bot/qq_bot.py)  │◀────│  (server.py)  │
└──────────────┘     └──────────────────┘     └──────┬───────┘
                                                     │
                                          ┌──────────┼──────────┐
                                          ▼                      ▼
                                   ┌─────────────┐     ┌───────────────┐
                                   │ Qwen2.5-7B  │     │ DuckDuckGo    │
                                   │ + LoRA 微调  │     │ 联网搜索       │
                                   └─────────────┘     └───────────────┘
```

## 项目结构

```
jx3_bot/
├── data/                          # 数据采集与处理
│   ├── crawl_wiki.py              # 爬取剑网三百科/攻略站
│   ├── crawl_community.py         # 爬取 NGA/百度贴吧社区数据
│   ├── build_dataset.py           # 数据清洗 + AI 生成 Q&A → JSONL
│   └── raw/                       # 原始爬取数据存放
├── train/                         # 模型训练
│   ├── sft_lora.py                # LoRA 微调 Qwen2.5-7B
│   └── merge_lora.py              # 合并 LoRA 权重到基座模型
├── inference/                     # 推理服务
│   ├── server.py                  # FastAPI REST API（支持流式输出）
│   └── web_search.py              # DuckDuckGo 联网搜索模块
├── bot/                           # QQ 机器人
│   ├── qq_bot.py                  # NoneBot2 + OneBot V11 机器人
│   └── config.py                  # 机器人配置（触发词、权限等）
└── requirements.txt               # Python 依赖
```

## 技术选型

| 组件 | 选择 | 说明 |
|------|------|------|
| 基座模型 | Qwen2.5-7B-Instruct | 中文能力最强的开源模型 |
| 微调方式 | LoRA (rank=64) | 24GB 显存可跑，参数高效 |
| 训练框架 | transformers + peft + trl | HuggingFace 生态，成熟稳定 |
| 推理服务 | FastAPI + uvicorn | 轻量高性能，支持流式 SSE |
| 联网搜索 | DuckDuckGo Search | 免费，无需 API Key |
| QQ 机器人 | NoneBot2 + go-cqhttp | 最成熟的 QQ 机器人方案 |
| 训练精度 | bf16 | 显存减半，训练稳定 |

## 快速开始

### 环境准备

```bash
cd jx3_bot
pip install -r requirements.txt
```

硬件要求：
- **LoRA 微调**：NVIDIA GPU，24GB 显存（RTX 3090/4090）
- **QLoRA 微调**（`--use-4bit`）：12GB 显存即可
- **推理**：~16GB 显存（bf16 加载 7B 模型）

### Step 1: 数据采集

```bash
# 爬取百科/攻略站数据
python data/crawl_wiki.py --output data/raw/wiki_data.json

# 爬取 NGA/贴吧社区数据
python data/crawl_community.py --output data/raw/community_data.json
```

### Step 2: 构建训练集

```bash
# 方式 A：仅用内置种子数据（快速验证，无需爬虫）
python data/build_dataset.py --seed-only --output data/train.jsonl

# 方式 B：种子数据 + 爬取数据 + AI 扩充（推荐，需要 Claude API Key）
python data/build_dataset.py \
    --wiki data/raw/wiki_data.json \
    --community data/raw/community_data.json \
    --api-key sk-xxx \
    --output data/train.jsonl
```

训练数据格式（Qwen Chat JSONL）：

```json
{
  "messages": [
    {"role": "system", "content": "你是剑三小助手..."},
    {"role": "user", "content": "纯阳适合新手玩吗？"},
    {"role": "assistant", "content": "纯阳非常适合新手！..."}
  ]
}
```

### Step 3: LoRA 微调

```bash
cd train

# 标准 LoRA（需要 24GB 显存）
python sft_lora.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train-data ../data/train.jsonl \
    --output-dir ./output/jx3_lora

# 显存不够？使用 QLoRA 4bit 量化（12GB 即可）
python sft_lora.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train-data ../data/train.jsonl \
    --use-4bit \
    --output-dir ./output/jx3_lora

# 或者用更小的模型
python sft_lora.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --train-data ../data/train.jsonl \
    --output-dir ./output/jx3_lora
```

训练参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 3 | 训练轮数 |
| `--batch-size` | 2 | 每卡批次大小 |
| `--grad-accum` | 8 | 梯度累积步数（等效 batch=16） |
| `--lr` | 2e-4 | 学习率 |
| `--max-seq-len` | 1024 | 最大序列长度 |
| `--lora-rank` | 64 | LoRA 秩 |
| `--lora-alpha` | 128 | LoRA 缩放系数 |
| `--use-4bit` | false | 启用 QLoRA 4bit 量化 |

### Step 4: 合并权重（可选）

```bash
# 将 LoRA 合并到基座，生成独立模型（部署更方便）
python merge_lora.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --lora-path ./output/jx3_lora \
    --output-dir ./output/jx3_merged
```

也可以跳过合并，推理时直接加载 base + LoRA adapter。

### Step 5: 启动推理服务

```bash
cd inference

# 使用合并后的模型
python server.py --model ../train/output/jx3_merged

# 或使用 base + LoRA adapter
python server.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --lora ../train/output/jx3_lora
```

测试 API：

```bash
# 普通问答
curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"question": "纯阳适合新手吗？"}'

# 涉及最新内容会自动联网搜索
curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"question": "最新版本更新了什么？"}'

# 健康检查
curl http://localhost:8000/health
```

### Step 6: 启动 QQ 机器人

**前置：安装 go-cqhttp**

1. 下载 [go-cqhttp](https://docs.go-cqhttp.org/)
2. 配置 QQ 账号和 WebSocket 连接
3. 启动 go-cqhttp

**启动机器人**

```bash
cd bot
# 按需修改 config.py 中的配置
python qq_bot.py
```

群聊使用方式：

```
@剑三小助手 纯阳适合新手吗？
/jx3 怎么赚钱？
/jx3 最新活动是什么？    ← 会自动联网搜索
```

## 联网搜索机制

当用户问题包含以下关键词时，自动触发联网搜索：

> 最新、更新、新版本、新活动、维护、公告、赛季、什么时候、现在、目前...

流程：

```
用户提问
  │
  ├── 包含时效性关键词？
  │     ├── 是 → DuckDuckGo 搜索 → 提取摘要 → 拼入 prompt → 模型生成
  │     └── 否 → 直接由模型回答
  │
  └── 返回回答（标注是否使用了搜索）
```

## 训练思路（借鉴 Claude）

本项目借鉴了 Claude 的训练方法论：

1. **高质量 SFT 数据**：手工编写种子数据确保质量基线，再用 AI 基于真实资料批量扩充
2. **人设一致性**：通过 system prompt 定义统一的回答风格（友好、耐心、口语化）
3. **准确性优先**：训练数据基于游戏实际机制，不编造信息
4. **迭代优化**：收集用户反馈 → 补充数据 → 重新微调

## 显存估算

| 场景 | 模型 | 方式 | 显存需求 |
|------|------|------|---------|
| 训练 | Qwen2.5-7B | LoRA bf16 | ~20GB |
| 训练 | Qwen2.5-7B | QLoRA 4bit | ~12GB |
| 训练 | Qwen2.5-3B | LoRA bf16 | ~10GB |
| 推理 | Qwen2.5-7B | bf16 | ~16GB |
| 推理 | Qwen2.5-7B | int4 量化 | ~6GB |

## 后续优化方向

- [ ] 增加多轮对话支持（维护对话历史）
- [ ] 接入 RAG 知识库（向量检索剑三 Wiki）
- [ ] DPO/RLHF 对齐（收集用户 👍👎 反馈训练）
- [ ] 模型量化部署（GPTQ/AWQ int4，降低推理显存）
- [ ] 支持图片问答（识别游戏截图回答问题）
- [ ] 定期自动更新数据（版本更新后自动爬取+微调）
