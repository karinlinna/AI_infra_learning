# 剑网三萌新问答机器人

基于 **Qwen2.5-14B-Instruct** 微调的剑网三游戏问答助手，支持联网搜索，可部署为 QQ 机器人。

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
                                   │ Qwen2.5-14B │     │ DuckDuckGo    │
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
│   ├── sft_lora.py                # LoRA 微调 Qwen2.5-14B
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
| 基座模型 | Qwen2.5-14B-Instruct | 中文能力强，14B 参数量兼顾质量与效率 |
| 微调方式 | LoRA (rank=32) + QLoRA 4bit | 24GB 显存可跑，参数高效 |
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
- **QLoRA 微调**（`--use-4bit`，推荐）：NVIDIA GPU，24GB 显存（RTX 3090/4090）
- **LoRA 微调**：40GB+ 显存（A100 40GB/80GB）
- **推理**：~30GB 显存（bf16 加载 14B 模型），或 ~10GB（int4 量化）

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
export ANTHROPIC_BASE_URL="https://api.duckcoding.ai" 
python data/build_dataset.py --wiki data/raw/wiki_data.json --community data/raw/community_data.json --api-key sk- --output data/train.jsonl 

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

> 训练产物默认保存到 AutoDL 服务器的 `/root/data/` 目录下，便于持久化存储。

```bash
cd train

# QLoRA 4bit（推荐，24GB 显存即可）
python sft_lora.py --model Qwen/Qwen2.5-14B-Instruct --train-data ../data/train.jsonl --use-4bit --output-dir /root/autodl-tmp/jx3_lora

# 标准 LoRA（需要 40GB+ 显存，如 A100）
python sft_lora.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --train-data ../data/train.jsonl \
    --output-dir /root/autodl-tmp/jx3_lora

# 显存不够？用更小的模型
python sft_lora.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train-data ../data/train.jsonl \
    --output-dir /root/autodl-tmp/jx3_lora
```

训练参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 3 | 训练轮数 |
| `--batch-size` | 2 | 每卡批次大小 |
| `--grad-accum` | 8 | 梯度累积步数（等效 batch=16） |
| `--lr` | 1e-4 | 学习率 |
| `--max-seq-len` | 2048 | 最大序列长度 |
| `--lora-rank` | 32 | LoRA 秩 |
| `--lora-alpha` | 64 | LoRA 缩放系数 |
| `--use-4bit` | false | 启用 QLoRA 4bit 量化 |

### Step 4: 合并权重（可选）

```bash
# 将 LoRA 合并到基座，生成独立模型（部署更方便），数据放在数据盘                                  
python merge_lora.py --base-model Qwen/Qwen2.5-14B-Instruct --lora-path /root/autodl-tmp/jx3_lora --output-dir /root/autodl-tmp/jx3_merged      

查看文件占用：du -sh /root/* /root/.cache /root/.local 2>/dev/null | sort -rh | head -20
17G     /root/autodl-tmp
9.4G    /root/github
7.8G    /root/miniconda3
56K     /root/.local
0       /root/tf-logs
0       /root/data
0       /root/autodl-pub
```

也可以跳过合并，推理时直接加载 base + LoRA adapter。

### Step 5: 启动推理服务

```bash
cd inference

# 使用合并后的模型
python server.py --model /root/autodl-tmp/jx3_merged  

# 或使用 base + LoRA adapter
python server.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --lora /root/data/jx3_lora
```

测试 API：

```bash
# 普通问答
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question": "纯阳适合新手吗？"}'

# 涉及最新内容会自动联网搜索
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question": "最新出的门派是什么？有什么技能？"}'

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
| 训练 | Qwen2.5-14B | QLoRA 4bit | ~24GB |
| 训练 | Qwen2.5-14B | LoRA bf16 | ~40GB |
| 训练 | Qwen2.5-7B | LoRA bf16 | ~20GB |
| 推理 | Qwen2.5-14B | bf16 | ~30GB |
| 推理 | Qwen2.5-14B | int4 量化 | ~10GB |
| 推理 | Qwen2.5-7B | bf16 | ~16GB |

## 后续优化方向

- [ ] 增加多轮对话支持（维护对话历史）
- [ ] 接入 RAG 知识库（向量检索剑三 Wiki）
- [ ] DPO/RLHF 对齐（收集用户 👍👎 反馈训练）
- [ ] 模型量化部署（GPTQ/AWQ int4，降低推理显存）
- [ ] 支持图片问答（识别游戏截图回答问题）
- [ ] 定期自动更新数据（版本更新后自动爬取+微调）
