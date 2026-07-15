# 第 3 节课 · 学生课件：落地一个可上线的 Agent/RAG 应用项目

> **这节课补你最大的短板：后端部署。**
> **配套代码**：
> - `student/litianwang/tesla.py` —— 你 Tesla RAG 项目的可运行骨架（`python tesla.py` 直接跑）
> - `llm_learning/jx3_bot/inference/server.py` —— 真实的 FastAPI 推理服务（含流式输出）
> - `memory_optimization/` —— 真实的后端内存调优案例（补后端 sense）

---

## 0. demo 和"可上线"差在哪

```
demo（跑在 notebook）          可上线（别人能用）
  一个 .py 文件            →  REST/流式 API 服务
  手动跑                  →  Docker 一键部署
  出错就崩                →  健康检查 + 重试 + 日志
  单次调用                →  能扛并发、显存可控
  只有代码                →  README + 架构图 + 部署文档
```

> 面试官问"你这个项目能上线吗？"——差的不是模型，是**服务化、部署、可观测、能扛并发**这些"不出错时看不见"的东西。今天全补上。

---

## 1. RAG 长板深挖：把 tesla.py 讲透

先跑：`cd student/litianwang && python tesla.py`，看三个问题的召回命中和精排。

### 1.1 三种切分为什么一起用

```python
def build_chunks(pages):
    for pg in pages:
        parents[pid] = {...整页文本...}                              # 父块：喂给 LLM 的完整上下文
        sents = semantic_split(pg["text"])                          # 语义切分：按句号断
        for ck in sliding_window_chunks(sents, size=2, stride=1):   # 滑窗：块间重叠
            children.append({"chunk_id": ..., "parent_id": pid, ...})  # 子块：细粒度检索
```
- **子块检索、父块喂 LLM**：子块粒度细检索准，命中后按 `parent_id` 回溯父块拿完整上下文 = "用存储换检索质量"。
- 滑窗 `stride < size` → 块间重叠 → 防关键信息落在切割边界被割裂。
- 语义切分在句子边界断，不死板按字数。

### 1.2 三路召回互补 —— 你项目最核心的亮点

| 召回路 | 模拟的真实组件 | 擅长 | 盲区 |
|---|---|---|---|
| **Dense 稠密** | Qwen3-Embedding | 语义相似，换个说法也能召回 | 精确术语、型号不敏感 |
| **Sparse 稀疏** | BGE-M3（词权重）| 关键词权重，兼顾一部分语义 | 介于两者之间 |
| **BM25 字面** | 手写标准 BM25 | 精确关键词（按钮名、错误码）| 不懂同义 |

**记住这个活证据**（tesla.py 里真实发生的）：
> 问"冬天准备"时，**Sparse 和 BM25 都被"充电"页干扰误召回了 p12，但 Dense 语义召回正确命中了 p93（预热电池）**，最后精排把 p93 排第一。
> **这就是多路召回互补：字面匹配失效时，语义召回兜底。** 面试讲这个例子，比背五个名词强十倍。

> 🎯 **必答追问**："BGE-M3 你用它哪部分？" → **sparse（词权重）那一路**（答不出=只会名词）。

### 1.3 RRF 融合 + Reranker 精排

```python
def rrf_fuse(rank_lists, k=60):
    for rl in rank_lists:
        for rank, (cid, _) in enumerate(rl):
            scores[cid] += 1.0 / (k + rank + 1)     # 只看排名倒数，不看原始分数
```

**两个分水岭面试点：**
1. **RRF 为什么只用排名不用分数？** → 三路分数尺度完全不同（余弦相似度 vs BM25 分）没法直接加；RRF 只用排名规避归一化问题。k 常用 60。
2. **有了 RRF 为什么还要 Reranker？**
   - 召回 = **bi-encoder**：query 和 doc 分开编码，快但糙，做粗筛。
   - Reranker = **cross-encoder**：query+doc 拼一起送模型，精但慢，只对少量候选算。
   - 两阶段：便宜召回粗筛几十条 → 贵的 Reranker 精排 top-k。**用速度换范围、用精度换质量。**

### 1.4 从 demo 升级到生产的替换清单
```
embed() 词袋 hash    → Qwen3-Embedding 真实模型
Python 循环算余弦     → Milvus ANN 索引
MockMongo 内存 dict  → pymongo (MongoDB)
rerank 重合度打分     → BGE-Reranker cross-encoder
generate 抽取式拼接   → 微调后的 LLM 生成
```

---

## 2. 把模型/RAG 变成服务：FastAPI + SSE（补短板核心）

参考 `jx3_bot/inference/server.py`。整体架构：
```
前端/QQ → FastAPI 推理服务 → Qwen2.5 + LoRA
                  ↓
          联网搜索（时效性问题）
```

### 2.1 一个最小推理 API

```python
app = FastAPI(title="问答助手 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)   # 跨域，前端能调

class ChatRequest(BaseModel):        # Pydantic：声明即校验 + 自动生成 /docs 文档
    question: str
    stream: bool = False
    max_new_tokens: int = 512

@app.post("/chat")
async def chat(req: ChatRequest):
    answer, ... = generate_answer(req.question, ...)
    return ChatResponse(answer=answer, ...)

@app.get("/health")                  # 健康检查
async def health():
    return {"status": "ok", "model_loaded": model is not None}
```

**三个后端基础（面试点）：**
- **Pydantic `BaseModel`**：请求体声明即校验，自动生成 OpenAPI 文档（访问 `/docs`）。
- **`/health` 为什么必须有？** → k8s 存活/就绪探针、负载均衡摘除故障节点都靠它。
- **CORS 中间件**：前端跨域调用的必备。

### 2.2 流式输出 SSE —— 体验的关键

```python
def generate_stream(question, ...):
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    thread = Thread(target=model.generate, kwargs={**inputs, "streamer": streamer, ...})  # 生成放子线程
    thread.start()
    for text_chunk in streamer:
        yield f"data: {json.dumps({'text': text_chunk}, ensure_ascii=False)}\n\n"   # SSE 格式
    yield "data: [DONE]\n\n"
# 路由：return StreamingResponse(generate_stream(...), media_type="text/event-stream")
```
- **为什么要流式？** → 降低 TTFT（首字延迟），用户不用等整段生成完。
- **为什么放 `Thread`？** → `model.generate` 是阻塞的，放子线程才能一边生成一边 `yield`。

### 2.3 加载模型：base + LoRA

```python
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
if lora_path:
    model = PeftModel.from_pretrained(model, lora_path)   # 基座 + LoRA，不必合并
model.eval()
```
> 🎯 **部署两种方式**：① `merge_and_unload()` 合并成独立模型（部署简单）；② base + adapter 分开（省存储、可热切换多个 adapter）。

---

## 3. 部署与工程化

### 3.1 Docker 打包（你没做过，重点学）

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt    # 先装依赖：代码变了不用重装（分层缓存）
COPY . .
EXPOSE 8000
CMD ["python", "inference/server.py", "--model", "/models/jx3_merged"]
```
- **为什么用 Docker？** → 环境一致（解决"在我机器上能跑"）、一键部署、隔离。
- GPU 容器要用 `nvidia/cuda` 基础镜像 + `docker run --gpus all`。

### 3.2 显存估算 —— 上线绕不开的账

| 场景 | 模型 | 方式 | 显存 |
|---|---|---|---|
| 推理 | Qwen2.5-14B | bf16 | ~30GB |
| 推理 | Qwen2.5-14B | int4 | ~10GB |
| 训练 | Qwen2.5-14B | QLoRA 4bit | ~24GB |
| 训练 | Qwen2.5-14B | LoRA bf16 | ~40GB |

**两个必背经验公式：**
- **推理显存 ≈ 参数量 × 2 字节**（bf16）+ KV Cache。
- **训练显存 ≈ 参数量 × 12～16 字节**（参数 + 梯度 + Adam 的 m/v 两份状态）。
- **降显存**：量化（int4 ~1/4）、QLoRA、KV Cache 优化、更小模型。

### 3.3 生产化清单
- `/health` 健康检查、日志、异常兜底（OOM 怎么办）、限流、超时。
- 真实生产推理用 **vLLM / TGI**（PagedAttention + Continuous Batching，吞吐比裸 HF 高 10~20 倍，第 4 节讲）。

---

## 4. 后端工程 sense：一个真实内存调优案例

参考 `memory_optimization/`（真实 Java SCADA 应用把内存压到 <600MB 的任务）。**你不用精通 Java，但要懂方法论和术语。**

### 三个可迁移的工程思维

1. **验收指标要可量化、口径要统一**：目标"RSS < 600MB"，明确主指标 `VmRSS`、辅助指标 `docker stats`、观察窗口第 5/15/30 分钟三次达标才算。
   → 和你 RAG 的 Recall@k、Agent 成功率是**同一个思维**：先定指标、定口径、定观察窗口。

2. **约束优先**：不改功能、可回滚、分阶段验证。生产优化铁律——**先保证不破坏，再谈优化**。

3. **排查靠数据不靠猜**：
   ```bash
   cat /proc/${JAVA_PID}/status | egrep 'VmRSS|VmSize|Threads'
   ```
   **RSS / VmSize / PSS 的区别**是后端面试常问的内存概念。

### 后端术语"够用"地图（不求精通，求能对话）

| 概念 | 一句话 | 你的现状 |
|---|---|---|
| REST / RPC | HTTP 接口 / 服务间高效调用(gRPC) | 会 FastAPI(REST) ✅ |
| 数据库 | MySQL / Redis / MongoDB / 向量库 | 用过 MongoDB、Milvus ✅ |
| 消息队列 | Kafka/RabbitMQ 解耦、削峰、异步 | 了解概念即可 |
| 容器化 | Docker 打包、K8s 编排 | 本节补 Docker ✅ |
| 高并发 | 缓存、限流、异步、水平扩展 | 了解概念即可 |

> **定心丸**：主攻 AI 应用岗**不需要你精通后端**，但要能说清"我的模型服务怎么部署、怎么扛并发、显存怎么控"。纯后端（元宝后台）不建议主投。

**练一个排查题**："RAG 服务上线后高峰变慢怎么办？"
→ 先看指标（延迟在检索还是生成）→ 加缓存（高频问题）→ 三路召回并行 → 水平扩展 → 换 vLLM 提吞吐。

---

## 5. 你的简历主打项目蓝图

```
项目名：个人知识库 Agent（或复用 Tesla 车书）
├── 检索层：三路召回(Dense+Sparse+BM25) + RRF + BGE-Reranker   ← tesla.py 升级
├── 生成层：LLM + 强 prompt（只基于检索答、带溯源、不知道就说不知道）
├── Agent 层：工具调用（查询改写/计算）+ 多轮记忆                ← 第 2 节 llmkit
├── 服务层：FastAPI + SSE 流式 + /health                        ← jx3_bot/server.py
├── 部署层：Docker 打包 + 显存估算 + 日志                        ← 本节
└── 文档层：README + 架构图 + 部署说明 + 效果量化               ← 面试要能展示
```

---

## 6. 课后作业

- [ ] 跑通 `tesla.py`，改一个问题，观察三路召回命中差异，写清"哪一路救了场"。
- [ ] 用 FastAPI 把第 2 节写的 Agent（或 tesla.py 检索）包成 `/chat` + `/health`，本地 curl 调通。
- [ ] 写一个最小 Dockerfile 打包，`docker run` 起来。
- [ ] 起草项目 README：架构图（用上面的蓝图）+ 一段"这个项目解决什么问题"。

---

## 7. 面试自检清单

1. 三种切分各解决什么？为什么一起用？
2. 三路召回各自盲区？举"字面失效语义兜底"的例子。
3. RRF 为什么只用排名不用分数？
4. 有 RRF 为什么还要 Reranker？bi-encoder vs cross-encoder？
5. Pydantic BaseModel 的作用？为什么要 `/health`？
6. SSE 流式怎么实现？为什么放子线程？为什么要流式？
7. LoRA 部署两种方式的取舍？
8. 推理/训练显存怎么估算？怎么降？
9. 为什么用 Docker？分层缓存是什么？
10. 服务变慢怎么排查？
