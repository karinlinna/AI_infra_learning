# 第 3 节课 · 教师教案：落地一个可上线的 Agent/RAG 应用项目（120 分钟）

> **学员**：王利田（主攻 LLM 应用/Agent 开发，**最大短板 = 后端部署工程**）
> **配套代码**：
> - `student/litianwang/tesla.py`（RAG 全链路骨架：多路召回 + RRF + Reranker，学员简历主打项目的可运行版）
> - `llm_learning/jx3_bot/`（真实的端到端项目：数据→LoRA 微调→FastAPI 服务→QQ 机器人，含流式输出）
> - `memory_optimization/`（真实的 Java 后端内存调优案例，用来补"后端工程 sense"）
> **本节目标**：① 把第 2 节的零件拼成一个**能上线**的项目蓝图；② 深挖 RAG 长板（tesla.py），让学员能讲透；③ 用 FastAPI + SSE 把模型暴露成服务、Docker 打包——**补上简历最缺的"后端部署"**；④ 建立后端工程的基本认知，能接住面试追问。
> **本节产出**：一个 GitHub 可展示、能本地/云端跑起来、含架构图 README 的项目规划。

---

## 时间轴总览

| 时段 | 模块 | 时长 | 配套 |
|---|---|---|---|
| 00:00–00:08 | 开场：什么叫"可上线" | 8 min | — |
| 00:08–00:38 | RAG 长板深挖（tesla.py 走读） | 30 min | tesla.py |
| 00:38–01:05 | 把模型/RAG 变成服务（FastAPI + SSE） | 27 min | jx3_bot/server.py |
| 01:05–01:30 | 部署与工程化（Docker / 显存 / 健康检查） | 25 min | jx3_bot |
| 01:30–01:52 | 后端工程短板补齐（内存调优真实案例） | 22 min | memory_optimization |
| 01:52–02:00 | 项目蓝图收口 + 作业 | 8 min | — |

---

## 00:00–00:08 ｜ 开场：什么叫"可上线"（8 min）

**开场话术：**
> "第 1 节课我们判断你最大的短板是**没有后端工程经验**、**Agent 项目缺落地**。今天这节课就是来补这块的。面试官问'你这个项目能上线吗？'——`demo` 和 `可上线` 差的不是模型，是**服务化、部署、可观测、能扛并发**这些'不出错时看不见'的东西。今天你会拿到一个能写进简历、能在 GitHub 展示、能被追问也不虚的项目蓝图。"

**板书 · demo vs 可上线的差距：**
```
demo（跑在 notebook 里）        可上线（别人能用）
  一个 .py 文件               →  REST/流式 API 服务
  手动跑                     →  Docker 一键部署
  出错就崩                   →  健康检查 + 重试 + 日志
  单次调用                   →  能扛并发、显存可控
  只有代码                   →  README + 架构图 + 部署文档
```

---

## 00:08–00:38 ｜ RAG 长板深挖：tesla.py 走读（30 min）

> **教学目标**：这是学员简历第一资产。第 2 节 L06 只讲了单路 RAG，这节把它升级成学员简历里的**三路召回 + RRF + Reranker**，且是**能跑的代码**（`python tesla.py`）。

**先跑：** `cd student/litianwang && python tesla.py`，让学员看三个问题的召回命中和精排结果。

### 关键点 1：切分策略——为什么三种一起用（8 min）

**带学员看 `build_chunks`（父子文档 + 滑动窗口 + 语义切分）：**
```python
def build_chunks(pages):
    parents, children = {}, []
    for pg in pages:
        pid = f"p{pg['page']}"
        parents[pid] = {"parent_id": pid, "page": pg["page"], "text": pg["text"], "figures": pg["figures"]}
        sents = semantic_split(pg["text"])                          # 语义切分：按句号断句
        for j, ck in enumerate(sliding_window_chunks(sents, size=2, stride=1)):  # 滑窗：块间重叠
            children.append({"chunk_id": f"{pid}_c{j}", "parent_id": pid, "page": pg["page"], "text": ck})
    return parents, children
```
**讲透（面试常问）：**
- **子块检索、父块喂 LLM**——`child` 粒度细检索准，命中后按 `parent_id` 回溯 `parent` 拿完整上下文。这就是"用存储换检索质量"。
- 滑窗 `stride < size` → 块间重叠 → 防止关键信息落在切割边界被割裂。
- 语义切分在句子边界断，不死板按字数。

### 关键点 2：三路召回互补——本项目最核心亮点（12 min）

**带学员对照三个召回函数：**
```python
def dense_recall(query, chunks, topk=5):   # 模拟 Qwen3-Embedding：懂语义，换说法也能召回
    qv = embed(query); return sorted([(c["chunk_id"], cosine(qv, embed(c["text"]))) for c in chunks], key=lambda x:-x[1])[:topk]
def sparse_recall(query, chunks, topk=5):  # 模拟 BGE-M3 稀疏权重：只有共现词才贡献分数
    ...
class BM25:                                # 字面召回：精确关键词匹配（型号、按钮名、错误码）
    ...
```

**必讲的教学高潮**（tesla.py 注释里就有，务必现场演示）：
> 跑第三个问题"冬天准备"时，**Sparse 和 BM25 都被"充电"页干扰误召回了 p12，但 Dense 语义召回正确命中了 p93（预热电池）**，最后精排把 p93 排到第一——**这就是多路召回互补的活证据：字面匹配失效时，语义召回兜底。**

**面试追问准备（让学员现场答）：**
- "BGE-M3 你用它哪部分？"→ sparse（词权重）那一路（答不出=只会名词）。
- "Dense、Sparse、BM25 各自的盲区是什么？"

### 关键点 3：RRF 融合 + Reranker 精排（10 min）

```python
def rrf_fuse(rank_lists, k=60):
    scores = defaultdict(float)
    for rl in rank_lists:
        for rank, (cid, _) in enumerate(rl):
            scores[cid] += 1.0 / (k + rank + 1)     # 只看排名倒数，不看原始分数
    return sorted(scores.items(), key=lambda x: -x[1])
```
**讲两个分水岭面试点：**
1. **RRF 为什么只用排名不用分数？** → 三路召回分数尺度完全不同（余弦相似度 vs BM25 分），没法直接加；RRF 只用排名天然规避归一化问题。k 常用 60。
2. **有了 RRF 为什么还要 Reranker？** → 召回是 **bi-encoder**（query 和 doc 分开编码，快但糙，做粗筛）；Reranker 是 **cross-encoder**（query+doc 拼一起送模型，精但慢，只对少量候选算）。两阶段：便宜召回粗筛几十条 → 贵的 Reranker 精排 top-k。

> **升级路线板书**（demo → 生产，tesla.py 注释里都标了）：
> ```
> embed() 词袋 hash      → Qwen3-Embedding 真实模型
> Python 循环算余弦       → Milvus ANN 索引
> MockMongo 内存 dict    → pymongo (MongoDB)
> rerank 重合度打分       → BGE-Reranker cross-encoder
> generate 抽取式拼接     → 微调后的 LLM 生成
> ```

---

## 00:38–01:05 ｜ 把模型/RAG 变成服务：FastAPI + SSE（27 min，补短板核心）

> **教学目标**：这是学员**最缺的一环**。用 `jx3_bot/inference/server.py` 这个真实服务做模板，讲清"怎么把一个模型变成别人能调的 API"。

**先看整体架构（jx3_bot README 的图）：**
```
QQ 群聊 → NoneBot2 机器人 → FastAPI 推理服务 → Qwen2.5-14B + LoRA
                                    ↓
                            DuckDuckGo 联网搜索（时效性问题）
```

### 关键点 1：一个最小的推理 API（10 min）

**带学员看 `server.py` 的骨架：**
```python
app = FastAPI(title="剑网三问答助手 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)   # 跨域，前端能调

class ChatRequest(BaseModel):        # Pydantic 定义请求体 = 自动校验 + 文档
    question: str
    stream: bool = False
    max_new_tokens: int = 512
    temperature: float = 0.7

@app.post("/chat")
async def chat(req: ChatRequest):
    answer, search_used, ctx = generate_answer(req.question, ...)
    return ChatResponse(answer=answer, search_used=search_used, search_context=ctx)

@app.get("/health")                  # 健康检查：k8s/负载均衡靠它判断服务活没活
async def health():
    return {"status": "ok", "model_loaded": model is not None}
```
**讲三个后端基础（面试点）：**
- **Pydantic `BaseModel`**：请求体声明即校验，自动生成 OpenAPI 文档（`/docs`）。
- **`/health` 健康检查**：为什么必须有？→ k8s 存活/就绪探针、负载均衡摘除故障节点都靠它。
- **CORS 中间件**：前端跨域调用的必备。

### 关键点 2：流式输出 SSE——体验的关键（10 min）

**带学员看流式生成（这段最能体现工程功底）：**
```python
def generate_stream(question, ...):
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    generation_kwargs = {**inputs, "max_new_tokens": max_new_tokens, "streamer": streamer, ...}
    thread = Thread(target=model.generate, kwargs=generation_kwargs)   # 生成放子线程，主线程边生成边吐
    thread.start()
    for text_chunk in streamer:
        yield f"data: {json.dumps({'text': text_chunk}, ensure_ascii=False)}\n\n"   # SSE 格式
    yield "data: [DONE]\n\n"

# 路由里：
return StreamingResponse(generate_stream(...), media_type="text/event-stream")
```
**讲清：**
- **为什么要流式？** → 降低 TTFT（首字延迟），用户不用等整段生成完（连回第 2 节 L08）。
- **SSE 格式**：`data: {...}\n\n`，`[DONE]` 收尾。为什么用 `Thread`？→ `model.generate` 是阻塞的，放子线程才能一边生成一边 `yield`。

### 关键点 3：加载模型 = base + LoRA（7 min）

```python
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
if lora_path:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, lora_path)   # 基座 + LoRA adapter，不必合并
model.eval()
```
**面试点**：部署两种方式——① `merge_and_unload()` 合并成独立模型（部署简单，推理不受影响）；② base + LoRA adapter 分开加载（省存储、可热切换多个 adapter）。`bfloat16` + `device_map="auto"` 是标配。

---

## 01:05–01:30 ｜ 部署与工程化（25 min）

### 关键点 1：Docker 打包（10 min）

**现场带学员写一个最小 Dockerfile（学员没做过，重点手把手）：**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "inference/server.py", "--model", "/models/jx3_merged"]
```
**讲清面试常问：**
- **为什么用 Docker？** → 环境一致（"在我机器上能跑"问题）、一键部署、隔离。
- 分层缓存：先 COPY requirements 再装依赖，代码变了不用重装依赖。
- GPU 容器要用 `nvidia/cuda` 基础镜像 + `--gpus all`。

### 关键点 2：显存估算——上线绕不开的账（10 min）

**用 jx3_bot README 的显存表讲（学员要能算这笔账）：**

| 场景 | 模型 | 方式 | 显存 |
|---|---|---|---|
| 推理 | Qwen2.5-14B | bf16 | ~30GB |
| 推理 | Qwen2.5-14B | int4 量化 | ~10GB |
| 训练 | Qwen2.5-14B | QLoRA 4bit | ~24GB |
| 训练 | Qwen2.5-14B | LoRA bf16 | ~40GB |

**给两个经验公式（面试必背，第 4 节还会深挖）：**
- **推理显存 ≈ 参数量 × 2 字节**（bf16）+ KV Cache。
- **训练显存 ≈ 参数量 × 12～16 字节**（参数 + 梯度 + Adam 的 m/v 两份状态）。
- **降显存手段**：量化（int4 降到 ~1/4）、QLoRA、KV Cache 优化、更小模型。

### 关键点 3：生产化清单（5 min）
- 健康检查 `/health`、日志、异常兜底（模型 OOM 怎么办）、限流、超时、`0.0.0.0` 对外暴露的安全。
- 推理加速：真实生产用 **vLLM / TGI**（PagedAttention + Continuous Batching，吞吐比裸 HF 高 10~20 倍，第 4 节讲）。

---

## 01:30–01:52 ｜ 后端工程短板补齐：一个真实内存调优案例（22 min）

> **教学目标**：学员简历没有后端工程经验。用 `memory_optimization/` 这个**真实的 Java SCADA 应用内存优化任务**，让学员建立"生产系统排查"的 sense——不需要她精通 Java，而是懂**方法论和术语**，面试被追问"你懂后端吗"时能接住。

**带学员读这个案例（`memory_optimization/2026-07-10-...md`）：**

### 讲三个可迁移的工程思维（10 min）

1. **验收指标要可量化、口径要统一**（案例里的精华）：
   > 目标是"Java 进程 RSS/RES 常驻内存 < 600MB"。案例明确了：主指标用 `/proc/<pid>/status` 的 `VmRSS`，辅助指标用 `docker stats`，**观察窗口**在 healthy 后第 5/15/30 分钟三次采样都达标才算。
   - **迁移到面试**："你怎么衡量项目效果？"——先定指标、定口径、定观察窗口。和 RAG 的 Recall@k、Agent 的成功率是同一个思维。

2. **约束优先：不改功能、可回滚、分阶段验证**：
   > "不改变原有功能、接口契约、数据库语义……所有变更必须可回滚。"
   - 生产系统优化的铁律：**先保证不破坏，再谈优化**。

3. **排查靠数据不靠猜**：
   ```bash
   JAVA_PID=$(docker exec app sh -c "pgrep -f 'java.*app.jar' | head -1")
   docker exec app sh -c "cat /proc/${JAVA_PID}/status | egrep 'VmRSS|VmSize|Threads'"
   docker exec app sh -c "cat /proc/${JAVA_PID}/smaps_rollup | egrep 'Rss|Pss|Private'"
   ```
   - **RSS/VmSize/PSS 的区别**是后端面试常问的内存概念。

### 讲后端基础术语地图（8 min，补认知）

学员投 AI 应用岗也会被问一点后端基础。给一张"够用"的地图（不求精通，求能对话）：

| 概念 | 一句话 | 学员现状 |
|---|---|---|
| REST / RPC | HTTP 接口 / 服务间高效调用（gRPC/Thrift）| 会 FastAPI(REST) |
| 数据库 | MySQL(关系)、Redis(缓存)、MongoDB(文档)、向量库(Milvus) | 用过 MongoDB、Milvus ✅ |
| 消息队列 | Kafka/RabbitMQ 解耦、削峰、异步 | 需了解概念 |
| 容器化 | Docker 打包、K8s 编排 | 本节补 Docker |
| 高并发 | 缓存、限流、异步、水平扩展 | 需了解概念 |

> **给学员定心丸**：主攻 AI 应用岗**不需要你精通后端**，但要能说清"我的模型服务怎么部署、怎么扛并发、显存怎么控"。今天补的就是这些。纯后端（元宝后台）不建议主投（第 1 节已判定）。

### 互动（4 min）
> "如果你的 RAG 服务上线后，高峰期响应变慢，你会怎么排查？"
> → 引导：先看指标（延迟在哪一步：检索 / 生成）→ 加缓存（高频问题）→ 三路召回并行 → 加机器水平扩展 → 换 vLLM 提吞吐。

---

## 01:52–02:00 ｜ 项目蓝图收口 + 作业（8 min）

**本节交付——学员简历主打项目蓝图（板书）：**
```
项目名：个人知识库 Agent（或复用 Tesla 车书）
├── 检索层：三路召回(Dense+Sparse+BM25) + RRF + BGE-Reranker   ← tesla.py 升级
├── 生成层：LLM + 强 prompt（只基于检索答、带溯源、不知道就说不知道）
├── Agent 层：工具调用（查询改写/计算）+ 多轮记忆                ← 第 2 节 llmkit
├── 服务层：FastAPI + SSE 流式 + /health                        ← jx3_bot/server.py
├── 部署层：Docker 打包 + 显存估算 + 日志                        ← 本节
└── 文档层：README + 架构图 + 部署说明 + 效果量化               ← 面试要能展示
```

**课后作业（第 4、5 节要用）：**
- [ ] 把 `tesla.py` 跑通，改一个问题，观察三路召回命中差异，写清"哪一路救了场"。
- [ ] 用 FastAPI 把你第 2 节写的 Agent（或 tesla.py 的检索）包成一个 `/chat` 接口 + `/health`，本地 curl 调通。
- [ ] 写一个最小 Dockerfile 把它打包，`docker run` 起来。
- [ ] 起草项目 README：架构图（用上面的蓝图）+ 一段"这个项目解决什么问题"。

> **结束语**："现在你有的不再是'一个跑在 notebook 里的 demo'，而是'一个能部署、能被调用、能讲清架构的项目'。后端部署这块短板，今天补上了。下节课我们回补算法基础，把面试八股也拿下。"

---

## 附：本节高频面试题清单

1. RAG 里三种切分（滑窗/父子/语义）各解决什么？为什么一起用？
2. 三路召回各自的盲区？为什么要三路？（举"字面失效语义兜底"的例子）
3. RRF 为什么只用排名不用分数？k 取多少？
4. 有 RRF 为什么还要 Reranker？bi-encoder vs cross-encoder 区别？
5. FastAPI 里 Pydantic BaseModel 的作用？为什么要 `/health`？
6. 流式输出 SSE 怎么实现？为什么要放子线程？为什么要流式？
7. LoRA 部署两种方式（合并 vs adapter）的取舍？
8. 推理/训练显存怎么估算？怎么降显存？
9. 为什么用 Docker？分层缓存是什么？
10. RSS / VmSize / PSS 的区别？服务变慢怎么排查？
