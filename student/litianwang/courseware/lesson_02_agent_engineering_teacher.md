# 第 2 节课 · 教师教案：LLM 应用与 Agent 工程实战（120 分钟）

> **学员**：王利田（主攻 LLM 应用/Agent 开发）
> **配套代码**：`AIagent/`（一套渐进式、离线可跑、零依赖的 Agent 教程，9 节课 + `llmkit` 框架）
> **本节目标**：① 建立"Agent = 模型 + 工具 + 循环 + 记忆 + 知识"的完整心智模型；② 跑通 `lessons/01–09` 全链路；③ 学员能不看教程独立写出一个带工具调用和记忆的 Agent；④ 说清 ReAct Loop 每一步在干什么。
> **课堂形式**：跑代码看现象 → 讲原理 → 挖面试点 → 学员当场改代码。**每个知识点都先跑再讲。**

> **开课前准备（老师）**：
> ```bash
> cd AIagent
> python run_all.py          # 确认 9 课全部能离线跑通（默认 Mock，无需 API Key）
> ```

---

## 时间轴总览

| 时段 | 模块 | 时长 | 对应课程 |
|---|---|---|---|
| 00:00–00:08 | 开场：Agent 到底是什么 + 框架心智模型 | 8 min | llmkit 架构 |
| 00:08–00:25 | 基础：裸调用 + 提示工程/结构化输出 | 17 min | L01–02 |
| 00:25–00:50 | 核心：工具调用 + Agent Loop（ReAct） | 25 min | L03–04 |
| 00:50–01:15 | 增强：记忆 + RAG | 25 min | L05–06 |
| 01:15–01:35 | 进阶：多智能体 | 20 min | L07 |
| 01:35–01:55 | 生产：优化 + 可观测性 | 20 min | L08–09 |
| 01:55–02:00 | 学员实操作业 + 总结 | 5 min | — |

---

## 00:00–00:08 ｜ 开场：Agent 是什么 + 框架心智模型（8 min）

**开场话术：**
> "很多人以为 Agent 是玄学。今天我用一套能离线跑的代码告诉你——**Agent 就是一个循环**：模型说要用什么工具，我们执行，把结果喂回去，循环到它说'不用了'为止。9 节课我们从'裸调用大模型'一路搭到'生产级 Agent'，每节课都能跑。这套代码你能讲透，AI 应用开发岗的面试你就稳了一半。"

**先建立框架心智模型（关键，后面所有课都靠它）：**

板书 `llmkit` 的核心设计——**依赖倒置**：
```
上层代码（Agent / RAG / 多智能体）
        │  只依赖
        ▼
   LLMProvider（抽象基类）  ← 框架的"接缝"
        │  实现
   ┌────┼────┐
 Mock  Claude  OpenAI/兼容端点
（离线）        （DeepSeek/通义/vLLM…）
```

**讲三句话（每句都是面试点）：**
1. "换模型只改一个环境变量 `LLM_PROVIDER`——这就是**依赖倒置**，上层永远只认 `LLMProvider` 抽象。"
2. "默认走 `MockProvider`，离线、确定性、零依赖，不烧 token 就能跑通全部课程——这是**可测试替身（test double）**，生产 Agent 工程必备。"
3. "缓存、追踪都是'包一层 Provider'——**代理/装饰器模式**，横切关注点不污染业务代码。等下你会看到它出现好几次。"

> **板书**：`Agent = 模型 + 工具 + 循环 + 记忆 + 知识 + 护栏`

---

## 00:08–00:25 ｜ 基础：裸调用 + 结构化输出（17 min）

### L01 裸调用：LLM 是无状态函数（7 min）

**先跑：** `python lessons/lesson_01_bare_call.py`

**核心一句话：** LLM 本质是"给定对话，预测下一段文本"的**无状态函数**。API 不记得上一轮，历史要每次自己带上。

**带学员看这段代码（多轮对话手动维护历史）：**
```python
history = [Message.system("你是助手，请记住用户告诉你的信息。")]
def ask(text):
    history.append(Message.user(text))
    resp = llm.chat(history)
    history.append(Message.assistant(resp.content))  # ← 把模型回复也存回历史
ask("我叫李雷。"); ask("我叫什么名字？")   # 第二问能答对，唯一原因是带上了第一轮
```

**互动提问：**
> "ChatGPT 为什么能记住你上一句话？是 API 有记忆吗？"
> → **正确答案**：不是。是客户端每次把**完整历史重发**。这也解释了为什么对话越长越贵越慢（token 随历史线性增长）。

**面试点**：忘记把 assistant 回复存回 history → 模型"失忆"，这是新手常见 bug。

### L02 提示工程 & 结构化输出（10 min）

**先跑：** `python lessons/lesson_02_prompt_and_structured.py`

**讲提示工程四支柱**：角色设定 / 清晰指令 / 少样本(few-shot) / 输出约束。

**核心代码——自我修复循环（self-healing loop，Agent 工程极常用）：**
```python
for attempt in range(max_retries + 1):
    resp = llm.chat(messages)
    try:
        return extract_json(resp.content)          # 抠 JSON 再解析
    except (ValueError, json.JSONDecodeError) as e:
        if attempt == max_retries: raise
        messages.append(Message.assistant(resp.content))
        messages.append(Message.user(f"上面的输出无法解析为 JSON（{e}）。请只输出合法 JSON。"))
```

**讲两个关键点：**
1. **别直接 `json.loads(resp.content)`**——模型常包 ```` ```json ```` 围栏或加解释文字，会炸。要先 `find("{")`/`rfind("}")` 抠出来。
2. **"把错误回灌给模型让它自己纠错"是通用套路**——不止 JSON，工具参数错、代码错都能这么修。

**面试点**：生产中优先用厂商原生"结构化输出/严格模式"（更稳），本课教的是通用兜底原理。为什么要结构化？因为 Agent 里模型输出要喂给下游程序，自由文本没法可靠解析。

---

## 00:25–00:50 ｜ 核心：工具调用 + Agent Loop（25 min，本节重心）

### L03 工具调用 / Function Calling（12 min）

**先跑：** `python lessons/lesson_03_tools.py`

**核心一句话：** 工具 = 把模型"意图"翻译成对真实世界的操作。这是"从聊天机器人迈向 Agent 的关键一步"。

**讲完整四步握手（板书）：**
```
1. 给模型工具 schema  →  2. 模型返回 tool_use（我要调 calculator，参数 {...}）
      ↓
4. 模型基于结果作答  ←  3. 我们执行工具，把结果回传（带 tool_call_id）
```

**核心代码——`@tool` 装饰器从函数自动生成 schema：**
```python
def tool(fn):
    sig = inspect.signature(fn)
    for pname, p in sig.parameters.items():
        json_type = _PY_TO_JSON.get(p.annotation, "string")   # int→integer, str→string
        props[pname] = {"type": json_type, "description": f"参数 {pname}"}
        if p.default is inspect.Parameter.empty: required.append(pname)
    desc = (fn.__doc__ or fn.__name__).strip().splitlines()[0]  # docstring 首行当描述
    return Tool(spec=ToolSpec(name=fn.__name__, description=desc, parameters={...}), func=fn)
```

**讲三个面试点：**
1. **schema 的 description 直接决定模型是否/何时正确调用**——这是"提示工程在工具层的延伸"。
2. **工具异常必须捕获并回传给模型**（而不是抛出中断）——让模型有机会纠错。
3. **安全**：演示里 calculator 用 `eval(expr, {"__builtins__": {}}, {})`，注释明确标"生产勿直接 eval"（eval 注入风险，安全面试点）。

### L04 Agent Loop / ReAct（13 min，最重要）

**先跑：** `python lessons/lesson_04_agent_loop.py`

**核心一句话：** Agent = 模型 + 工具 + "思考→行动→观察"循环。一个任务可能需连续多次工具调用（先查汇率再算总价）。**所有花哨的 Agent 框架，内核都是这个循环。**

**核心代码——循环本体（`llmkit/agent.py`）：**
```python
for step in range(1, self.max_steps + 1):
    resp = self.llm.chat(messages, tools=specs)
    if not resp.wants_tools:                       # 模型不再要工具 → 收尾
        return AgentResult(resp.content, step, total, messages)
    messages.append(Message.assistant(resp.content, tool_calls=resp.tool_calls))  # 记录意图
    for tc in resp.tool_calls:                      # 逐个执行，按 tool_call_id 回传
        result = self.tools.call(tc.name, tc.arguments)
        messages.append(Message.tool(result, tc.id, tc.name))
# 触达步数上限 → 安全退出（生产中应告警）
```

**互动提问（关键面试题）：**
> "如果我把 `max_steps` 去掉会怎样？"
> → **正确答案**：模型可能陷入无限工具调用死循环，烧光预算。`max_steps` 是**生产必备护栏**。

**再讲两个坑：**
- 消息追加顺序严格：`assistant(含 tool_calls)` 必须在对应 `tool` 结果**之前**，否则厂商 API 报错。
- `on_event` 回调把可观测性做成一等公民（为 L09 埋点铺路）。

> **本模块小结板书**：`Agent Loop：while 模型要工具 { 执行 → 回传 }，max_steps 兜底`

---

## 00:50–01:15 ｜ 增强：记忆 + RAG（25 min）

### L05 记忆 Memory（12 min）

**先跑：** `python lessons/lesson_05_memory.py`

**核心：** 短期记忆=对话历史本身，但上下文窗口有限、越长越贵越慢。三条应对路：

| 策略 | 做法 | 代价 |
|---|---|---|
| 滑动窗口 | 保留最近 N 轮（但**始终保留 system**）| 可能丢中间信息 |
| 摘要压缩 | 让模型总结旧对话成一条 system 消息 | 多一次 LLM 调用 + 有损 |
| 长期记忆 | 跨会话持久化（JSON/向量库），新会话注入 | 需外部存储 |

**核心代码——摘要压缩（用模型压缩模型的历史）：**
```python
def summarize_history(llm, history):
    convo = "\n".join(f"{m.role}: {m.content}" for m in history if m.role != "system")
    resp = llm.chat([Message.system("把下面的对话压缩成不超过 50 字的要点，只保留关键事实。"),
                     Message.user(convo)])
    return Message.system(f"[历史摘要] {resp.content}")  # 一条摘要替换掉冗长原文
```

**面试点**：滑动窗口若把 system 也滑掉 → Agent 丢人设/指令，本课刻意分开处理。经典面试题"**上下文满了怎么办？**"→ 截断/滑窗/摘要/外部记忆四条路，各有取舍。

### L06 RAG 检索增强（13 min）

**先跑：** `python lessons/lesson_06_rag.py`（看 `demo_without_vs_with_rag`：无 RAG 答不出私有知识 → 注入后答对）

**核心：** 模型知识有截止日期、不懂你的私有资料、会幻觉。RAG = 把相关资料检索出来塞进提示，让模型**开卷答题**。五步：切块→向量化→建库→检索→拼接生成。

**核心代码——RAG 三步：**
```python
hits = store.search(question, k=2)                               # 1) 检索 top-k
context = "\n".join(d.text.strip() for d, _ in hits)            # 2) 拼接
messages = [Message.system("只根据下面资料回答，资料没有就说不知道，不要编造。\n\n资料:\n" + context),
            Message.user(question)]
resp = llm.chat(messages)                                        # 3) 生成
```

**讲三个面试点（连到学员的 Tesla 项目）：**
1. **切块为什么要 overlap？** → 避免把一句话从中间切断丢语义。
2. **抗幻觉的关键约束**：system prompt 里"资料没有就说不知道"。
3. **玩具 vs 生产**：`HashingEmbedder`→真实 embedding API，内存向量库→FAISS/Milvus，**检索+拼接逻辑完全不变**——这正是抽象的价值。

> **关键衔接**：这一课是学员 Tesla RAG 项目的"最小骨架"。老师点明："你简历里的多路召回 + RRF + Reranker，就是把这里的单路 `store.search` 升级成三路 + 融合 + 精排。第 3、4 节课我们深挖。"

---

## 01:15–01:35 ｜ 进阶：多智能体（20 min）

**先跑：** `python lessons/lesson_07_multi_agent.py`

**核心：** 单 Agent 塞太多职责会"样样通样样松"。拆成各有专长的 Agent 更可控可测。两种经典编排：

**核心代码——流水线（研究→写作→审校，靠 system prompt 分化角色）：**
```python
researcher = Agent(llm, tools=ToolRegistry(search_web),
                   system="你是研究员。使用搜索工具收集资料，只输出要点清单。", max_steps=3)
writer   = Agent(llm, system="你是作家。把要点扩写成通顺短文。", max_steps=1)
reviewer = Agent(llm, system="你是审校。检查通顺、有无问题，输出定稿。", max_steps=1)
notes = researcher.run(...).answer          # 通信 = 自然语言传中间产物
draft = writer.run(f"根据要点写短文：\n{notes}").answer
final = reviewer.run(f"审校并定稿：\n{draft}").answer
```

**讲三个点：**
1. **同一个 `Agent` 类，仅靠 system prompt 差异**就能造出不同角色——最朴素也最稳。
2. `max_steps` 按职责设：研究员要用工具给 3，写作/审校纯生成给 1。
3. 另一种模式：**协调者/路由**（主管 Agent 按任务派活），本课用关键词路由演示骨架。

**面试点**：多智能体 vs 单 Agent 多工具的边界？流水线的**错误传播**（研究员错了会污染下游）。

---

## 01:35–01:55 ｜ 生产：优化 + 可观测性（20 min，demo→生产的一跃）

### L08 优化与性能调优（12 min）

**先跑：** `python lessons/lesson_08_optimization.py`

**核心：** 生产 Agent 围绕三维度——**延迟 / 成本 / 稳定**。

**核心代码——并发提吞吐（Mock 带 latency 直观演示）：**
```python
llm = MockProvider(latency=0.2); questions = [f"问题{i}" for i in range(8)]
# 串行 ~1.6s；并发线程池 ~0.2s，约 8x
with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(lambda q: llm.chat([Message.user(q)]), questions))
```

**七个手段速查（板书）：**
1. 流式 → 降 TTFT（首字延迟）｜ 2. 并发 → 提吞吐 ｜ 3. 精确缓存（内容哈希）｜ 4. 语义缓存（向量相似度阈值）｜ 5. 模型路由（简单→便宜模型）｜ 6. 上下文裁剪 + 提示缓存（稳定前缀放最前省 ~90% 输入成本）｜ 7. 重试指数退避 + jitter。

**两个面试点：**
- **精确缓存 vs 语义缓存权衡**：语义缓存命中率高，但**阈值太低会返回错答案**。
- **jitter 为什么关键？** → 避免大量客户端同一时刻重试造成"惊群"。
- Python 并发受 GIL 限制，但 LLM 调用是 **I/O 密集**（等网络），线程池仍有效。

### L09 可观测性 & 评估（8 min）

**先跑：** `python lessons/lesson_09_observability.py`

**核心一句话：** "没有观测就无法优化——你无法改进你看不见的东西。"

**核心代码——Tracer（又一个代理模式 + `try/finally` 保证异常也埋点）：**
```python
class Tracer(LLMProvider):
    def chat(self, messages, ...):
        t0 = time.time(); ok = True
        try:    resp = self.inner.chat(...); return resp
        except Exception: ok = False; raise
        finally:
            self.spans.append(Span("chat", (time.time()-t0)*1000, resp.usage if ok else Usage(), ok))
```

**讲两个点：**
1. **埋点放 `finally`**——成功和失败都要记录（失败率是关键指标）。
2. **成本估算**：输入/输出 token 单价不同（output 通常贵数倍）。
3. **评估 = Agent 版回归测试**：用例 + 期望关键词给 Agent 打分，防迭代退化。实践中用 "LLM-as-judge"。

> **点睛**：让学员发现——`CachingProvider`（L08）和 `Tracer`（L09）是**同一个代理模式**。"看出这个 pattern，你就理解了生产 Agent 怎么加功能而不改业务代码。"

---

## 01:55–02:00 ｜ 学员实操作业 + 总结（5 min）

**本节主线复述：** `Agent = 模型(L01) + 提示/结构化(L02) + 工具(L03) + 循环(L04) + 记忆(L05) + 知识(L06) + 协作(L07) + 优化(L08) + 观测(L09)`。

**课后作业（下节课前完成，第 3 节要用）：**
- [ ] **不看教程**，用 `llmkit` 独立写一个 Agent：至少 2 个自定义工具（如"查天气"+"计算器"）+ 短期记忆（多轮），跑通一个需要连续两次工具调用的任务。
- [ ] 把 `MockProvider` 换成真实模型（改 `LLM_PROVIDER`，用免费/低价的 DeepSeek 或通义），观察 L02 结构化输出、L06 RAG 的**真实效果**（Mock 是模板化输出看不到理想效果）。
- [ ] 写一段 200 字说明：你的 Agent Loop 每一步在干什么（面试要能讲）。

> **结束语**："今天你把 Agent 的每一层都拆开看了一遍。下节课我们把这些拼成一个**能上线的项目**——补上你简历最缺的那块：后端部署。"

---

## 附：本节高频面试题清单（老师可随堂抽查）

1. ChatGPT 怎么记住上文？（客户端重发历史，API 无状态）
2. 为什么 Agent Loop 要 `max_steps`？（防死循环烧钱）
3. 工具调用完整四步握手是什么？
4. `@tool` 的 schema description 为什么重要？
5. 上下文满了怎么办？（截断/滑窗/摘要/外部记忆）
6. RAG 切块为什么要 overlap？RAG vs 微调 vs 长上下文怎么选？
7. 精确缓存 vs 语义缓存的取舍？
8. 重试为什么要加 jitter？
9. 代理/装饰器模式在这套框架里出现在哪几处？（Caching、Tracer）
10. 为什么要用 Mock？（离线、不烧 token、可复现异常路径）
