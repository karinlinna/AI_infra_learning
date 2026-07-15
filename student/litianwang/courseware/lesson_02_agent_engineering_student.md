# 第 2 节课 · 学生课件：LLM 应用与 Agent 工程实战

> **配套代码**：`AIagent/`（离线可跑、零依赖、无需 API Key）
> **先跑起来**：
> ```bash
> cd AIagent
> python run_all.py                        # 一键跑全部 9 课
> python lessons/lesson_04_agent_loop.py   # 单独跑某一课
> ```
> **本节目标**：上完这节课，你要能**不看教程独立写出一个带工具调用和记忆的 Agent**，并说清 Agent Loop 每一步在干什么。

---

## 0. 一句话看懂 Agent

> **Agent = 模型 + 工具 + 循环 + 记忆 + 知识 + 护栏**

别把 Agent 想成玄学。它就是一个循环：**模型说要用什么工具 → 我们执行 → 把结果喂回去 → 循环到它说"不用了"。** 这 9 节课就是把这个公式的每一块拆开给你看。

### 先建立框架心智模型：`llmkit`

整套课程共用一个极简框架 `llmkit`，核心是一条**依赖倒置**的接缝：

```
你的代码（Agent / RAG / 多智能体）
        │  只依赖
        ▼
   LLMProvider（抽象基类）      ← 换模型只改这里下面
        │
   ┌────┼────┐
 Mock  Claude  OpenAI/兼容端点(DeepSeek/通义/vLLM…)
```

**记住三件事（都是面试点）：**
1. **换模型只改一个环境变量 `LLM_PROVIDER`**——这就是依赖倒置，上层永远只认 `LLMProvider`。
2. **默认 `MockProvider`**：离线、确定性、零依赖，不烧 token 就能跑通全部课程。它是"可测试替身"，生产 Agent 工程必备。
3. **缓存、追踪都是"包一层 Provider"**——代理/装饰器模式，后面会出现好几次，你要能认出来。

---

## 1. 裸调用：LLM 是无状态函数（L01）

**核心**：LLM 是"给定对话，预测下一段文本"的**无状态函数**。API 不记得上一轮，历史每次自己带上。

```python
history = [Message.system("你是助手，请记住用户告诉你的信息。")]
def ask(text):
    history.append(Message.user(text))
    resp = llm.chat(history)
    history.append(Message.assistant(resp.content))  # ← 关键：把模型回复也存回历史
ask("我叫李雷。"); ask("我叫什么名字？")   # 第二问答对，唯一原因是带上了第一轮历史
```

> 🎯 **面试题**："ChatGPT 为什么能记住上文？" → **不是 API 有记忆，是客户端每次把完整历史重发。** 所以对话越长越贵越慢（token 随历史线性增长）。
> ⚠️ **常见 bug**：忘记把 assistant 回复存回 history → 模型"失忆"。

---

## 2. 提示工程 & 结构化输出（L02）

**提示工程四支柱**：角色设定 / 清晰指令 / 少样本(few-shot) / 输出约束。

**为什么要结构化输出**：Agent 里模型输出要喂给下游程序，自由文本没法可靠解析，必须 JSON。

**核心套路——自我修复循环（超常用）：**
```python
for attempt in range(max_retries + 1):
    resp = llm.chat(messages)
    try:
        return extract_json(resp.content)         # 抠 JSON 再解析
    except (ValueError, json.JSONDecodeError) as e:
        if attempt == max_retries: raise
        messages.append(Message.assistant(resp.content))
        messages.append(Message.user(f"输出无法解析为 JSON（{e}）。请只输出合法 JSON。"))  # 回灌纠错
```

**两个关键点：**
- 别直接 `json.loads(resp.content)`——模型常包 ```` ```json ```` 围栏或加解释文字会炸，要先抠 `{...}`。
- **"把错误回灌给模型让它自己纠错"是通用套路**——工具参数错、代码错都能这么修。

> 🎯 **面试点**：生产中优先用厂商原生"结构化输出/严格模式"（更稳），本课教的是通用兜底原理。

---

## 3. 工具调用 / Function Calling（L03）

**核心**：工具 = 把模型"意图"翻译成对真实世界的操作。**这是从聊天机器人迈向 Agent 的关键一步。**

**完整四步握手：**
```
1. 给模型工具 schema  →  2. 模型返回 tool_use（我要调 X，参数 {...}）
      ↓
4. 模型基于结果作答  ←  3. 我们执行工具，结果回传（带 tool_call_id）
```

**`@tool` 装饰器从函数自动生成 schema：**
```python
@tool
def get_weather(city: str) -> str:
    """查询城市天气"""          # ← docstring 首行 = 工具描述，模型靠它判断何时调用
    ...
# 装饰器自动把 city:str 变成 {"type":"string"}，无默认值的参数进 required
```

**三个面试点：**
1. **schema 的 description 直接决定模型是否/何时正确调用**——这是"提示工程在工具层的延伸"。
2. **工具异常必须捕获并回传给模型**（而不是抛出中断），让模型有机会纠错。
3. **安全**：别用 `eval` 直接执行模型给的表达式（注入风险）。演示代码用 `eval(expr, {"__builtins__": {}}, {})` 做了沙箱化并注明"生产勿直接 eval"。

---

## 4. Agent Loop / ReAct —— 本节最重要（L04）

**核心**：Agent = 模型 + 工具 + "思考→行动→观察"循环。一个任务可能需连续多次工具调用（先查汇率再算总价）。**所有花哨的 Agent 框架，内核都是这个循环。**

```python
for step in range(1, self.max_steps + 1):
    resp = self.llm.chat(messages, tools=specs)
    if not resp.wants_tools:                        # 模型不再要工具 → 收尾返回
        return AgentResult(resp.content, step, total, messages)
    messages.append(Message.assistant(resp.content, tool_calls=resp.tool_calls))  # 记录意图
    for tc in resp.tool_calls:                       # 逐个执行，按 id 回传
        result = self.tools.call(tc.name, tc.arguments)
        messages.append(Message.tool(result, tc.id, tc.name))
# 触达 max_steps → 安全退出（生产中应告警）
```

> 🎯 **必考面试题**："去掉 `max_steps` 会怎样？" → **模型可能陷入无限工具调用死循环，烧光预算。`max_steps` 是生产必备护栏。**
> ⚠️ **坑**：消息顺序严格——`assistant(含 tool_calls)` 必须在对应 `tool` 结果**之前**，否则 API 报错。

**记住这句板书**：`Agent Loop = while 模型要工具 { 执行 → 回传 }，max_steps 兜底`

---

## 5. 记忆 Memory（L05）

短期记忆 = 对话历史，但上下文窗口有限、越长越贵越慢。三条路：

| 策略 | 做法 | 代价 |
|---|---|---|
| **滑动窗口** | 保留最近 N 轮（**但始终保留 system**）| 可能丢中间信息 |
| **摘要压缩** | 让模型总结旧对话成一条 system 消息 | 多一次 LLM 调用 + 有损 |
| **长期记忆** | 跨会话持久化（JSON/向量库），新会话注入 | 需外部存储 |

```python
def summarize_history(llm, history):
    convo = "\n".join(f"{m.role}: {m.content}" for m in history if m.role != "system")
    resp = llm.chat([Message.system("把对话压缩成不超过 50 字的要点，只保留关键事实。"),
                     Message.user(convo)])
    return Message.system(f"[历史摘要] {resp.content}")  # 一条摘要替换掉冗长原文
```

> 🎯 **经典面试题**："上下文满了怎么办？" → **截断 / 滑窗 / 摘要 / 外部记忆**四条路，各有取舍。
> ⚠️ 滑动窗口别把 system 也滑掉，否则 Agent 丢人设/指令。

---

## 6. RAG 检索增强（L06）—— 你 Tesla 项目的骨架

**核心**：模型知识有截止日期、不懂你的私有资料、会幻觉。RAG = 把相关资料检索出来塞进提示，让模型**开卷答题**。五步：切块→向量化→建库→检索→拼接生成。

```python
hits = store.search(question, k=2)                              # 1) 检索 top-k
context = "\n".join(d.text.strip() for d, _ in hits)          # 2) 拼接
messages = [Message.system("只根据下面资料回答，资料没有就说不知道，不要编造。\n\n资料:\n" + context),
            Message.user(question)]
resp = llm.chat(messages)                                       # 3) 生成
```

**三个面试点：**
1. **切块为什么要 overlap？** → 避免把一句话从中间切断丢语义。
2. **抗幻觉的关键约束**：system prompt 里"资料没有就说不知道"。
3. **玩具 vs 生产**：哈希 embedding→真实 embedding API，内存向量库→FAISS/Milvus，**检索+拼接逻辑完全不变**——这就是抽象的价值。

> 🔗 **和你简历的连接**：这个单路 `store.search` 就是你 Tesla 项目的最小骨架。你的**三路召回 + RRF + Reranker** = 把这里升级成"三路检索 + 融合 + 精排"。第 3、4 节深挖。

---

## 7. 多智能体（L07）

**核心**：单 Agent 塞太多职责会"样样通样样松"。拆成各有专长的 Agent 更可控可测。

```python
researcher = Agent(llm, tools=ToolRegistry(search_web),
                   system="你是研究员。收集资料，只输出要点清单。", max_steps=3)
writer   = Agent(llm, system="你是作家。把要点扩写成短文。", max_steps=1)
reviewer = Agent(llm, system="你是审校。检查并输出定稿。", max_steps=1)
notes = researcher.run(...).answer                    # 通信 = 自然语言传中间产物
draft = writer.run(f"根据要点写短文：\n{notes}").answer
final = reviewer.run(f"审校并定稿：\n{draft}").answer
```

**两种经典编排**：流水线（A→B→C）、协调者/路由（主管派活）。

> 🎯 **面试点**：多智能体 vs 单 Agent 多工具的边界？流水线的**错误传播**（研究员错了会污染下游）。
> **技巧**：同一个 `Agent` 类，仅靠 **system prompt 差异**就能造出不同角色——最朴素也最稳。

---

## 8. 优化与性能调优（L08）—— demo→生产的一跃

生产 Agent 围绕三维度：**延迟 / 成本 / 稳定**。七个手段：

| 维度 | 手段 |
|---|---|
| 延迟 | ① 流式（降 TTFT 首字延迟）② 并发线程池（提吞吐）|
| 成本 | ③ 精确缓存（内容哈希）④ 语义缓存（向量相似度阈值）⑤ 模型路由（简单→便宜模型）⑥ 上下文裁剪 + 提示缓存（稳定前缀放最前省 ~90% 输入成本）|
| 稳定 | ⑦ 重试指数退避 + jitter |

```python
# 并发提吞吐：串行 8 个请求 ~1.6s，并发 ~0.2s，约 8x
with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(lambda q: llm.chat([Message.user(q)]), questions))
```

> 🎯 **面试点**：
> - **精确缓存 vs 语义缓存**：语义缓存命中率高，但阈值太低会返回**错答案**。
> - **jitter 为什么关键**？→ 避免大量客户端同一时刻重试造成"惊群"。
> - Python 有 GIL，但 LLM 调用是 **I/O 密集**（等网络），线程池仍然有效。

---

## 9. 可观测性 & 评估（L09）

**核心**："没有观测就无法优化——你无法改进你看不见的东西。"

```python
class Tracer(LLMProvider):        # ← 又一个代理模式！和 L08 的 CachingProvider 是同一个套路
    def chat(self, messages, ...):
        t0 = time.time(); ok = True
        try:    resp = self.inner.chat(...); return resp
        except Exception: ok = False; raise
        finally:  # ← 埋点放 finally：成功和失败都要记录（失败率是关键指标）
            self.spans.append(Span("chat", (time.time()-t0)*1000, resp.usage if ok else Usage(), ok))
```

**三件事**：埋点追踪（延迟/token/成败）+ 成本核算（输入/输出 token 单价不同，output 贵数倍）+ 评估（用例+期望打分，= Agent 版回归测试，防迭代退化）。

> 💡 **发现这个 pattern**：`CachingProvider`（L08）和 `Tracer`（L09）都是"包一层 `inner` Provider"的**代理模式**。看懂它，你就理解了生产 Agent 怎么加功能（缓存/追踪/重试）而不改业务代码。

---

## 10. 本节作业（第 3 节要用）

- [ ] **不看教程**，用 `llmkit` 独立写一个 Agent：≥2 个自定义工具（如"查天气"+"计算器"）+ 短期记忆（多轮），跑通一个需要**连续两次工具调用**的任务。
- [ ] 把 `MockProvider` 换成真实模型（改 `LLM_PROVIDER`，用低价的 DeepSeek/通义），观察 L02 结构化输出、L06 RAG 的**真实效果**（Mock 是模板化输出，看不到理想效果）。
- [ ] 写 200 字说明你的 Agent Loop 每一步在干什么。

---

## 11. 面试自检清单（能答上算过关）

1. ChatGPT 怎么记住上文？
2. 为什么 Agent Loop 要 `max_steps`？
3. 工具调用完整四步握手？
4. `@tool` 的 schema description 为什么重要？
5. 上下文满了怎么办？（四条路）
6. RAG 切块为什么要 overlap？RAG vs 微调 vs 长上下文怎么选？
7. 精确缓存 vs 语义缓存的取舍？
8. 重试为什么要加 jitter？
9. 代理/装饰器模式在这套框架里出现在哪几处？
10. 为什么要用 Mock？
