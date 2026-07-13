# AI Agent 开发教学 Demo

一套**渐进式、可离线运行**的 AI Agent 开发教程。从"裸调用大模型"一路讲到"生产级性能调优与可观测性"，覆盖 Agent 开发的核心知识点。

核心设计：一个可切换后端的极简框架 **`llmkit`** + **9 节循序渐进的课程**。默认使用**离线 Mock 后端**，学员**无需 API Key、无需联网、零第三方依赖**即可跑通全部课程；改一个环境变量即可接入真实的 Claude 或 OpenAI/国产/本地模型。

---

## 快速开始

```bash
# 1. 无需安装任何依赖，直接跑（默认离线 Mock）
python run_all.py            # 一键运行全部 9 课
python lessons/lesson_01_bare_call.py   # 或单独运行某一课

# 2. 想接真实模型：安装 SDK 并设置环境变量
pip install -r requirements.txt
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-xxxx
python lessons/lesson_04_agent_loop.py
```

切换后端只需改环境变量（详见 `.env.example`）：

| `LLM_PROVIDER` | 说明 |
|---|---|
| `mock`（默认） | 离线、确定性、零依赖，课堂即开即用 |
| `anthropic` | Claude（`claude-opus-4-8`，自适应思考） |
| `openai` | OpenAI 及一切 OpenAI 兼容端点（DeepSeek / 通义 / Kimi / vLLM / Ollama，改 `OPENAI_BASE_URL` 即可） |

---

## 课程大纲与知识点地图

| 课 | 主题 | 覆盖的核心知识点 |
|---|---|---|
| **01** | 裸调用 LLM | 对话补全、system/user/assistant 角色、**API 无状态**、多轮历史管理、流式输出 |
| **02** | 提示工程 & 结构化输出 | 提示工程四支柱、少样本、**JSON 结构化输出**、解析校验、**自我修复重试循环** |
| **03** | 工具调用 | Function Calling 完整握手、`@tool` 自动生成 schema、工具执行与结果回传 |
| **04** | Agent 循环 | **ReAct 循环**（思考→行动→观察）、`max_steps` 护栏、工具异常处理、事件回调 |
| **05** | 记忆 | 短期记忆、**上下文管理**（滑动窗口 / 摘要压缩）、跨会话**长期记忆**持久化 |
| **06** | RAG 检索增强 | 切块→向量化→建库→检索→拼接生成、离线 embedding、有/无 RAG 对比 |
| **07** | 多智能体 | **流水线协作**（研究→写作→审校）、**协调者/路由**模式、Agent 间通信 |
| **08** | **优化与性能调优** | **延迟**（流式 TTFT、并发）、**成本**（精确缓存、语义缓存、模型路由、上下文裁剪、提示缓存）、**稳定性**（重试+指数退避+jitter） |
| **09** | 可观测性 & 评估 | 调用埋点追踪、**token 成本核算**、**评估/回归测试框架** |

> 建议按 01→09 顺序讲解，每一课都在前一课基础上"长出"新能力：01-02 打基础，03-04 从聊天机器人进化为 Agent，05-07 补齐记忆/知识/协作，08-09 完成从 demo 到生产的关键一跃。

---

## 项目结构

```
aiagent/
├── llmkit/                    # 可切换后端的极简 Agent 框架（教学内核）
│   ├── types.py               #   provider 无关的数据类型（Message/ToolCall/Usage…）
│   ├── base.py                #   LLMProvider 抽象基类（整个框架的"接缝"）
│   ├── factory.py             #   get_llm()：一行切换后端
│   ├── providers/
│   │   ├── mock.py            #   离线 Mock（可编排行为，驱动所有课程）
│   │   ├── anthropic_provider.py  #   Claude 适配器
│   │   └── openai_provider.py     #   OpenAI/兼容端点适配器
│   ├── tools.py               #   @tool 装饰器 + ToolRegistry
│   ├── agent.py               #   可复用的 ReAct Agent 循环
│   ├── rag.py                 #   本地 embedding + 内存向量库
│   ├── cache.py               #   精确缓存（代理模式）
│   └── retry.py               #   指数退避重试
├── lessons/                   # 9 节课，每节都可独立运行
│   └── lesson_01..09_*.py
├── run_all.py                 # 一键跑全部课程
├── requirements.txt
└── .env.example
```

---

## 设计理念（讲课时可强调）

1. **面向接口，而非面向某个模型**：上层 Agent/RAG/多智能体代码只依赖 `LLMProvider` 抽象，换模型 = 换一个环境变量。这就是"可切换 LLM"的工程本质（依赖倒置）。
2. **离线可测的替身（test double）**：`MockProvider` 让你在不烧 token、不联网的情况下写单测、做演示、复现问题——生产级 Agent 工程的必备实践。
3. **横切关注点用代理/装饰器实现**：缓存（`CachingProvider`）、追踪（`Tracer`）都是"包一层"，不污染业务逻辑。
4. **护栏优先**：`max_steps`、工具异常回传、重试退避——demo 与生产的差距，往往就在这些"不出错时看不见"的地方。

---

## 接真实模型时的注意点

- Mock 的回答是模板化的（它不真的"理解"上下文），因此**结构化输出（02）**和 **RAG 生成（06）**的最终答案，需切到真实模型才能看到理想效果；但这两课的**机制**（解析重试、检索排序）在 Mock 下已完整可见。
- 用 Claude 时，`llmkit` 默认走 `claude-opus-4-8` + 自适应思考；用 OpenAI 兼容端点时，设置 `OPENAI_BASE_URL` 与 `OPENAI_MODEL` 即可对接国产/本地模型。
