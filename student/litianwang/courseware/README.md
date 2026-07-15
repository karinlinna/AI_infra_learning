# 王利田 · AI 应用 / Agent 开发求职提升课 —— 课件总览

> **学员**：王利田（澳国立 ML&CV 硕士在读，2026.12 毕业，27 届应届）
> **主攻方向**：LLM 应用 / AI Agent 开发工程师　**辅攻**：NLP/大模型应用算法、AI 产品/解决方案
> **提升周期**：秋招前 3–4 个月冲刺，共 5 节课
> **本目录**：第 1 节（求职规划）的落地执行，把「有基础」做成「有作品、能上线、能讲清楚」。

---

## 一、这套课件怎么用

每节课都提供**两个版本**，配套仓库里**真实可运行的代码**：

| 版本 | 文件名 | 给谁 | 用途 |
|---|---|---|---|
| **教师教案** | `*_teacher.md` | 老师 | 时间轴、讲授脚本、互动追问、板书、参考答案、易错点 |
| **学生课件** | `*_student.md` | 学员 | 概念精讲、代码走读、图解、动手练习、面试考点、课后作业 |

> **核心理念**：不空讲概念。每一个知识点都对应仓库里一段能跑的代码（`AIagent/`、`llm_learning/mini_infra/`、`tesla.py` 等），**先看现象、再讲原理、最后落到面试怎么答**。

---

## 二、5 节课能力地图

```
第1节  求职规划        ──▶  方向定位 + GAP 分析 + 课程规划          [已完成]
                             │
第2节  Agent 工程实战   ──▶  裸调用→结构化→工具→Agent Loop→记忆
                             →RAG→多智能体→优化→可观测（AIagent 9 课）
                             │  产出：能独立写出带工具+记忆的 Agent
                             ▼
第3节  落地可上线项目   ──▶  RAG(tesla.py) + 微调部署(jx3_bot)
                             + FastAPI/Docker + 后端工程短板补齐
                             │  产出：一个 GitHub 可展示、能跑的项目
                             ▼
第4节  算法基础回补     ──▶  Transformer/Attention 手推 + LoRA/QLoRA
                             + 训练/推理/Infra 八股（gpt2learn + mini_infra）
                             │  产出：白板能讲清 Attention & LoRA，20 道八股
                             ▼
第5节  简历+模拟面试    ──▶  简历 STAR 化 + 项目高光讲述 + 模拟面试
                             + 投递 SOP（qa.md 深挖题库）
                                产出：可直投简历 + 2 分钟项目讲述 + 投递清单
```

---

## 三、每节课的量化目标（学员自检用）

| 课 | 主题 | 上完必须达到 | 对应仓库资产 |
|---|---|---|---|
| 2 | Agent 工程实战 | 不看教程独立写出带工具调用+记忆的 Agent；说清 ReAct Loop 每一步 | `AIagent/lessons/01–09` |
| 3 | 落地项目 | 有一个能本地/云端跑起来、含架构图 README 的 GitHub 项目；补上"后端部署" | `student/litianwang/tesla.py`、`llm_learning/jx3_bot/` |
| 4 | 算法回补 | 白板讲清 Attention & LoRA；准备 20 道高频八股标准答案 | `llm_learning/gpt2learn/`、`llm_learning/mini_infra/`、`llm_training_guide/` |
| 5 | 简历面试 | 一份可直投简历 + 2 分钟项目高光 + 一份投递清单 | `student/litianwang/qa.md`、`introduce.md` |

---

## 四、目录结构

```
courseware/
├── README.md                                  # 本文件（总览 + 学习地图）
├── lesson_02_agent_engineering_teacher.md     # 第2节 教师教案
├── lesson_02_agent_engineering_student.md     # 第2节 学生课件
├── lesson_03_project_deployment_teacher.md    # 第3节 教师教案
├── lesson_03_project_deployment_student.md    # 第3节 学生课件
├── lesson_04_algorithm_fundamentals_teacher.md# 第4节 教师教案
├── lesson_04_algorithm_fundamentals_student.md# 第4节 学生课件
├── lesson_05_resume_interview_teacher.md       # 第5节 教师教案
└── lesson_05_resume_interview_student.md        # 第5节 学生课件
```

> 第 1 节《求职规划》见上级目录 `lesson_01_career_planning.md`（本身就是教师教案格式）。

---

## 五、贯穿全程的三条主线

1. **RAG 是长板 → 磨到能讲透**：Tesla 项目的多路召回 + RRF + Reranker，是简历第一资产（第 3、4、5 节反复打磨）。
2. **Agent + 部署是短板 → 补成作品**：从第 2 节跑通 `AIagent` 全链路，到第 3 节做出一个能上线的 Agent 应用，补齐"工程落地"。
3. **算法基础 + 表达是校招关 → 绕不过**：第 4 节回补 Transformer/微调八股，第 5 节把项目讲成"面试语言"。

---

## 六、给学员的一句话

> 方向已经定了（第 1 节）。接下来这 4 节课，每一节都有**能跑的代码**和**可交付的产出**。
> 不要停留在"看懂"——要做到**能复现、能改、能讲**。面试官不问你看过什么，只问你做过什么、为什么这么做。
