# 第 4 节课 · 教师教案：算法基础回补 + LLM 训练/推理面试八股（120 分钟）

> **学员**：王利田（ML 硕士，算法基础有遗忘，校招笔面必考八股）
> **配套代码/笔记**：
> - `llm_learning/gpt2learn/`（`gpt2note.md` Transformer 全流程、`transformer_letter.md` 架构优化全景 + Attention 手算例、`gpt2_test.py` 极小 GPT2）
> - `llm_learning/llm_training_guide/`（`note.md` 训练全流程、`ml_fundamentals.md` 经典 ML、`math_note.md` 数学基础、`llm_training_guide.py` 8 步可运行实战）
> - `llm_learning/mini_infra/`（纯 CPU 复现 Infra 8 维度：GEMM/NanoGPT/并行/通信/存储/容错/推理/成本）
> **本节目标**：① 学员能**白板手推 Attention 和 LoRA**；② 备好 **20 道高频八股**的标准答案；③ 建立"数学→训练→推理→Infra"的知识串联。
> **课堂形式**：白板推导 + 手算例 + 跑代码验证 + 面试问答演练。**能推的必须让学员上白板推。**

> **开课前（老师）**：`python llm_learning/mini_infra/run_all.py`、`python llm_learning/gpt2learn/gpt2_test.py` 确认能跑，随堂演示"数值等价"。

---

## 时间轴总览

| 时段 | 模块 | 时长 | 白板产出 |
|---|---|---|---|
| 00:00–00:05 | 开场：为什么八股绕不过 | 5 min | — |
| 00:05–00:35 | Attention 手推（面试第一考点） | 30 min | 一张 Attention 公式图 |
| 00:35–00:52 | 现代架构组件（GQA/RoPE/RMSNorm/SwiGLU/Flash） | 17 min | 主流标配表 |
| 00:52–01:20 | LoRA/QLoRA 手推 + 训练流程 | 28 min | LoRA 低秩图 + 显存账 |
| 01:20–01:42 | 推理优化 + Infra（KV Cache/vLLM/FLOPs/并行） | 22 min | 6ND + 并行表 |
| 01:42–01:55 | ML/数学八股速查 | 13 min | 交叉熵三重身份 |
| 01:55–02:00 | 20 道八股清单 + 作业 | 5 min | — |

---

## 00:00–00:05 ｜ 开场（5 min）

> "你是 ML 硕士，底子在，但校招笔面的八股是**默写题**——不是理解就行，要能**白板手推、脱口而出**。今天我们把 Transformer、LoRA、训练、推理、Infra 的高频考点过一遍，每个能推的我都让你上白板。仓库里的代码能跑，我们边推边验证。目标是下课时你手里有 20 道八股的标准答案。"

---

## 00:05–00:35 ｜ Attention 手推：面试第一考点（30 min）

> **教学核心**：`transformer_letter.md` 有一个完整的"我爱你"手算例子（dk=2 一路算到 Output₁=[3.406, 4.406]），**务必带学员在白板上完整走一遍**。会推 = 真懂，只会背公式 = 一问就穿。

### 第一步：QKV 从哪来 + 图书馆比喻（6 min）
- 同一个 token embedding 分别乘 `W_Q, W_K, W_V` 得到 Q/K/V。
- 比喻：**Q = 我想找什么（查询）**，**K = 我有什么标签（索引）**，**V = 我实际内容**。
- 板书：`Q = XW_Q, K = XW_K, V = XW_V`

**追问准备**："Q、K 能不能共用一个矩阵？" → 不能。Q·K 算的是"匹配度"，匹配后真正要读的是 V（内容），三者角色不同。

### 第二步：公式逐项拆解 + 手推（15 min）

**白板写出核心公式，逐项让学员解释：**
```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```
| 项 | 作用 | 面试追问 |
|---|---|---|
| `QKᵀ` | 相关度矩阵（每个 token 和所有 token 点积）| 复杂度？O(n²·d) |
| `/√d_k` | 防点积过大 → softmax 饱和 → 梯度消失 | **"不除会怎样？"** → 梯度消失，训练不动 |
| `softmax` | 归一化成概率分布（每行和=1）| 为什么用指数？（保正、放大差距、最大熵）|
| `· V` | 按权重加权求和 | — |

**带学员手推"我爱你"例子**（用 `transformer_letter.md` 的数值）：算 `Q·Kᵀ` → 除 √2 → softmax → 加权 V → 得 Output₁。**让学员自己算一遍第二个 token 的输出。**

### 第三步：多头注意力（6 min）
- 一组 QKV 只能捕捉一种关系；多头并行跑多组，分别关注语法/语义/位置。
- 实现：`d_model` 拆成 H 个 `d_k = d_model/H`；输出 `Concat(head₁..head_H)·W_O`。
- **源码细节**（gpt2note.md）：GPT-2 用一个 `Conv1D(3*n_embd)` 一次算出 QKV 再 split，比三个独立矩阵高效；`view+transpose` 变形是零拷贝（只改 shape/stride）。

### 第四步：因果掩码（3 min）
- **因果掩码**：不许看未来 token，未来位置设 -∞，softmax 后权重=0（防作弊）。
- **Padding 掩码**：PAD 无语义不参与，也设 -∞。
- 跑 `gpt2_test.py` 验证：随机初始化模型 loss ≈ `ln(vocab_size)` = ln(1000) ≈ 6.9（**这个数字串起交叉熵+随机猜+词表大小三个概念，是"你真懂了吗"的验证题**）。

> **本模块白板产出**：一张完整的 Attention 数据流图 + 学员能独立手推一遍。

---

## 00:35–00:52 ｜ 现代架构组件（17 min）

> **教学核心**：`transformer_letter.md` 有完整的"原始 Transformer → 现代主流"演进。学员要能答出"现在大模型的标准架构组件"，这是加分项。

### 逐个讲（每个配一句"解决什么问题"）（12 min）

| 组件 | 替代了什么 | 解决的问题 | 代表 |
|---|---|---|---|
| **RoPE** 旋转位置编码 | 正弦/可学习位置编码 | `qₘᵀkₙ` 只依赖相对位置 m-n，支持长度外推 | Llama/Qwen |
| **GQA** 分组查询注意力 | MHA | KV 头数从 h 降到 g 组，**减小 KV Cache**（推理显存瓶颈）| Llama2-70B(h=64,g=8) |
| **RMSNorm** | LayerNorm | 去掉均值中心化，只缩放，更省算力 | Llama |
| **Pre-Norm** | Post-Norm | 梯度直接走残差路径，训练更稳 | GPT-2 起 |
| **SwiGLU** | GELU FFN | 门控机制，效果更好 | Llama/PaLM |
| **FlashAttention** | 朴素注意力 | tiling+online softmax，不写 n×n 到 HBM，**加速 2-4x 且精确等价** | 全部 |

**板书 · 现代主流标配（背下来）：**
```
GQA + RoPE + RMSNorm(Pre-Norm) + SwiGLU + FlashAttention2 + bf16
（Llama3 / Mistral / Qwen2 / Gemma2 都是这套）
```

### MHA → MQA → GQA 演进（5 min，重点考点）
| 方案 | KV 头数 | KV Cache | 代表 |
|---|---|---|---|
| MHA | h | 全量 | 原始 Transformer |
| MQA | 1（所有 Q 共享一组 KV）| 1/h | Google 2019 |
| GQA | g 组（1<g<h）| 1/g | Llama2-70B |

**面试追问**："FlashAttention 改变计算结果了吗？" → **没有，只优化 IO（数学精确等价）。** 这题答错 = 没真懂。

---

## 00:52–01:20 ｜ LoRA/QLoRA 手推 + 训练流程（28 min）

### 第一步：LoRA 的数学基础——SVD → 低秩（10 min）

> **教学核心**：`math_note.md` 有 **SVD → 低秩近似 → LoRA** 完整故事线，这是让学员"真懂 LoRA 而非背概念"的关键。白板推。

**白板讲故事：**
1. 全量微调得到权重更新 `ΔW`（和 W 一样大）。
2. 对 ΔW 做 **SVD** 分解 `ΔW = UΣVᵀ`，发现**奇异值只有前几个大**——说明 ΔW 是**低秩**的。
3. 既然低秩，就别学整个 ΔW，直接学两个瘦长矩阵 `ΔW ≈ B·A`（B: d×r, A: r×d，r 很小）。
4. **LoRA**：冻结原 W，只训 A、B：`W' = W + B·A`。

**板书公式 + 参数量对比：**
```
W' = W + (α/r)·B·A        r=16, α=32(≈2r)
4×4 矩阵：全量 16 参数 → r=1 的 A(4×1)+B(1×4) = 8 参数
Qwen 实际：只训 ~0.68% 参数
```
- **为什么 work？** 微调只是让模型"某方向偏一偏"，偏移信息维度低，两个瘦长矩阵就够。
- 一句话金句（`math_note.md`）："**SVD 是事后验尸，LoRA 是直接利用规律构造。**"

**追问准备**：
- "lora_alpha 作用？" → 缩放系数 `α/r`，控制 LoRA 更新的幅度。
- "target_modules 选哪些？" → attention(q/k/v/o_proj) + FFN(gate/up/down_proj)。

### 第二步：QLoRA + 部署（5 min）
- **QLoRA** = 4bit NF4 量化加载基座（7B: 16bit ~14GB → 4bit ~3.5GB）+ 双重量化（再省~10%）+ 在其上加 LoRA。
- **部署**：`merge_and_unload()` 合并 `W_new = W + BA`，**推理速度不受影响**（连回第 3 节）。

### 第三步：训练流程全景（8 min）

**板书 · LLM 训练三阶段（对照 `note.md` + `claude_analysis.md`）：**
```
预训练(Pretrain)          SFT(指令微调)           RLHF/对齐
Next Token Prediction  →  instruction→output  →  RM + PPO/DPO
海量无标注文本            结构化问答对            人类偏好数据
学"世界知识"             学"听懂指令"            学"符合偏好/无害"
```
- **预训练**：Next Token Prediction，loss = 交叉熵 = 负对数似然。"语言是世界知识的压缩表示"。
- **SFT**：只预训练的模型会"续写"（"请翻译：你好"→继续续写），SFT 后"听懂指令"直接答"Hello"。用 ChatML 模板。
- **RLHF**：SFT → 人类偏好训 Reward Model → PPO 对齐。目标 = Helpful + Harmless（有张力）。
- **Constitutional AI（Anthropic）**：用 AI 按宪法原则自评自改 → RLAIF，可无限扩展标注。

### 第四步：训练工程账（5 min）

**板书 · 训练显存构成（面试必考）：**
```
训练显存 ≈ 参数 + 梯度 + Adam状态(m,v 两份) + 激活值
        ≈ 参数量 × 12~16 字节（bf16 参数+梯度，fp32 优化器）
```
- **梯度累积**：batch=2 累积 8 步 = 等效 batch 16，显存只占 2 条。
- **混合精度**：bf16（指数位同 fp32，范围大不易溢出）> fp16。
- **梯度检查点**：只存部分层激活，反向重算，显存 O(L)→O(√L)，代价 +30% 计算。
- **DeepSpeed ZeRO**：Stage1 分片优化器状态、Stage2 +梯度、Stage3 +参数。

---

## 01:20–01:42 ｜ 推理优化 + Infra（22 min）

> **教学核心**：`mini_infra/` 用纯 CPU 复现了 Infra 8 维度，且每个都有"数值等价验证"。学员投 AI 应用/算法岗，推理和显存/成本是高频考点。

### 第一步：KV Cache（6 min）
- 生成第 N 个 token 需要前面所有 K/V，缓存后每步只算新 token 的 K/V，计算 O(n²)→每步 O(1)，代价是显存随序列线性增长。
- **为什么只缓存 K/V 不缓存 Q？** → Q 只对当前 token 有用；K/V 要被所有后续 token 复用。
- 跑 `mini_infra/7_inference/inference_engine.py` 看**朴素 vs KV Cache vs Continuous Batching**三种对比。

**Prefill vs Decode（推理优化的核心分野）：**
- Prefill：一次处理整个 prompt，**计算密集（compute-bound）**。
- Decode：每步 1 token，**访存密集（memory-bound）**。

### 第二步：vLLM 与推理引擎（6 min）
- **PagedAttention（vLLM）**：KV Cache 按页管理（类比 OS 虚拟内存分页），解决内存碎片，利用率 50%→95%，吞吐比 HF 高 14-24 倍。**推理岗必考。**
- **Continuous Batching**：不像 static batching 要等最长请求结束，短请求完成即释放 slot，GPU 利用率最大化。
- **投机采样**：小模型猜 k 个 token，大模型一次并行验证，2-3x 加速且**输出分布与只用大模型完全一致**。
- 量级参考：Llama-2-7B 单 A100，HF ~30 tok/s、vLLM ~600、TRT-LLM ~800。

### 第三步：FLOPs 与成本（5 min）
**板书 · 两个必背公式：**
```
训练 FLOPs ≈ 6 × N(参数) × D(token)     ← 前向2 + 反向4
推理显存 ≈ N × 2 字节(bf16) + KV Cache
```
- **6ND 里的 6 怎么来？** 前向每参数每 token 2 FLOPs（一乘一加），反向 4，共 6。
- **MFU（Model FLOPs Utilization）**：实际算力利用率，通常只有 30-60%（通信、访存、气泡吃掉）。
- Chinchilla 最优：训练 token ≈ 20× 参数量。

### 第四步：三种并行（5 min）
**板书对照（`mini_infra/3_parallel`）：**
| 并行 | 切什么 | 通信原语 | 特点 |
|---|---|---|---|
| **DP** 数据并行 | 数据分片，每卡全量模型 | **AllReduce** 梯度 | 梯度平均 = 全量梯度 |
| **TP** 张量并行 | 单个算子的权重矩阵 | AllReduce/AllGather 激活 | 限单机（NVLink 高带宽）|
| **PP** 流水线并行 | 按层切到不同卡 | P2P send/recv | 有流水线气泡 |

- **ZeRO** 三阶段分片优化器状态/梯度/参数（省显存到 1/N）。
- **All-Reduce = Reduce-Scatter + All-Gather**（可白板推）。

---

## 01:42–01:55 ｜ ML/数学八股速查（13 min）

> **教学核心**：`ml_fundamentals.md` + `math_note.md` 覆盖极全。这里挑校招高频，重点讲**串联**（面试加分）。

### 交叉熵的三重身份（4 min，最值得讲的串联）
```
分类损失（ml_fundamentals）= 负对数似然（math_note MLE）= LLM 预训练目标（note.md）
                    三者是同一个东西
```
- 交叉熵来自 MLE：伯努利分布 → 似然连乘 → 取对数 → 加负号。
- **分类为什么用交叉熵不用 MSE？** → Sigmoid+MSE 非凸易梯度消失（错得越离谱梯度越趋0）；Sigmoid+BCE 梯度化简成干净的 `(ŷ-y)x`。

### 梯度消失的三处呼应（3 min）
- RNN 梯度消失（→ LSTM 三门解决）
- 深层网络梯度消失（→ 残差连接：求导 +1 直通高速公路）
- Sigmoid+MSE 梯度消失（→ 换交叉熵）
> 面试问"梯度消失"可从多角度答，显示体系化理解。

### 其他高频速点（6 min，快问快答）
- **残差为什么解决梯度消失？** → `output = x + f(x)`，求导有 +1 那条直通路径。
- **Adam 存什么？** → m（一阶动量，方向）+ v（二阶动量，步长），所以显存开销大。
- **L1 vs L2 正则？** → L1 产生稀疏解（梯度常数能推到 0），L2 参数趋 0 但不为 0。
- **BN vs LN？** → LN 对每个样本的特征维归一化（LLM 用），BN 对 batch 维（CV 用）。
- **PCA / SVD 区别？** → SVD 任意矩阵可分解且一定存在；PCA 是协方差矩阵的特征值分解。
- **PPL 困惑度** = `exp(平均loss)`，越低越好，GPT-2 约 18，好 LLM <10。

---

## 01:55–02:00 ｜ 20 道八股清单 + 作业（5 min）

**发给学员的 20 道高频八股（标准答案见学生课件）：**
1. 手写 Attention 公式并解释每一步
2. 为什么除以 √d_k？不除会怎样？
3. Q/K 为什么要分开？
4. 多头注意力为什么比单头好？
5. 因果掩码 vs Padding 掩码
6. 位置编码为什么需要？RoPE 好在哪？
7. Pre-Norm vs Post-Norm？RMSNorm 为什么省？
8. 残差为什么解决梯度消失？
9. FlashAttention 为什么快？改变结果吗？
10. MQA/GQA 解决什么问题？
11. LoRA 原理？为什么能减少参数？数学基础？
12. QLoRA 和 LoRA 区别？
13. 预训练 vs SFT vs RLHF？
14. 训练显存花在哪？怎么估算？
15. 梯度累积 / 梯度检查点 / ZeRO 各是什么？
16. KV Cache 原理？为什么只缓存 K/V？
17. PagedAttention 解决什么？
18. 6ND 公式怎么来？MFU 是什么？
19. DP/TP/PP 各切什么？通信原语？
20. 交叉熵为什么不用 MSE？和 MLE 什么关系？

**课后作业（第 5 节模拟面试要用）：**
- [ ] **白板默写**：Attention 公式全流程 + LoRA 低秩图，录视频自查能否不看稿讲清。
- [ ] 跑 `mini_infra/run_all.py`，挑 KV Cache 推理、三种并行两个模块，看懂"数值等价验证"。
- [ ] 用 `llm_training_guide.py` 跑一次 SFT+LoRA（`--stage sft`，Qwen2-0.5B + QLoRA，消费级显卡可跑），**补上"端到端跑通一次微调"的实操**。
- [ ] 把 20 道八股每题写出 3-5 句标准答案，下节课抽查。

> **结束语**："算法基础是校招绕不过的关，但对你不是从零学，是**唤醒 + 默写**。这 20 题练熟，笔面八股就稳了。下节课我们把简历和项目也打磨好，做最后的模拟面试。"

---

## 附：教师随堂可演示的"数值等价"锚点

- `gpt2_test.py`：随机初始化 loss ≈ ln(vocab) —— 验证交叉熵理解。
- `mini_infra/3_parallel/data_parallel.py`：梯度平均 vs 全量梯度差异 < 1e-10 —— "分布式不是魔法，是可验证的数学等价切分"。
- `mini_infra/1_compute_core/tensor_ops_demo.py`：FP16 溢出 `60000×2 = inf` —— 引出混合精度/Loss Scaling。
- `mini_infra/7_inference/inference_engine.py`：朴素 vs KV Cache 的 token/s 对比。
