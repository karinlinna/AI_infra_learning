# 第 4 节课 · 学生课件：算法基础回补 + LLM 面试八股（20 题标准答案）

> **这不是从零学，是唤醒 + 默写。** 你是 ML 硕士，底子在，校招八股要的是**白板能推、脱口而出**。
> **配套**：`gpt2learn/`（Transformer 原理）、`llm_training_guide/`（训练/数学/ML）、`mini_infra/`（Infra 8 维度，纯 CPU 可跑）。
> **用法**：每道题先自己答，再对答案；标 ⭐ 的是**必须能白板手推**的。

---

## 一、Transformer / Attention（面试第一考点）

### ⭐ Q1. 手写 Attention 公式并解释每一步

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```
- **Q/K/V 来源**：同一 token embedding 分别乘 `W_Q, W_K, W_V`。比喻：Q=我想找什么，K=我有什么标签，V=我实际内容。
- `QKᵀ`：相关度矩阵，每个 token 和所有 token 点积。
- `/√d_k`：缩放，防点积过大。
- `softmax`：每行归一化成概率分布（和=1）。
- `·V`：按权重加权求和。
- **复杂度**：O(n²·d)。

> 练法：用 `transformer_letter.md` 的"我爱你"手算例（dk=2）在白板走一遍到 Output=[3.406, 4.406]。

### ⭐ Q2. 为什么除以 √d_k？不除会怎样？
d_k 大时点积数值大 → softmax 进入饱和区（一个值接近 1 其余接近 0）→ **梯度消失**，训练不动。除以 √d_k 把方差拉回稳定范围。

### Q3. Q/K 为什么要分开，不能共用一个矩阵？
Q·K 算的是"匹配度/相关性"，匹配后真正要读取的是 V（内容）。Q 和 K 角色不同（查询 vs 被查询的索引），共用会限制表达能力。

### Q4. 多头注意力为什么比单头好？
一组 QKV 只能捕捉一种关系模式；多头把 `d_model` 拆成 H 份并行，每个头关注不同模式（语法/语义/位置），再 `Concat·W_O` 融合。类似 CNN 的多个卷积核。

### Q5. 因果掩码 vs Padding 掩码
- **因果掩码**：防作弊，不许看未来 token，未来位置设 -∞，softmax 后权重=0。GPT(Decoder) 用。
- **Padding 掩码**：PAD 无语义不参与注意力，也设 -∞。
- BERT(Encoder) 双向无因果掩码；GPT 单向有。

### Q6. 位置编码为什么需要？RoPE 好在哪？
注意力本身无序（"我爱你" vs "你爱我"没位置就一样），RNN 天生有序不需要。
- 正弦编码：`PE(pos,2i)=sin(pos/10000^(2i/d))`，不同频率唯一编码位置。
- GPT-2 用**可学习位置嵌入**。
- **RoPE（旋转位置编码）**：把位置编码成旋转矩阵作用于 Q、K，关键性质 `qₘᵀkₙ` 只依赖相对位置 **m-n**，且支持**长度外推**。Llama/Qwen/Mistral 采用。
- **ALiBi**：不加位置编码，在 attention score 上加与距离成正比的负偏置，外推性好。

### Q7. Pre-Norm vs Post-Norm？RMSNorm 为什么省？
- **Post-Norm**（原论文）：`LN(x + Sublayer(x))`。
- **Pre-Norm**（GPT-2 起主流）：`x + Sublayer(LN(x))`，梯度直接走残差路径，训练更稳。
- **RMSNorm**：去掉均值中心化，只做缩放 `x/√(mean(x²))·γ`，更省算力，Llama 用。
- **LN vs BN**：LN 对每个样本的特征维归一化（LLM 用，不受 batch 影响）；BN 对 batch 维（CV 用）。

### ⭐ Q8. 残差连接为什么能解决梯度消失？
`output = x + f(x)`，求导时有 **+1** 那条直通路径，梯度不经过权重矩阵直接回传到浅层，避免连乘导致的梯度消失。退化时至少是恒等映射。（ResNet，何恺明 2015）

### Q9. FlashAttention 为什么快？改变计算结果吗？
用 **tiling + online softmax**，把注意力分块计算，避免把 n×n 的注意力矩阵写入慢速 HBM，IO 从 O(n²) 降到 O(n²d/M)，加速 2-4x。**不改变数学结果（精确等价），只优化 IO。** ← 这题答"改变了"就错了。

### Q10. MQA/GQA 解决什么问题？
减小 **KV Cache**（推理时的显存瓶颈）。
| 方案 | KV 头数 | KV Cache |
|---|---|---|
| MHA | h | 全量 |
| MQA | 1（所有 Q 共享一组 KV）| 1/h |
| GQA | g 组（1<g<h）| 1/g |
GQA 是 MHA 和 MQA 的折中，Llama2-70B 用 h=64, g=8。

> **现代主流标配（背下来）**：`GQA + RoPE + RMSNorm(Pre-Norm) + SwiGLU + FlashAttention2 + bf16`

---

## 二、微调与训练

### ⭐ Q11. LoRA 原理？为什么能减少参数？数学基础是什么？
**数学基础 = SVD / 低秩近似：**
1. 全量微调得到更新 `ΔW`（和 W 一样大）。
2. 对 ΔW 做 SVD，发现**奇异值只有前几个大** → ΔW 是**低秩**的。
3. 既然低秩，直接学两个瘦长矩阵 `ΔW ≈ B·A`（B: d×r, A: r×d，r 很小）。
4. **LoRA**：冻结原 W，只训 A、B：`W' = W + (α/r)·B·A`。

```
r=16, α=32(≈2r)；target_modules = q/k/v/o_proj + gate/up/down_proj
Qwen 实际只训 ~0.68% 参数
```
- **为什么 work**：微调只是让模型"某方向偏一偏"，偏移信息维度低，两个瘦长矩阵够用。
- 金句："**SVD 是事后验尸，LoRA 是直接利用规律构造。**"
- **lora_alpha**：缩放系数 α/r，控制 LoRA 更新幅度。

### Q12. QLoRA 和 LoRA 区别？
QLoRA = **4bit NF4 量化加载基座**（7B: 16bit ~14GB → 4bit ~3.5GB）+ **双重量化**（再省~10%）+ 在其上加 LoRA。让 24GB 消费级显卡能微调 14B 模型。
- **部署**：`merge_and_unload()` 把 `W_new = W + BA` 合并，推理速度不受影响；或 base + adapter 分开加载（可热切换）。

### Q13. 预训练 vs SFT vs RLHF？
```
预训练              SFT                RLHF/对齐
Next Token       instruction→output   RM + PPO/DPO
海量无标注文本     结构化问答对          人类偏好数据
学"世界知识"       学"听懂指令"         学"符合偏好/无害"
```
- **预训练**：Next Token Prediction，loss = 交叉熵 = 负对数似然。"语言是世界知识的压缩表示"。
- **SFT**：只预训练的模型会"续写"，SFT 后"听懂指令"直接回答。用 ChatML 模板。
- **RLHF**：SFT → 人类偏好训 Reward Model → PPO 对齐，目标 = Helpful + Harmless（有张力）。
- **Constitutional AI（Anthropic）**：AI 按宪法原则自评自改 → RLAIF，标注可无限扩展。

### ⭐ Q14. 训练显存花在哪？怎么估算？
```
训练显存 ≈ 参数 + 梯度 + Adam状态(m,v 两份) + 激活值
        ≈ 参数量 × 12~16 字节
推理显存 ≈ 参数量 × 2 字节(bf16) + KV Cache
```
- 0.5B 模型：参数(bf16)1GB + 优化器4GB + 梯度1GB ≈ 6GB + 激活值。

### Q15. 梯度累积 / 梯度检查点 / ZeRO 各是什么？
- **梯度累积**：batch=2 累积 8 步 = 等效 batch 16，显存只占 2 条（用时间换显存）。
- **梯度检查点**：只存部分层激活，反向时重算，显存 O(L)→O(√L)，代价 +30% 计算（用算力换显存）。
- **混合精度**：bf16（指数位同 fp32，范围大不易溢出）优于 fp16。
- **DeepSpeed ZeRO**：Stage1 分片优化器状态、Stage2 +梯度、Stage3 +参数，省显存到 ~1/N。

---

## 三、推理优化与 Infra

### ⭐ Q16. KV Cache 原理？为什么只缓存 K/V 不缓存 Q？
生成第 N 个 token 需要前面所有 token 的 K/V，缓存后每步只算**新 token** 的 K/V，计算从 O(n²)→每步 O(1)。代价：显存随序列线性增长。
- **为什么只缓存 K/V？** → **Q 只对当前 token 有用；K/V 要被所有后续 token 复用。**
- **Prefill（算 prompt，compute-bound）vs Decode（每步 1 token，memory-bound）** 是推理优化的核心分野。

### Q17. PagedAttention（vLLM）解决什么？
KV Cache 的**内存碎片**问题。把 KV Cache 按页管理（类比 OS 虚拟内存分页），内存利用率 50%→95%，吞吐比 HF 高 14-24 倍。
- **Continuous Batching**：短请求完成即释放 slot，不等最长请求，GPU 利用率最大化。
- **投机采样**：小模型猜 k 个 token，大模型一次并行验证，2-3x 加速且**输出分布与只用大模型完全一致**。

### ⭐ Q18. 6ND 公式怎么来？MFU 是什么？
```
训练 FLOPs ≈ 6 × N(参数) × D(token)
```
- **6 怎么来**：前向每参数每 token 2 FLOPs（一乘一加），反向 4，共 6。
- **MFU（Model FLOPs Utilization）**：实际算力利用率，通常只有 **30-60%**（通信、访存、流水线气泡吃掉）。
- Chinchilla 最优：训练 token ≈ **20× 参数量**。

### Q19. DP/TP/PP 各切什么？通信原语？
| 并行 | 切什么 | 通信原语 | 特点 |
|---|---|---|---|
| **DP** 数据并行 | 数据分片，每卡全量模型 | **AllReduce** 梯度 | 梯度平均 = 全量梯度 |
| **TP** 张量并行 | 单个算子的权重矩阵 | AllReduce/AllGather 激活 | 限单机（NVLink 高带宽）|
| **PP** 流水线并行 | 按层切到不同卡 | P2P send/recv | 有流水线气泡 |
- **All-Reduce = Reduce-Scatter + All-Gather**（可白板推）。
- 3D 并行：TP 机内 + PP 机间 + DP 最外层。

---

## 四、ML / 数学基础（校招高频）

### ⭐ Q20. 交叉熵为什么不用 MSE？和 MLE 什么关系？
**三重身份（记住这个串联，面试加分）：**
```
分类损失 = 负对数似然(MLE) = LLM 预训练目标   —— 三者是同一个东西
```
- **为什么不用 MSE**：Sigmoid+MSE **非凸**易梯度消失（错得越离谱梯度越趋 0）；Sigmoid+BCE 梯度化简成干净的 `(ŷ-y)x`，凸优化。
- **和 MLE 关系**：交叉熵 = 负对数似然。最小化交叉熵 = 最大化正确标签的预测概率。
- LLM 损失 `L = -Σ log P(xₜ|x<ₜ)` 就是交叉熵。

### 其他快问快答
| 题 | 答 |
|---|---|
| **梯度下降为什么减梯度？** | 梯度指向增大最快方向，最小化要反方向 |
| **Adam 存什么？** | m(一阶动量=方向) + v(二阶动量=步长)，所以显存开销大 |
| **L1 vs L2 正则？** | L1 产生稀疏解（梯度常数能推到 0，特征选择）；L2 参数趋 0 但不为 0 |
| **过拟合怎么判断/解决？** | 训练低测试高（差距大）；增数据/正则/Dropout/早停 |
| **Dropout 为什么推理关闭？** | 训练时随机丢神经元防过拟合，推理要用完整网络 |
| **PCA vs SVD？** | SVD 任意矩阵可分解且一定存在；PCA 是协方差矩阵的特征值分解 |
| **Softmax 为什么用指数？** | 保正、放大差距、来自最大熵原理 |
| **PPL 困惑度？** | `exp(平均loss)`，越低越好；GPT-2 约 18，好 LLM <10 |
| **LSTM 怎么解决 RNN 梯度消失？** | 三门（遗忘/输入/输出）+ 细胞态提供梯度直通路径 |
| **随机初始化模型 loss 多少？** | ≈ ln(vocab_size)，词表 1000 时 ≈ 6.9 |

### 梯度消失的三处呼应（问到就多角度答，显体系）
1. RNN 梯度消失 → LSTM 三门解决
2. 深层网络梯度消失 → 残差连接（求导 +1）
3. Sigmoid+MSE 梯度消失 → 换交叉熵

---

## 五、一条主线串起所有数学（`math_note.md` 结尾图）

```
Tokenizer → Embedding(矩阵) → Attention(矩阵) → FFN(矩阵)
    → Softmax(概率) → 交叉熵(MLE) → 反向传播(链式法则) → Adam(梯度下降)
```
> 能把这条线讲通，说明你不是背知识点，是**理解了整个体系**——这是算法面试最想看到的。

---

## 六、课后作业（第 5 节模拟面试要用）

- [ ] **白板默写**：Attention 公式全流程 + LoRA 低秩图，录视频自查能否不看稿讲清。
- [ ] 跑 `mini_infra/run_all.py`，看懂 KV Cache 推理 + 三种并行的"数值等价验证"。
- [ ] 用 `llm_training_guide.py --stage sft` 跑一次 SFT+LoRA（Qwen2-0.5B+QLoRA，消费级显卡可跑），**补上"端到端跑通一次微调"的实操**——这是简历硬通货。
- [ ] 20 道八股每题写出 3-5 句标准答案。

---

## 七、记住

> 算法基础对你不是从零学，是**唤醒 + 默写**。这 20 题练到白板脱口而出，笔面八股就稳了。
