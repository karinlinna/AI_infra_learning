# Transformer 架构优化全景

## 1. 注意力机制优化

### 1.1 Multi-Head Attention (MHA) — 原始 Transformer

原始 Transformer 中，Q、K、V 各有独立的 $h$ 组投影矩阵：

$$
\text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)
$$

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O
$$

其中每个头的维度为 $d_k = d_{\text{model}} / h$。

**KV Cache 开销**：推理时需要缓存所有层、所有头的 K 和 V，显存占用为：

$$
\text{KV Cache} = 2 \times L \times h \times n \times d_k
$$

其中 $L$ 为层数，$h$ 为头数，$n$ 为序列长度。

---

### 1.2 Multi-Query Attention (MQA) — Google, 2019

**核心思想**：所有 Q 头共享同一组 K 和 V。

$$
Q_i = XW_i^Q, \quad K = XW^K, \quad V = XW^V
$$

- Q 仍有 $h$ 组投影，K 和 V 只有 **1 组**
- KV Cache 降为原来的 $\frac{1}{h}$
- 速度提升显著，但质量略有下降

---

### 1.3 Grouped-Query Attention (GQA) — Llama 2, 2023

**核心思想**：MHA 和 MQA 的折中，将 $h$ 个 Q 头分成 $g$ 组，每组共享一套 KV。

$$
\text{group}_j = \text{Attention}\!\Big(\text{Concat}(Q_{j_1}, \dots, Q_{j_k}),\; K_j,\; V_j\Big)
$$

| 配置 | KV 头数 | 等价于 |
|------|---------|--------|
| $g = h$ | $h$ | MHA |
| $g = 1$ | $1$ | MQA |
| $1 < g < h$ | $g$ | GQA |

**Llama 2 70B** 使用 $h=64, g=8$，KV Cache 缩小 8 倍，质量接近 MHA。

---

### 1.4 Sparse Attention — Longformer / BigBird, 2020

标准注意力复杂度为 $O(n^2)$，对长序列不可行。稀疏注意力只计算部分位置对：

- **Sliding Window**：每个 token 只关注局部窗口 $w$ 内的 token
- **Global Token**：少量特殊 token（如 `[CLS]`）可以 attend 到所有位置
- **Random**：随机采样部分位置

$$
\text{复杂度}: O(n^2) \rightarrow O(n \cdot w)
$$

---

### 1.5 Flash Attention — Dao, 2022

**不改变数学计算**，只优化 GPU 内存访问模式：

标准实现的瓶颈在于 attention 矩阵 $(n \times n)$ 需要完整写入 HBM（高带宽显存）：

```
标准流程: Q,K → SRAM → S(n×n) → 写回 HBM → 读回 SRAM → softmax → 写回 HBM → ×V
Flash:    Q,K,V 分块 → 全部在 SRAM 完成 → 只把最终结果写回 HBM
```

通过 **tiling（分块）+ online softmax（在线重计算）**，避免 $O(n^2)$ 的中间矩阵写入 HBM：

$$
\text{IO 复杂度}: O(n^2) \rightarrow O\!\left(\frac{n^2 d}{M}\right)
$$

其中 $M$ 为 SRAM 大小。实际加速 **2-4x**，且精确等价于标准 attention。

---

### 1.6 Linear Attention — Katharopoulos, 2020

用核函数近似 softmax，将注意力计算顺序调换：

标准 attention：

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

Linear attention 用特征映射 $\phi$ 替代 softmax：

$$
\text{Attn}(Q, K, V) = \frac{\phi(Q)\;(\phi(K)^T V)}{\phi(Q)\;\phi(K)^T \mathbf{1}}
$$

先算 $\phi(K)^T V$（大小为 $d \times d$），再乘 $\phi(Q)$：

$$
O(n^2 d) \rightarrow O(n d^2)
$$

当 $n \gg d$ 时效果显著，但精度通常不如标准 attention。

---

## 2. 位置编码优化

### 2.1 原始正弦位置编码 — Transformer, 2017

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

固定编码，加到输入 embedding 上。不含可学习参数，但缺乏外推能力。

---

### 2.2 RoPE (Rotary Position Embedding) — Su, 2021

**核心思想**：将位置信息编码为旋转矩阵，直接作用于 Q 和 K。

对于位置 $m$ 的向量，每两个维度作为一对，施加旋转：

$$
\begin{pmatrix} q_{2i}^{(m)} \\ q_{2i+1}^{(m)} \end{pmatrix}
=
\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
$$

其中 $\theta_i = 10000^{-2i/d}$。

**关键性质**：$q_m^T k_n$ 只依赖 $m - n$（相对位置），天然支持相对位置编码。

**被 Llama, Qwen, Mistral 等主流模型采用。**

---

### 2.3 ALiBi (Attention with Linear Biases) — Press, 2022

**完全不加位置编码**，而是在 attention score 上加一个与距离成正比的负偏置：

$$
\text{score}(i, j) = q_i^T k_j - m \cdot |i - j|
$$

其中 $m$ 是每个头不同的斜率，几何序列分配：

$$
m \in \left\{ 2^{-\frac{8}{h}}, 2^{-\frac{16}{h}}, \dots, 2^{-8} \right\}
$$

距离越远 penalty 越大，自然衰减远距离的 attention。**无需训练位置参数**，外推性能好。

---

### 2.4 YaRN / NTK-aware Scaling — 2023

为扩展 RoPE 模型的上下文长度而设计。

**NTK-aware**：不均匀缩放 $\theta_i$，高频维度少缩放，低频维度多缩放：

$$
\theta_i' = \left(\alpha \cdot 10000\right)^{-2i/d}
$$

其中 $\alpha$ 为缩放因子。

**YaRN**：在 NTK 基础上加入注意力分布的温度修正，进一步提升长文本质量。

使得 4K 训练的模型可以推理 **128K+** 的上下文。

---

## 3. 归一化与激活函数

### 3.1 Pre-Norm vs Post-Norm

**Post-Norm**（原始 Transformer）：

$$
x = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

**Pre-Norm**（GPT-2 起）：

$$
x = x + \text{Sublayer}(\text{LayerNorm}(x))
$$

Pre-Norm 让梯度直接通过残差路径流动，**训练更稳定**，是当前主流做法。

---

### 3.2 RMSNorm — Zhang & Sennrich, 2019

LayerNorm 需要计算均值和方差：

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta
$$

RMSNorm 去掉均值中心化，只做缩放：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}} \cdot \gamma
$$

计算更简单，效果相当。**被 Llama 系列采用。**

---

### 3.3 SwiGLU — Shazeer, 2020

标准 FFN：

$$
\text{FFN}(x) = \text{GELU}(xW_1)\, W_2
$$

SwiGLU 加入门控机制：

$$
\text{SwiGLU}(x) = \left(\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}}\right) W_{\text{down}}
$$

其中 $\text{Swish}(x) = x \cdot \sigma(x)$，$\odot$ 为逐元素乘法。

门控让网络学习**选择性**地通过信息，实验表明效果优于 GELU/ReLU。**被 Llama, PaLM 采用。**

---

## 4. 架构级优化

### 4.1 Mixture of Experts (MoE) — Switch Transformer / Mixtral

将 FFN 层替换为多个"专家"网络，通过门控路由选择 Top-K 个专家：

$$
y = \sum_{i=1}^{K} g_i(x) \cdot E_i(x)
$$

其中 $g(x) = \text{TopK}(\text{softmax}(xW_g))$ 为门控函数，$E_i$ 为第 $i$ 个专家。

| 模型 | 总参数 | 激活参数 | 专家数 |
|------|--------|---------|--------|
| Mixtral 8x7B | 47B | 13B | 8 选 2 |
| Switch Transformer | 1.6T | ~同 T5-Base | 2048 |

**大参数量 + 小计算量**，效率极高。

---

### 4.2 DeepNorm — Microsoft, 2022

为训练超深网络（1000 层+），修改残差连接的缩放：

$$
x = \text{LayerNorm}(\alpha \cdot x + \text{Sublayer}(x))
$$

其中 $\alpha = (2N)^{1/4}$（$N$ 为层数），同时初始化时对 sublayer 参数做 $\beta = (8N)^{-1/4}$ 缩放。

确保梯度在极深网络中保持稳定。

---

## 5. 长上下文优化

### 5.1 KV Cache 压缩

推理时 KV Cache 是显存主要瓶颈。压缩方法包括：

- **量化**：将 KV 从 fp16 量化到 int8/int4
- **Eviction**：基于 attention score 丢弃不重要的旧 KV
- **Sliding Window**（Mistral）：只保留最近 $w$ 个 token 的 KV

$$
\text{显存}: O(L \cdot h \cdot n \cdot d_k) \rightarrow O(L \cdot h \cdot w \cdot d_k)
$$

---

### 5.2 Ring Attention — 2023

将长序列切分到多个 GPU，每个 GPU 持有一段 Q，KV 在 GPU 间**环形传递**：

```
GPU 0: Q_0, 计算 Attn(Q_0, KV_0) → 传 KV_0 给 GPU 1, 接收 KV_3
GPU 1: Q_1, 计算 Attn(Q_1, KV_1) → 传 KV_1 给 GPU 2, 接收 KV_0
GPU 2: Q_2, 计算 Attn(Q_2, KV_2) → 传 KV_2 给 GPU 3, 接收 KV_1
GPU 3: Q_3, 计算 Attn(Q_3, KV_3) → 传 KV_3 给 GPU 0, 接收 KV_2
```

每个 GPU 只需要 $O(n/p)$ 的显存（$p$ 为 GPU 数），理论上支持**无限长**上下文。

---

## 6. 训练优化

### 6.1 ZeRO (Zero Redundancy Optimizer) — DeepSpeed

标准数据并行中，每张 GPU 都持有完整的模型参数、梯度、优化器状态。ZeRO 分三个阶段逐步消除冗余：

| 阶段 | 分片内容 | 显存节省 |
|------|---------|---------|
| ZeRO-1 | 优化器状态 | ~4x |
| ZeRO-2 | + 梯度 | ~8x |
| ZeRO-3 | + 模型参数 | ~$N_d$x ($N_d$ 为 GPU 数) |

---

### 6.2 混合精度训练

- **fp32**（32 bit）：精度高，显存大
- **fp16**（16 bit）：显存减半，但容易溢出
- **bf16**（16 bit）：指数位与 fp32 相同，范围大，**训练更稳定**

$$
\text{显存}: \frac{fp32}{2} \approx bf16, \quad \text{速度}: bf16 \approx 2 \times fp32
$$

现代训练标配 **bf16 + fp32 master weights**。

---

### 6.3 Gradient Checkpointing

正常训练需要保存所有中间激活值用于反向传播：

$$
\text{显存} \propto O(L \cdot n \cdot d)
$$

Gradient checkpointing 只保存部分层的激活值，其余在反向传播时**重新计算**：

$$
\text{显存}: O(L) \rightarrow O(\sqrt{L}), \quad \text{计算}: \sim 1.3\times
$$

用约 30% 额外计算换取大幅显存节省。

---

## 7. 当前主流模型标配

| 组件 | 主流选择 |
|------|---------|
| 注意力 | GQA |
| 位置编码 | RoPE |
| 归一化 | RMSNorm (Pre-Norm) |
| FFN 激活 | SwiGLU |
| 注意力加速 | Flash Attention 2 |
| 训练精度 | bf16 |

代表模型：**Llama 3, Mistral, Qwen 2, Gemma 2**

# Attention
## 
一个简单的例子      
                                           
  假设输入序列是 "我 爱 你"，3 个词，每个词的嵌入维度为 4。                     
                                                                                
  第 0 步：输入向量                                                             
                                                                                
  假设经过 Embedding 后：                                                       
   
  "我" → x₁ = [1, 0, 1, 0]                                                      
  "爱" → x₂ = [0, 1, 0, 1]                                                      
  "你" → x₃ = [1, 1, 0, 0]
                                                                                
  第 1 步：生成 Q、K、V
                                                                                
  每个词通过三个权重矩阵 Wq、Wk、Wv 分别映射为 Query、Key、Value。              
   
  为了简化，假设维度 dk = 2，且经过线性变换后得到：                             
                  
  Q₁ = [1, 0]    K₁ = [0, 1]    V₁ = [1, 2]                                     
  Q₂ = [0, 1]    K₂ = [1, 0]    V₂ = [3, 4]
  Q₃ = [1, 1]    K₃ = [1, 1]    V₃ = [5, 6]                                     
                                                                                
  第 2 步：计算点积得分（以"我"为例）                                           
                                                                                
  计算 Q₁ 与所有 K 的点积，看"我"对每个词的关注程度：                           
                  
  Q₁ · K₁ = 1×0 + 0×1 = 0                                                       
  Q₁ · K₂ = 1×1 + 0×0 = 1
  Q₁ · K₃ = 1×1 + 0×1 = 1                                                       
   
  得分：[0, 1, 1]                                                               
                  
  第 3 步：缩放（Scale）                                                        
   
  除以 √dk = √2 ≈ 1.414，防止点积值过大导致 softmax 梯度消失：                  
                  
  [0/1.414,  1/1.414,  1/1.414]                                                 
  = [0,  0.707,  0.707]                                                         
                                                                                
  第 4 步：Softmax 归一化                                                       
                                                                                
  将得分转换为概率分布（注意力权重）：                                          
   
  softmax([0, 0.707, 0.707])                                                    
                  
  e⁰     = 1.000                                                                
  e⁰·⁷⁰⁷ = 2.028
  e⁰·⁷⁰⁷ = 2.028                                                                
  总和    = 5.056 
                                                                                
  权重 = [1/5.056,  2.028/5.056,  2.028/5.056]                                  
       = [0.198,    0.401,        0.401]
                                                                                
  含义："我"这个词对"爱"和"你"的关注度相同（各                                  
  40.1%），对自己的关注度较低（19.8%）。
                                                                                
  第 5 步：加权求和

  用注意力权重对 V 进行加权求和，得到"我"的输出：                               
   
  Output₁ = 0.198 × V₁ + 0.401 × V₂ + 0.401 × V₃                                
          = 0.198 × [1,2] + 0.401 × [3,4] + 0.401 × [5,6]                       
          = [0.198, 0.396] + [1.203, 1.604] + [2.005, 2.406]
          = [3.406, 4.406]                                                      
                  
  这个 [3.406, 4.406] 就是"我"这个位置融合了整个序列信息后的新表示。            
                  
  总结流程                                                                      
                  
  Q·Kᵀ → 点积得分 → ÷√dk 缩放 → Softmax → × V 加权求和 → 输出                   
                                                                                
  对"爱"和"你"也执行同样的操作，就得到了整个序列的 Self-Attention               
  输出。每个词的输出都包含了它对其他所有词的"关注"信息。