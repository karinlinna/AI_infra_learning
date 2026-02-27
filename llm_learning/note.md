# LLM 学习笔记

## RAG 简介

**什么是 RAG？** 将文本转换成向量，通过余弦相似度检索最相关的内容。

> 可以使用 `gpt2_test.py` 查看 GPT 的工业化实现。

---

## Transformer 基本原理与流程

### 一、词嵌入（Embedding）

模型不认识文字，需要先把 token 转成向量：

```
"我"   → [0.12, -0.34, 0.56, ...]   (维度 d_model，如 512)
"爱"   → [0.78, 0.21, -0.45, ...]
"学习" → [-0.33, 0.67, 0.11, ...]
```

这个映射通过一个可学习的 **Embedding 矩阵**实现，大小为 `vocab_size × d_model`。

#### 第一步：分词（Tokenization）

把文本拆成 token，每个 token 对应一个整数 ID：

```
"我"   → ID 42
"爱"   → ID 156
"学习" → ID 789
```

#### 第二步：查表（Embedding Lookup）

有一个大矩阵，叫 Embedding 矩阵，形状是 `vocab_size × d_model`：

```
           维度0   维度1   维度2  ...  维度511
ID 0     [ 0.01,  0.23, -0.11, ...,  0.45 ]
ID 1     [ 0.33, -0.12,  0.78, ..., -0.09 ]
...
ID 42    [ 0.12, -0.34,  0.56, ...,  0.88 ]  ← "我"
...
ID 156   [ 0.78,  0.21, -0.45, ...,  0.33 ]  ← "爱"
...
ID 789   [-0.33,  0.67,  0.11, ..., -0.22 ]  ← "学习"
```

所谓"转成向量"，就是用 token 的 ID 作为行号，去这个矩阵里取对应那一行。**本质就是查表。**

#### 第三步：得到输入矩阵

```
"我爱学习" → 3 个 token → 3 个向量 → 形状 [3, 512] 的矩阵
```

这个矩阵就是后续 Transformer 层的输入。

#### 关键点

- 这个 Embedding 矩阵是**随机初始化**的，训练过程中不断更新
- 训练后，语义相近的词向量会靠得近（比如"猫"和"狗"的向量相似度高）
- `vocab_size` 决定能表示多少个不同的 token，`d_model` 决定每个向量多长（表达能力）

---

### 二、位置编码（Positional Encoding）

注意力机制本身是**无序的**——"我爱你"和"你爱我"如果不加位置信息，对模型来说完全一样。所以需要给每个位置加上位置信号。

原始论文用的是正弦/余弦函数：

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中：

- `pos` 是 token 在序列中的位置
- `i` 是向量维度的索引
- `d_model` 是嵌入维度

示例：

- `pos=0`（"我"）：用 `i=0,1,2,…,255` 代入公式，sin 和 cos 交替填充，得到一个 512 维向量
- `pos=1`（"爱"）：同样的方式，得到另一个 512 维向量
- `pos=2`（"学习"）：再得到一个 512 维向量

**直觉：** 不同频率的正弦波组合，能唯一编码每个位置，且相对位置关系可以通过线性变换捕捉。

> GPT-2 用的是**可学习的位置嵌入**（另一个 Embedding 矩阵），效果类似但更灵活。

**最终输入 = Token Embedding + Position Encoding**

#### Q：句子越长，后半段的权重会变低吗？

不会。`pos=0` 的编码向量是 512 个数，`pos=1000` 的编码向量也是 512 个数，这两组数的大小范围是一样的，都在 -1 到 1 之间。区别只是数值的**组合模式**不同，从而让模型能区分"这是第 0 个位置"和"这是第 1000 个位置"。

---

### 三、自注意力机制（Self-Attention）—— Transformer 的灵魂

#### 第一步：理解 Q、K、V

对于输入序列中的每个 token，通过三个不同的线性变换生成三个向量：

| 向量 | 角色 | 含义 |
|------|------|------|
| **Query (Q)** | 提问者 | 我想找什么信息？ |
| **Key (K)** | 被查询者的标签 | 我有什么信息？ |
| **Value (V)** | 被查询者的内容 | 我实际的内容是什么？ |

> **比喻：** 你去图书馆找书。Q 是你的需求（"我要机器学习的书"），K 是每本书的标签（"机器学习"、"烹饪"、"历史"），V 是书的实际内容。你用 Q 和每个 K 比较相关度，然后按相关度加权读取 V。

#### 第二步：计算注意力分数

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**1. $QK^T$：计算相关度矩阵**

每个 token 都会生成一个 Q（查询）和一个 K（键），然后让每个 token 和句子里的每个 token（包括自己）算一次点积：

```
              k1(我)   k2(爱)   k3(学习)
q1(我)     │ 我↔我    我↔爱    我↔学习  │
q2(爱)     │ 爱↔我    爱↔爱    爱↔学习  │
q3(学习)   │ 学习↔我  学习↔爱  学习↔学习 │
```

> 这就是 **Self-Attention（自注意力）** 的含义——自己内部互相看。"所有 token"指的是当前输入序列里的全部 token，而非整个训练集。

**2. 除以 $\sqrt{d_k}$：缩放**

防止点积值过大导致 softmax 梯度消失。如果 $d_k=64$，点积可能很大，softmax 输出接近 one-hot，梯度 ≈ 0。

**3. Softmax：归一化成概率分布**（每行加起来 = 1）

示例：

```
e^2.0 = 7.39
e^5.0 = 148.41
e^1.0 = 2.72

总和 = 7.39 + 148.41 + 2.72 = 158.52

我↔我:    7.39  / 158.52 = 0.047
我↔爱:  148.41  / 158.52 = 0.936
我↔学习:  2.72  / 158.52 = 0.017
```

结果：`[0.047, 0.936, 0.017]`，加起来 = 1。

含义："我"这个 token 把 **93.6%** 的注意力放在"爱"上，4.7% 放在自己身上，1.7% 放在"学习"上。

**4. 乘以 V：按注意力权重加权求和，得到输出**

#### Q：Q、K、V 是怎么来的？

V 和 Q、K 一样，都是从同一个 token 的 Embedding 向量通过线性变换得来的：

```
token向量 × W_Q = Q（Query，查询：我想找什么）
token向量 × W_K = K（Key，键：我能被怎样找到）
token向量 × W_V = V（Value，值：我实际携带的信息）
```

以"我爱学习"为例，"爱"这个 token 的 Embedding 向量（512维）会同时生成：

- **Q(爱)**：用来去问"我该关注谁？"
- **K(爱)**：用来被别人匹配"爱这个 token 和你相关吗？"
- **V(爱)**：存放"爱"真正要传递的语义信息

> **为什么要分开？** Q 和 K 做点积是为了算匹配度，匹配完之后你真正要读的是 V（书的内容），不是标签本身。

#### 第三步：多头注意力（Multi-Head Attention）

一组 Q、K、V 只能捕捉一种关系模式。多头就是并行跑多组注意力，每组关注不同的关系：

- **Head 1**：可能关注语法关系（主语-谓语）
- **Head 2**：可能关注语义关系（近义词）
- **Head 3**：可能关注位置关系（相邻词）
- ...

最后把所有 Head 的输出拼接，再过一个线性层合并：

```
MultiHead = Concat(head1, head2, ..., headH) × W_O
```

实现上就是把 `d_model` 拆成 H 个 `d_k = d_model / H`：

```
d_model = 512, H = 8 → 每个头 d_k = 64
```

---

### 四、前馈网络（Feed-Forward Network, FFN）

注意力层之后跟一个两层的全连接网络：

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

```
d_model (512) → 4×d_model (2048) → d_model (512)
                 ↑ 升维，增加表达能力
```

- **注意力层**负责"信息交流"（token 之间互相看）
- **FFN** 负责"信息处理"（每个 token 独立地做非线性变换）

> 有研究认为 FFN 本质上是一个"记忆存储"——它把训练中学到的知识编码在权重中。

拆开来看三步：

1. **升维**（`x × W1`）：输入 `x` 的维度是 `d_model`（如 768），`W1` 把它投影到更大的维度 `d_ff`（通常是 `4 × d_model = 3072`）
2. **GELU 激活**：对升维后的结果逐元素施加非线性函数。GELU 形状类似 ReLU，但在零点附近是平滑的
3. **降维**（`× W2`）：把维度从 `d_ff` 投影回 `d_model`，恢复原来的形状

```python
# W1: (d_model, d_ff)    b1: (d_ff,)
# W2: (d_ff, d_model)    b2: (d_model,)

hidden = GELU(x @ W1 + b1)   # (batch, seq_len, d_ff)
output = hidden @ W2 + b2     # (batch, seq_len, d_model)
```

#### Q：激活函数的必要性

"激活函数"这个名字来源于生物神经元——神经元接收信号，当信号强到一定程度时才会"激活"并向下一个神经元传递信号。

在神经网络中，激活函数对每个数值施加一个**非线性变换**，决定这个信号"通过多少"：

```
输入:  [2.0,  -0.5,  3.1,  -1.8,  0.3]

ReLU（负数变 0，正数不变）:
输出:  [2.0,   0.0,  3.1,   0.0,  0.3]

GELU（负数被压到接近 0，但不完全为 0）:
输出:  [2.0,  -0.15,  3.1,  -0.02,  0.23]
```

可以把它理解成一个**过滤器/阀门**：正的大信号几乎原样通过，负的或很小的信号被大幅抑制。

**为什么需要？** 没有激活函数，不管堆多少层线性变换（矩阵乘法），本质上都只是一次线性变换。加了激活函数之后，网络才能学到"如果 A 且 B 则 C"这类复杂的非线性关系。

---

### 五、残差连接 + LayerNorm

每个子层（Attention、FFN）都包裹着两个关键机制：

```
output = LayerNorm(x + SubLayer(x))    ← Post-Norm（原始论文）
output = x + SubLayer(LayerNorm(x))    ← Pre-Norm （GPT-2 用的）
```

#### 残差连接（Residual Connection）

把输入直接加到输出上。好处是梯度可以直接"跳过"子层回传，解决深层网络梯度消失问题。即使子层学到的东西不好，至少不会比输入差（退化为恒等映射）。

**为什么能解决梯度消失？**

没有残差连接时，梯度的传播路径：

```
梯度 → 第6层 → 第5层 → 第4层 → ... → 第1层
        ×w6     ×w5     ×w4            ×w1
```

每一步都乘权重，层层衰减。

有残差连接时，`output = x + SubLayer(x)`，对它求导后关键是那个 **+1**。不管子层的梯度多小，总有一条"直通高速公路"让梯度以 1 的大小直接回传，不经过任何权重矩阵。

#### LayerNorm（层归一化）

对每个样本的特征维度做归一化（均值=0，方差=1），稳定训练。

示例：假设某个 token 经过某层后得到向量 `[100, 200, 50, 150]`：

```
1. 均值：(100+200+50+150)/4 = 125
2. 标准差：约 55.9
3. 归一化：

   (100-125)/55.9 = -0.45
   (200-125)/55.9 =  1.34
   (50-125)/55.9  = -1.34
   (150-125)/55.9 =  0.45

结果：[-0.45, 1.34, -1.34, 0.45]
```

**为什么需要？** 每一层的输出数值范围可能差异很大。不归一化的话，下一层输入忽大忽小，学习率难以调节，训练容易震荡或发散。

> **简单说：** 残差连接保证梯度传得回去，LayerNorm 保证数值不乱跑，两者配合让深层 Transformer 能稳定训练。

#### Q：Softmax 和归一化的区别

| | LayerNorm（归一化） | Softmax |
|---|---|---|
| **目的** | 稳定数值范围 | 表达相对大小关系 |
| **结果** | 均值=0，方差=1 | 全部为正，加起来=1（概率分布） |

Softmax 的公式：

$$softmax(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

核心是用了指数函数 $e^x$，带来两个效果：

1. **全部变成正数**：不管输入是负数还是正数，$e^x$ 永远大于 0
2. **放大差距**：比如输入 `[1, 3, 1]`：
   ```
   e^1 = 2.72,  e^3 = 20.09,  e^1 = 2.72
   总和 = 25.53
   结果：[0.11, 0.79, 0.11]
   ```
   原本 3 只是 1 的 3 倍，Softmax 之后 0.79 是 0.11 的 7 倍多

> 在 Transformer 里，Softmax 把注意力分数变成概率，让模型明确地"选择"该关注哪些 token。它不是为了稳定训练（那是 LayerNorm 的事），而是为了**做决策和加权**。

---

### 补充：Padding 掩码

训练时一个 batch 里有多个句子，但长度不同：

```
句子1: "我 爱 学习"  → 3 个 token
句子2: "你 好"       → 2 个 token
```

GPU 要求矩阵形状一致，所以短的句子要**填充（pad）**到一样长：

```
句子1: [我,  爱,  学习]
句子2: [你,  好,  PAD ]   ← 补一个无意义的占位符
```

PAD 没有语义，不应该参与注意力计算。所以用 Padding 掩码把 PAD 位置设成 $-\infty$，softmax 后权重为 0，模型就完全忽略这些填充位置。

> **简单说：** 因果掩码是防作弊（不许偷看未来），Padding 掩码是防干扰（不许关注垃圾填充）。实现方式都一样——把不该看的位置设成 $-\infty$。

---

### 六、一层 Transformer 的完整数据流

```
输入: x  (batch_size, seq_len, d_model)
│
├── 1. 残差保存:   residual = x
├── 2. LayerNorm:  x = LN(x)
├── 3. 自注意力:
│      Q = x × W_Q     (batch, seq_len, d_model)
│      K = x × W_K
│      V = x × W_V
│      reshape → (batch, num_heads, seq_len, head_dim)
│      attn = softmax(QK^T / √d_k + causal_mask) × V
│      attn = Concat(heads) × W_O
├── 4. 残差连接:   x = residual + attn
│
├── 5. 残差保存:   residual = x
├── 6. LayerNorm:  x = LN(x)
├── 7. FFN:
│      x = GELU(x × W1) × W2
├── 8. 残差连接:   x = residual + x
│
└── 输出: x  (batch_size, seq_len, d_model)
```

---

### 七、为什么 Transformer 这么强

- **并行性**：不像 RNN 逐步计算，所有位置可以同时处理
- **长距离依赖**：任意两个位置之间只有一步注意力的距离（RNN 需要经过中间所有步骤）
- **可扩展性**：结构简单统一，容易堆叠更多层、更多头，配合更多数据就能持续变强（scaling law）

### 实战：一个简单的 GPT-2 模型的生成与使用

#### GPT-2 配置参数与笔记概念对照

```python
mini_config = GPT2Config(
    vocab_size=1000,
    n_positions=64,
    n_embd=64,
    n_layer=2,
    n_head=2,
    n_inner=128,
    activation_function="gelu",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)
```

| 参数 | 值 | 对应笔记概念 |
|------|-----|-------------|
| `vocab_size` | 1000 | **一、词嵌入** — Embedding 矩阵的行数，决定能表示多少个不同的 token |
| `n_positions` | 64 | **二、位置编码** — 最大序列长度，位置编码矩阵的行数（GPT-2 用可学习的位置嵌入） |
| `n_embd` | 64 | **一、词嵌入** — 就是 `d_model`，每个 token 向量的维度 |
| `n_layer` | 2 | **六、完整数据流** — 堆叠多少个 Transformer 层（每层包含一次自注意力 + 一次 FFN） |
| `n_head` | 2 | **三、多头注意力** — 注意力头的数量 H，每个头的维度 `d_k = n_embd / n_head = 32` |
| `n_inner` | 128 | **四、FFN** — 就是 `d_ff`，前馈网络升维后的维度（通常是 `4 × d_model`，这里 `128 = 2 × 64`） |
| `activation_function` | `"gelu"` | **四、FFN** — FFN 中间的激活函数，GELU 在零点附近比 ReLU 更平滑 |
| `resid_pdrop` | 0.1 | **五、残差连接** — 残差连接之后的 dropout |
| `embd_pdrop` | 0.1 | **一 + 二** — Token Embedding + Position Encoding 之后的 dropout |
| `attn_pdrop` | 0.1 | **三、自注意力** — softmax 之后、乘以 V 之前的 dropout |

> 三个 dropout 参数的作用：训练时随机丢弃 10% 的神经元输出（置为 0），防止过拟合。推理时不生效。


#### 结合源码：配置如何生成模型？词汇从哪来？怎么训练？

---

#### Q1：配置是怎么生成模型的？

`GPT2LMHeadModel(mini_config)` 这一行触发了整个模型的构建，源码调用链如下：

```
GPT2LMHeadModel.__init__(config)
├── self.transformer = GPT2Model(config)
│   ├── self.wte = nn.Embedding(vocab_size=1000, n_embd=64)    # token嵌入矩阵 [1000×64]
│   ├── self.wpe = nn.Embedding(n_positions=64, n_embd=64)     # 位置嵌入矩阵 [64×64]
│   ├── self.drop = nn.Dropout(embd_pdrop=0.1)                 # 嵌入后的dropout
│   ├── self.h = ModuleList([                                   # n_layer=2 个Block
│   │   GPT2Block(config, layer_idx=0)
│   │   GPT2Block(config, layer_idx=1)
│   │   ])
│   │   每个Block内部：
│   │   ├── self.ln_1 = LayerNorm(64)                           # 注意力前的LayerNorm
│   │   ├── self.attn = GPT2Attention(config)
│   │   │   ├── self.c_attn = Conv1D(3*64, 64)                 # 一次性生成Q,K,V [64→192]
│   │   │   ├── self.c_proj = Conv1D(64, 64)                   # 输出投影
│   │   │   ├── self.attn_dropout = Dropout(0.1)
│   │   │   └── self.resid_dropout = Dropout(0.1)
│   │   ├── self.ln_2 = LayerNorm(64)                           # FFN前的LayerNorm
│   │   └── self.mlp = GPT2MLP(n_inner=128, config)
│   │       ├── self.c_fc = Conv1D(128, 64)                    # 升维 64→128
│   │       ├── self.act = gelu                                 # 激活函数
│   │       ├── self.c_proj = Conv1D(64, 128)                  # 降维 128→64
│   │       └── self.dropout = Dropout(0.1)
│   └── self.ln_f = LayerNorm(64)                               # 最终LayerNorm
└── self.lm_head = nn.Linear(64, 1000, bias=False)              # 输出层 [64→1000]
```

> 叫"隐藏层"是因为这些中间向量是模型内部的表示，不是输入（token id）也不是最终输出（词汇表概率），是藏在模型里面的中间状态。

关键点：

- 所有权重都是**随机初始化**的（`initializer_range=0.02` 的正态分布），不是预训练的
- `lm_head.weight` 和 `wte.weight` 是**权重绑定**的（tied weights），即输入嵌入矩阵和输出投影矩阵共享同一份参数
- 源码里 Q、K、V 不是三个独立矩阵，而是用一个 `Conv1D(3*n_embd, n_embd)` 一次算出来再 split，效率更高

---

#### Q2：词汇从哪里来？

`vocab_size=1000` 只是告诉模型"词汇表有 1000 个 token"，但**并没有定义这 1000 个 token 分别是什么**。

实际的词汇表由 **Tokenizer** 提供，和模型是分开的两个东西：

```python
# 模型只知道：有1000个ID（0~999），每个对应一个64维向量
self.wte = nn.Embedding(1000, 64)   # 随机初始化

# Tokenizer负责：文字 ↔ ID 的映射
# GPT-2 官方用的是 BPE (Byte Pair Encoding) tokenizer
# 词汇表存在 vocab.json 和 merges.txt 里
```

测试代码里用的是随机生成的假 token ID，没有经过 tokenizer，所以输出也是无意义的：

```python
input_ids = torch.randint(0, 1000, (batch_size, seq_length))
```

真正使用时需要：

```python
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # 加载官方词汇表(50257个token)
input_ids = tokenizer.encode("我爱学习", return_tensors="pt")
```

---

#### Q3：怎么训练的？

代码里的这一行其实已经展示了训练的核心逻辑：

```python
outputs = model(input_ids, labels=input_ids)
```

源码中 `GPT2LMHeadModel.forward` 做了这些事：

```
输入: input_ids = [我, 爱, 学, 习]

1. 前向传播，得到每个位置的 logits (形状: [batch, seq_len, vocab_size])
   logits[0] = 模型在位置0预测的下一个token的概率分布
   logits[1] = 模型在位置1预测的下一个token的概率分布
   ...

2. 计算loss时，labels向左移一位（源码自动处理）：
   位置0的预测 → 应该预测出位置1的token（"爱"）
   位置1的预测 → 应该预测出位置2的token（"学"）
   位置2的预测 → 应该预测出位置3的token（"习"）

3. 用 CrossEntropyLoss 计算预测和真实label之间的差距
```

这就是 **Causal Language Modeling（因果语言模型）** 的训练方式——给定前面的 token，预测下一个 token。

完整的训练循环：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    input_ids = batch["input_ids"]
    outputs = model(input_ids, labels=input_ids)  # 前向 + 算loss
    loss = outputs.loss
    loss.backward()          # 反向传播，计算梯度
    optimizer.step()         # 更新权重
    optimizer.zero_grad()    # 清零梯度
```

> 每一步都在让模型的预测更接近真实的下一个 token，Embedding 矩阵、Q/K/V 权重、FFN 权重全部在这个过程中被更新。

---

### 八、gpt2_test 源码分析：断点调试指南

> 以下是按前向传播顺序推荐的断点位置，文件都在 `modeling_gpt2.py` 中。

#### 1. 入口 — `GPT2LMHeadModel.forward()`

- `modeling_gpt2.py:757` — `return_dict = ...`

这是调用 `model(input_ids, labels=input_ids)` 的入口，从这里开始跟踪。

#### 2. Embedding 层 — `GPT2Model.forward()`

- `modeling_gpt2.py:608` — `inputs_embeds = self.wte(input_ids)` — token embedding
- `modeling_gpt2.py:618` — `position_embeds = self.wpe(position_ids)` — 位置编码
- `modeling_gpt2.py:619` — `hidden_states = inputs_embeds + position_embeds` — 两者相加

> 在这里可以观察 `inputs_embeds` 和 `position_embeds` 的 shape 和值。

#### 3. Causal Mask 创建

- `modeling_gpt2.py:625` — `causal_mask = create_causal_mask(...)`

> 看看因果注意力掩码长什么样。

#### 4. Transformer Block 循环

- `modeling_gpt2.py:658` — `outputs = block(hidden_states, ...)`

> 循环遍历每一层 Block 的地方（配置有 2 层）。

#### 5. 单个 Block 内部 — `GPT2Block.forward()`

- `modeling_gpt2.py:287` — `hidden_states = self.ln_1(hidden_states)` — 第一个 LayerNorm
- `modeling_gpt2.py:288` — `attn_output, ... = self.attn(...)` — 自注意力
- `modeling_gpt2.py:298` — `hidden_states = attn_output + residual` — 残差连接 1
- `modeling_gpt2.py:321` — `hidden_states = self.ln_2(hidden_states)` — 第二个 LayerNorm
- `modeling_gpt2.py:322` — `feed_forward_hidden_states = self.mlp(hidden_states)` — FFN
- `modeling_gpt2.py:324` — `hidden_states = residual + feed_forward_hidden_states` — 残差连接 2

#### 6. 注意力机制内部 — `GPT2Attention.forward()`

- `modeling_gpt2.py:194` — `query, key, value = self.c_attn(hidden_states).split(...)` — QKV 投影
- `modeling_gpt2.py:200` — `query_states = query_states.view(shape_q).transpose(1, 2)` — reshape 成多头

#### 7. 注意力计算 — `eager_attention_forward()`

- `modeling_gpt2.py:53` — `attn_weights = torch.matmul(query, key.transpose(-1, -2))` — $QK^T$
- `modeling_gpt2.py:68` — `attn_weights = nn.functional.softmax(...)` — softmax
- `modeling_gpt2.py:74` — `attn_output = torch.matmul(attn_weights, value)` — attention × V

> 这是注意力的核心，可以观察 attention weights 的分布。

#### 8. MLP 内部 — `GPT2MLP.forward()`

- `modeling_gpt2.py:251` — `hidden_states = self.c_fc(hidden_states)` — 升维线性层
- `modeling_gpt2.py:252` — `hidden_states = self.act(hidden_states)` — GELU 激活
- `modeling_gpt2.py:253` — `hidden_states = self.c_proj(hidden_states)` — 降维线性层

#### 9. 最终输出 — 回到 GPT2Model & LMHead

- `modeling_gpt2.py:678` — `hidden_states = self.ln_f(hidden_states)` — 最终 LayerNorm
- `modeling_gpt2.py:777` — `logits = self.lm_head(hidden_states[:, slice_indices, :])` — 映射到词表
- `modeling_gpt2.py:782` — `loss = self.loss_function(logits, labels, ...)` — 计算交叉熵损失

#### 完整数据流总结

```
input_ids → wte(token emb) + wpe(pos emb) → dropout
  → [Block × 2]:
      → LayerNorm → Attention(QKV投影 → Q·K^T → softmax → ×V → 输出投影) → 残差
      → LayerNorm → MLP(升维 → GELU → 降维) → 残差
  → LayerNorm → lm_head(线性层) → logits → CrossEntropyLoss
```

> 在 PyCharm 中，按住 Cmd 点击 `GPT2LMHeadModel` 就能跳转到源码，然后在上面这些行号打断点，Debug 运行 `gpt2_test.py` 就能逐步跟踪整个流程了。


#### 阅读源码：Python 对象创建流程

```python
model = GPT2Model(config)   # __init__ 执行，搭建结构
output = model(input_ids)   # __call__ → forward 执行，数据流过
output2 = model(input_ids2) # forward 再次执行
```

**`__init__` 只跑一次，`forward` 每次推理/训练都跑。**

调用链总结：

```
model(input_ids)
  → __call__
    → GPT2LMHeadModel.forward()     ← 你跳转到的位置
      → self.transformer(input_ids)
        → GPT2Model.forward()        ← 内部再调用 transformer 主体
      → self.lm_head(hidden_states)  ← 输出 logits
```

> 两层 forward，外层是 `LMHeadModel`，内层是 `GPT2Model`。

---

#### Attention 内部的赋值与预训练

`_init_weights(self, module)` 是权重初始化方法，在模型创建时对每个子模块的参数赋初始值。

```python
self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
```

这一行就是 W_QKV 的定义（约第 190 行），一次性生成 Q、K、V。

> 没用 `from_pretrained("gpt2")` 加载预训练权重，而是用 `GPT2LMHeadModel(mini_config)` 从零创建，所以权重是**随机初始化**的。这就是为什么 loss 会很高——模型还没训练过。

---

#### 随机初始化的含义

**随机初始化**就是所有权重（包括生成 Q、K、V 的权重矩阵）都是用随机数填充的，没有经过任何训练。

以 Q、K、V 的权重为例：

```python
# 模型创建时，W_QKV 被初始化为：
W_QKV = N(0, 0.02)  # 从均值0、标准差0.02的正态分布随机采样

# 也就是说矩阵里全是类似这样的随机小数：
[[ 0.013, -0.007,  0.021, ...],
 [-0.015,  0.003,  0.018, ...],
 [ 0.009, -0.022,  0.001, ...]]
```

#### 随机初始化的缺点

**模型什么都不"懂"：**

- **Q 不知道该关注什么** —— 随机的 Q 生成的查询没有语义意义
- **K 不知道该匹配什么** —— 随机的 K 无法正确表示"我有什么信息"
- **V 传递的是垃圾信息** —— 随机的 V 没有学到有用的特征

具体表现：

- 注意力分数接近**均匀分布**（每个 token 对其他 token 的关注差不多），因为 Q·K^T 算出来的值都是随机噪声
- 最终预测下一个 token 时，相当于在词表上**随机猜**
- loss ≈ `ln(1000)` ≈ 6.9（词表大小 1000 的随机猜测交叉熵）

#### 对比预训练模型

```python
# 随机初始化（测试代码）
model = GPT2LMHeadModel(mini_config)              # loss ≈ 6.9

# 加载预训练权重
model = GPT2LMHeadModel.from_pretrained("gpt2")   # loss 很低
```

预训练模型的 W_Q、W_K、W_V 经过了海量文本训练，已经学会了：

- Q 能提出有意义的查询（比如"前面有没有主语？"）
- K 能正确标识自己的语法/语义角色
- V 能传递有用的上下文信息

> **总结：** 模型所有参数（不只是 QKV，还有 Embedding、MLP、LayerNorm 等）都是随机数，相当于一个"白纸"状态的模型，需要经过训练才能产生有意义的输出。

---