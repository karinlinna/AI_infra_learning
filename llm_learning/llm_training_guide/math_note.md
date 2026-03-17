# 数学基础笔记

> LLM 训练涉及的核心数学知识：线性代数、概率统计、微积分。

---

## 一、线性代数

### 1.1 矩阵运算

矩阵就是一个数字方阵，深度学习里几乎所有计算都是矩阵运算。

**矩阵乘法：**

$$A_{(2 \times 3)} \times B_{(3 \times 2)} = C_{(2 \times 2)}$$

$$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \times \begin{bmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{bmatrix} = \begin{bmatrix} 58 & 64 \\ 139 & 154 \end{bmatrix}$$

规则：左矩阵的行 **点乘** 右矩阵的列。左边列数必须等于右边行数。

**在 LLM 中的用途：**

神经网络的每一层本质上就是矩阵乘法：

$$\text{output} = \text{input} \times W + b$$

| 变量 | 形状 | 含义 |
|------|------|------|
| input | $[1, 512]$ | 一个 token 的 embedding（512 维向量） |
| $W$ | $[512, 512]$ | 权重矩阵（模型参数） |
| output | $[1, 512]$ | 变换后的向量 |

Attention 的 Q、K、V 计算、FFN 层，全都是矩阵乘法。

---

### 1.2 特征值分解（Eigendecomposition）

对一个方阵 $A$，如果存在向量 $v$ 和标量 $\lambda$ 满足：

$$A v = \lambda v$$

那么 $v$ 是**特征向量**，$\lambda$ 是**特征值**。

**直觉理解：**

矩阵 $A$ 作用到大多数向量上，既会改变方向又会改变长度。但特征向量很特殊——$A$ 作用到它身上只改变长度（缩放 $\lambda$ 倍），不改变方向。

```
普通向量:   A × [1, 1] = [3, 2]              ← 方向变了，长度也变了
特征向量:   A × [2, 1] = [6, 3] = 3 × [2, 1] ← 方向没变，只是变成了 3 倍长
                                     ↑
                                   特征值 λ=3
```

**验证例子：**

$$A = \begin{bmatrix} 2 & 2 \\ 1 & 3 \end{bmatrix}, \quad v = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

$$A v = \begin{bmatrix} 2 \times 2 + 2 \times 1 \\ 1 \times 2 + 3 \times 1 \end{bmatrix} = \begin{bmatrix} 6 \\ 3 \end{bmatrix} = 3 \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \lambda v \quad \checkmark$$

**特征值分解：** 把矩阵拆成特征向量和特征值的组合

$$A = V \Lambda V^{-1}$$

- $V$：特征向量组成的矩阵
- $\Lambda$：对角线上放特征值的对角矩阵

**在 ML 中的用途：** PCA 降维、理解数据的主要变化方向、协方差矩阵分析等。

---

### 1.3 PCA 降维（Principal Component Analysis）

**目标：** 把高维数据投影到低维，同时尽量保留信息（方差最大的方向）。

**例子：** 100 个学生，每人有 5 门课成绩（5 维数据），想降到 2 维来可视化。

```
原始数据（5 维）:
学生1: [数学90, 物理88, 化学85, 语文70, 英语72]
学生2: [数学60, 物理55, 化学58, 语文92, 英语90]
...

问题: 哪 2 个方向最能区分这些学生？
```

**步骤：**

**第 1 步：** 计算协方差矩阵 $C$（$5 \times 5$），描述各科成绩之间的相关性

**第 2 步：** 对 $C$ 做特征值分解 $C = V \Lambda V^T$

得到 5 个特征值和 5 个特征向量：

| 特征值 | 特征向量 | 含义 |
|--------|---------|------|
| $\lambda_1 = 25$ | $v_1 = [0.5,\; 0.5,\; 0.5,\; -0.1,\; -0.1]$ | "理科能力"方向 |
| $\lambda_2 = 15$ | $v_2 = [-0.1,\; -0.1,\; -0.1,\; 0.6,\; 0.6]$ | "文科能力"方向 |
| $\lambda_3 = 3$ | $v_3 = \ldots$ | 不太重要 |
| $\lambda_4 = 1$ | $v_4 = \ldots$ | 很不重要 |
| $\lambda_5 = 0.5$ | $v_5 = \ldots$ | 噪声 |

**第 3 步：** 取前 2 个特征向量（对应最大的特征值），作为新坐标轴

$$X_{\text{new}} = X \times [v_1, v_2] \quad \Rightarrow \quad \text{从 5 维降到 2 维}$$

**结果：**

```
原来: 每个学生是 5 个数字
现在: 每个学生是 2 个数字（理科得分, 文科得分）

    文科 ↑
         |  ● 文科生
         |  ●
         |      ● 均衡型
         |
         |              ● 理科生
         |              ●
         +-------------------→ 理科
```

- **特征向量**告诉你数据变化最大的方向是什么（新坐标轴）
- **特征值**告诉你那个方向上变化有多大（重要程度）

---

### 1.4 理解数据的主要变化方向

PCA 的核心思想，用一个更直观的例子：

```
2D 数据点分布:

    y ↑      . . .
      |    . . . . .
      |  . . . . . . .        ← 数据像一个倾斜的椭圆
      |    . . . . .
      |      . . .
      +------------------→ x
```

对这些点的协方差矩阵做特征值分解：

```
    y ↑      . . .
      |    . . . . .
      |  . . . .v₁. . .      v₁（特征值大）= 椭圆的长轴方向
      |    . . /. .            → 数据在这个方向上散布最广
      |      ./. .             → 这是"主要变化方向"
      +------/───────────→ x
             |
             v₂（特征值小）= 椭圆的短轴方向
              → 数据在这个方向上变化不大
```

- **特征值大** = 这个方向上数据变化剧烈 = 重要信息
- **特征值小** = 这个方向上数据变化很小 = 可以丢掉（降维）

---

### 1.5 协方差矩阵分析

协方差矩阵描述多个变量之间两两相关程度。

**例子：** 三个变量——身高(H)、体重(W)、鞋码(S)

$$C = \begin{array}{c|ccc} & H & W & S \\ \hline H & 25 & 15 & 10 \\ W & 15 & 20 & 8 \\ S & 10 & 8 & 9 \end{array}$$

- **对角线：** 各变量自身的方差
- **非对角线：** 两个变量的协方差（正数=正相关，负数=负相关）

含义：
- $C[H,H]=25$ → 身高的方差是 25（离散程度）
- $C[H,W]=15$ → 身高和体重正相关（高的人倾向于重）
- $C[H,S]=10$ → 身高和鞋码正相关

对协方差矩阵做特征值分解：

$$C \cdot v_1 = \lambda_1 \cdot v_1$$

| 特征值 | 特征向量 | 含义 |
|--------|---------|------|
| $\lambda_1 = 42$ | $v_1 = [0.65,\; 0.60,\; 0.45]$ | "体型大小"方向 |
| $\lambda_2 = 8$ | $v_2 = [0.4,\; -0.7,\; 0.5]$ | "身高高但体重轻"方向 |
| $\lambda_3 = 2$ | $v_3 = \ldots$ | 噪声 |

这告诉你：
- $v_1$：身高、体重、鞋码都是正系数 → 主要沿"体型大小"这一个维度共同变化
- $\lambda_1 = 42$ 远大于其他 → 这三个变量 80% 的变化都可以用"体型大小"这一个因素解释
- 实际上你只需要一个数字（"体型大小"得分）就能大致描述一个人的身高、体重、鞋码

#### 协方差矩阵的计算过程

**第一步：中心化（每个值减去均值）**

$$X = X_{\text{raw}} - \mu$$

$$X = \begin{bmatrix} -3 & -4 & -0.4 \\ 2 & 4 & 0.6 \\ -5 & -9 & -1.4 \\ 7 & 11 & 1.6 \\ -1 & -2 & -0.4 \end{bmatrix}$$

**第二步：计算协方差矩阵**

$$C = \frac{X^T X}{n}$$

逐个算出来：

$$C[H,H] = \frac{(-3)^2 + 2^2 + (-5)^2 + 7^2 + (-1)^2}{5} = \frac{88}{5} = 17.6$$

$$C[H,W] = \frac{(-3)(-4) + 2 \times 4 + (-5)(-9) + 7 \times 11 + (-1)(-2)}{5} = \frac{144}{5} = 28.8$$

最终协方差矩阵：

$$C = \begin{bmatrix} 17.6 & 28.8 & 4.16 \\ 28.8 & 47.6 & 6.56 \\ 4.16 & 6.56 & 1.04 \end{bmatrix}$$

**怎么读这个矩阵：**

- 对角线 → 方差：$C[W,W] = 47.6$ 体重的离散程度更大，$C[S,S] = 1.04$ 鞋码比较集中
- 非对角线 → 协方差：$C[H,W] = 28.8$ 身高与体重强正相关
- 矩阵是**对称的**：$C[H,W] = C[W,H]$（因为乘法交换律）

#### 公式总结

设有 $n$ 个样本，每个样本 $d$ 个特征：

$$\mu = \text{每列的均值} \quad (1 \times d)$$

$$X = X_{\text{raw}} - \mu \quad \text{（中心化）}$$

$$C = \frac{X^T X}{n} \quad (d \times d \text{ 的对称方阵})$$

维度变化：$X^T_{(d \times n)} \times X_{(n \times d)} = C_{(d \times d)}$

- 3 个特征 → 协方差矩阵是 $3 \times 3$
- 4096 维 embedding → 协方差矩阵是 $4096 \times 4096$

#### 完整流程：协方差矩阵 → 特征值分解 → PCA

```
协方差矩阵 C (d×d)
       ↓
特征值分解: C = V Λ Vᵀ
       ↓
特征向量 V 的每一列 = 数据变化的一个主方向
特征值 Λ 的对角线   = 每个方向上的变化程度（方差）
       ↓
取前 k 个最大特征值对应的特征向量
       ↓
投影: X_new = X × V_k  →  从 d 维降到 k 维  ← 这就是 PCA
```

#### 总结对比

| 应用 | 特征向量告诉你什么 | 特征值告诉你什么 |
|------|-------------------|-----------------|
| PCA 降维 | 新坐标轴的方向（投影方向） | 每个方向保留了多少信息 |
| 主要变化方向 | 数据散布最广的方向 | 散布的程度有多大 |
| 协方差矩阵 | 变量之间共同变化的模式 | 每种模式解释了多少整体变化 |

三者其实是**同一件事的不同说法**：对协方差矩阵做特征值分解 = PCA = 找数据的主要变化方向。

---

### 1.6 SVD（奇异值分解，Singular Value Decomposition）

特征值分解只能用于方阵，SVD 可以分解**任意形状**的矩阵：

$$A = U \Sigma V^T$$

| 矩阵 | 形状 | 含义 |
|------|------|------|
| $A$ | $m \times n$ | 任意矩阵 |
| $U$ | $m \times m$ | 正交矩阵（左奇异向量） |
| $\Sigma$ | $m \times n$ | 对角矩阵（奇异值，从大到小排列） |
| $V^T$ | $n \times n$ | 正交矩阵（右奇异向量） |

**直觉理解：** 任何矩阵变换 = 旋转 × 缩放 × 旋转

$$\text{原始数据} \xrightarrow{V^T \text{（旋转）}} \xrightarrow{\Sigma \text{（缩放）}} \xrightarrow{U \text{（再旋转）}} \text{变换后数据}$$

**关键性质：** 奇异值从大到小排列，前面几个奇异值包含了矩阵的主要信息，后面的可以丢掉。

**低秩近似的例子：**

| 表示方式 | 参数量 |
|---------|--------|
| 原始矩阵 $A_{(4096 \times 4096)}$ | 16M 个参数 |
| 低秩近似 $A \approx U_{16} \Sigma_{16} V_{16}^T$（只保留前 16 个奇异值） | $4096 \times 16 + 16 + 16 \times 4096 \approx$ **131K**（压缩 100 多倍） |

#### SVD 的具体分解实例

假设有一个 $2 \times 3$ 的矩阵（2 个用户对 3 部电影的评分）：

$$A = \begin{bmatrix} 3 & 1 & 1 \\ 1 & 3 & 1 \end{bmatrix}$$

SVD 分解：$A = U \Sigma V^T$

$$\begin{bmatrix} 3 & 1 & 1 \\ 1 & 3 & 1 \end{bmatrix} = \begin{bmatrix} -0.71 & 0.71 \\ -0.71 & -0.71 \end{bmatrix} \begin{bmatrix} 3.46 & 0 & 0 \\ 0 & 2.00 & 0 \end{bmatrix} \begin{bmatrix} -0.71 & -0.71 & 0 \\ 0.41 & -0.41 & 0.82 \\ 0.58 & -0.58 & -0.58 \end{bmatrix}$$

**三个矩阵各自的含义：**

怎么读奇异向量的系数：
1. **绝对值大小** → 这个元素在该模式中的"权重"有多大
2. **符号相同/相反** → 元素之间是"同向"还是"对立"
3. **接近零** → 该元素跟这个模式基本无关

**$U$ —— 行的模式（用户偏好模式）：**

$$U = \begin{bmatrix} -0.71 & 0.71 \\ -0.71 & -0.71 \end{bmatrix}$$

- 模式 1：$[-0.71,\; -0.71]$ → 两个用户权重相同 → "整体喜好"
- 模式 2：$[0.71,\; -0.71]$ → 两个用户权重相反 → "品味差异"

**$\Sigma$ —— 每种模式的重要程度：**

$$\Sigma = \begin{bmatrix} 3.46 & 0 & 0 \\ 0 & 2.00 & 0 \end{bmatrix}$$

- $\sigma_1 = 3.46$ → 模式 1（整体喜好）很重要
- $\sigma_2 = 2.00$ → 模式 2（品味差异）也重要，但稍弱

**$V^T$ —— 列的模式（电影特征模式）：**

$$V^T = \begin{bmatrix} -0.71 & -0.71 & 0 \\ 0.41 & -0.41 & 0.82 \\ 0.58 & -0.58 & -0.58 \end{bmatrix}$$

- 模式 1：$[-0.71,\; -0.71,\; 0]$ → 电影 1 和电影 2 同等重要，电影 3 不相关
- 模式 2：$[0.41,\; -0.41,\; 0.82]$ → 电影 1 和电影 2 相反，电影 3 独立

#### SVD 和特征值分解的关系

| | 特征值分解 | SVD |
|---|-----------|-----|
| 公式 | $A = V \Lambda V^{-1}$ | $A = U \Sigma V^T$ |
| 限制 | 只能用于方阵，且不一定存在 | 任意矩阵都能分解，一定存在 |

它们的数学联系：

$$A^T A = (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T U^T U \Sigma V^T = V \Sigma^2 V^T$$

这就是 $A^T A$ 的特征值分解！

所以：
- $V$ 的列 = $A^T A$ 的特征向量
- $\Sigma^2$ 的对角线 = $A^T A$ 的特征值
- $\sigma_i = \sqrt{\lambda_i}$（奇异值 = 特征值的平方根）

---

### 1.7 SVD → LoRA 的直接联系

LoRA 的核心假设来自 SVD：**微调时权重的变化 $\Delta W$ 是低秩的。**

对 $\Delta W_{(4096 \times 4096)}$ 做 SVD：

$$\Delta W = U \Sigma V^T$$

发现奇异值分布：

$$\sigma_1 = 5.2, \quad \sigma_2 = 3.1, \quad \sigma_3 = 0.8, \quad \sigma_4 = 0.01, \quad \sigma_5 = 0.001, \quad \ldots$$

$$\underbrace{\sigma_1, \sigma_2, \sigma_3}_{\text{有意义的变化}} \quad \underbrace{\sigma_4, \sigma_5, \ldots}_{\text{几乎为零，可以丢掉}}$$

只保留前 $r$ 个：$\Delta W \approx U_r \Sigma_r V_r^T$

**LoRA 简化：** 直接学两个小矩阵，不做 SVD

$$\Delta W = B \times A$$

| 矩阵 | 形状 | 类比 |
|------|------|------|
| $B$ | $4096 \times 16$ | 类比 $U_r \Sigma_r$ |
| $A$ | $16 \times 4096$ | 类比 $V_r^T$ |

参数量：$4096 \times 16 \times 2 = $ **131K**，而不是 $4096 \times 4096 = $ **16.8M**

**具体对比：**

$$B_{(4 \times 1)} \times A_{(1 \times 4)} = \Delta W_{(4 \times 4)}$$

$$\begin{bmatrix} 0.3 \\ 0.88 \\ -0.3 \\ 0.59 \end{bmatrix} \times \begin{bmatrix} 0.7 & 0.7 & 0.35 & 0.7 \end{bmatrix} = \begin{bmatrix} 0.21 & 0.21 & 0.105 & 0.21 \\ 0.62 & 0.62 & 0.308 & 0.62 \\ -0.21 & -0.21 & -0.105 & -0.21 \\ 0.41 & 0.41 & 0.207 & 0.41 \end{bmatrix} \approx \Delta W$$

**LoRA 训练时完全不做 SVD，不算奇异值。** 你只需要：

1. 选一个 $r$（比如 8）
2. 随机初始化 $B$ 和 $A$
3. 直接训练

SVD 只是理论上解释了"为什么 LoRA 能 work"——因为 $\Delta W$ 天然是低秩的。但实际训练时，$B$ 和 $A$ 通过梯度下降自己就能学到那些重要的方向。

> 打个比方：SVD 是"事后验尸"发现规律，LoRA 是直接利用这个规律去构造，跳过了验尸那一步。

#### 完整故事线

1. 研究者做实验：对大模型做全量微调
2. 微调完拿到 $\Delta W = W' - W$
3. 对 $\Delta W$ 做 SVD，发现奇异值：$[5.2,\; 3.1,\; 0.8,\; 0.01,\; 0.001,\; \ldots]$（只有几个大的，后面全接近零）
4. 结论：$\Delta W$ 是低秩的，4096 个维度里只有十几个有意义
5. LoRA 的想法：既然如此，一开始就别训练 $4096 \times 4096$，直接训练 $B_{(4096 \times 16)} \times A_{(16 \times 4096)}$ 就够了
6. 效果：参数量降 100 倍，效果接近全量微调

---

## 二、概率统计

### 2.1 贝叶斯定理（Bayes' Theorem）

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

| 符号 | 名称 | 含义 |
|------|------|------|
| $P(A \mid B)$ | 后验概率 | 看到证据 $B$ 之后，对 $A$ 的信念更新 |
| $P(B \mid A)$ | 似然 | 如果 $A$ 为真，观察到 $B$ 的可能性 |
| $P(A)$ | 先验概率 | 看到证据之前，对 $A$ 的初始信念 |
| $P(B)$ | 边际概率 | 观察到 $B$ 的总概率（归一化常数） |

**例子：** 一封邮件包含"中奖"这个词，它是垃圾邮件的概率？

已知：
- $P(\text{垃圾}) = 0.3$ ← 先验：30% 的邮件是垃圾邮件
- $P(\text{"中奖"} \mid \text{垃圾}) = 0.8$ ← 似然：垃圾邮件中 80% 含"中奖"
- $P(\text{"中奖"} \mid \text{正常}) = 0.01$ ← 正常邮件中 1% 含"中奖"

全概率公式：

$$P(\text{"中奖"}) = P(\text{"中奖"} \mid \text{垃圾}) \cdot P(\text{垃圾}) + P(\text{"中奖"} \mid \text{正常}) \cdot P(\text{正常})$$

$$= 0.8 \times 0.3 + 0.01 \times 0.7 = 0.247$$

代入贝叶斯定理：

$$P(\text{垃圾} \mid \text{"中奖"}) = \frac{0.8 \times 0.3}{0.247} = 0.97$$

→ 看到"中奖"后，是垃圾邮件的概率从 **30% 飙升到 97%**。

核心思想：虽然垃圾邮件只占 30%，但"中奖"这个词在垃圾邮件里出现的频率远高于正常邮件，所以一旦看到"中奖"，垃圾邮件的概率就飙到了 97%。这就是**贝叶斯更新**——用新证据修正先验信念得到后验信念。

**在 LLM 中：** 语言模型本质上就在做贝叶斯推理——根据已经看到的上文（证据），更新对下一个 token 的概率分布（后验）。不是说它内部真的在算贝叶斯公式，而是这个过程在概念上是等价的。

---

### 2.2 概率分布（Probability Distribution）

描述随机变量取各个值的概率。

**离散分布（和 LLM 直接相关）：**

模型输出的 logits 经过 softmax 后就是一个离散概率分布：

$$P(\text{猫}) = 0.35, \quad P(\text{狗}) = 0.25, \quad P(\text{鱼}) = 0.20, \quad P(\text{鸟}) = 0.10, \quad \ldots$$

$$\sum_i P(x_i) = 1.0$$

**常见分布：**

| 分布 | 公式 | 例子 |
|------|------|------|
| 均匀分布 | $P(x) = \frac{1}{n}$ | 掷骰子：每面概率 $= \frac{1}{6}$ |
| 正态分布 | $\mathcal{N}(\mu, \sigma^2)$ | 模型权重初始化 |
| 伯努利分布 | $P(1) = p$ | 抛硬币：正面概率 $p$ |
| 多项分布 | 骰子的推广 | LLM 采样时从词表中选词 |

**Softmax 函数——把任意实数变成概率分布：**

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

例子：

$$\text{输入 logits: } [2.0,\; 1.0,\; 0.1]$$

$$e^{2.0} = 7.39, \quad e^{1.0} = 2.72, \quad e^{0.1} = 1.11, \quad \text{总和} = 11.22$$

$$\text{输出概率: } \left[\frac{7.39}{11.22},\; \frac{2.72}{11.22},\; \frac{1.11}{11.22}\right] = [0.659,\; 0.242,\; 0.099]$$

$$\text{总和} = 1.0 \quad \checkmark$$

**为什么选 softmax？** $e^x$ 天然满足 $> 0$，保留了大小关系，同时放大差距。它来自统计学里的最大熵原理和对数线性模型。

---

### 2.3 最大似然估计（MLE, Maximum Likelihood Estimation）

**核心问题：** 给定观测到的数据，找到最可能产生这些数据的模型参数。

**例子：**

观测到抛硬币 10 次：8 次正面，2 次反面。硬币正面概率 $p$ 最可能是多少？

似然函数：

$$L(p) = p^8 \cdot (1-p)^2$$

取对数再求导，令导数 = 0：

$$\frac{d}{dp}\left[8 \ln(p) + 2 \ln(1-p)\right] = 0$$

$$\frac{8}{p} - \frac{2}{1-p} = 0 \quad \Rightarrow \quad \hat{p} = 0.8$$

**LLM 训练就是在做最大似然估计：**

训练数据："今天天气很好"

在模型参数 $\theta$ 下，这句话出现的概率：

$$L(\theta) = P(\text{"今"} \mid \theta) \times P(\text{"天"} \mid \text{"今"}, \theta) \times P(\text{"天"} \mid \text{"今天"}, \theta) \times \cdots$$

训练目标：找到 $\theta$ 使得 $L(\theta)$ 最大，等价于最小化 $-\log L(\theta)$：

$$-\log L(\theta) = \sum_t -\log P(x_t \mid x_1 \ldots x_{t-1}, \theta) = \text{交叉熵 Loss}$$

因为概率值在 0 到 1 之间，$\log$ 之后是负数，加负号翻正方便做最小化。

> **关键联系：** LLM 的训练损失（交叉熵）就是负对数似然。最小化 loss = 最大化似然 = 找到最可能生成训练数据的模型参数。

---

## 三、微积分

### 3.1 链式法则（Chain Rule）

如果 $y = f(g(x))$（复合函数），那么：

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

**简单例子：**

$$y = (3x + 1)^2$$

令 $g = 3x + 1$，则 $y = g^2$：

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = 2g \times 3 = 6(3x + 1)$$

**多层嵌套：**

$$y = f(g(h(x)))$$

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx} \quad \leftarrow \text{像链条一样串起来，所以叫"链式法则"}$$

**在神经网络中——每一层就是一个嵌套函数：**

$$x \xrightarrow{f_1} h_1 = \text{ReLU}(W_1 x + b_1) \xrightarrow{f_2} h_2 = \text{ReLU}(W_2 h_1 + b_2) \xrightarrow{f_3} y = \text{softmax}(W_3 h_2 + b_3) \xrightarrow{f_4} \text{Loss}$$

$$\frac{\partial \text{Loss}}{\partial W_1} = \frac{\partial \text{Loss}}{\partial y} \cdot \frac{\partial y}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}$$

链式法则：从输出一路乘回来。

---

### 3.2 梯度（Gradient）

导数的多维推广。对一个多变量函数，梯度是各个变量偏导数组成的向量。

$$f(x, y) = x^2 + 3xy + y^2$$

$$\nabla f = \left[\frac{\partial f}{\partial x},\; \frac{\partial f}{\partial y}\right] = [2x + 3y,\; 3x + 2y]$$

在点 $(1, 2)$ 处：

$$\nabla f(1, 2) = [2 \times 1 + 3 \times 2,\; 3 \times 1 + 2 \times 2] = [8,\; 7]$$

**梯度的物理含义：**

```
                     山顶（Loss 最大）
                    /    \
                  /   ↑    \         ↑ 梯度方向 = 函数增长最快的方向
                /     |      \       ↓ 负梯度方向 = 函数下降最快的方向
              /       ● 你在这   \
            /                     \
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                     山谷（Loss 最小）
```

**梯度下降：** 反复沿负梯度方向走

$$W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial \text{Loss}}{\partial W}$$

每一步都往 loss 下降最快的方向走一小步，重复几万次 → 走到山谷 → loss 最小 → 模型训练完成。

---

### 3.3 反向传播（Backpropagation）= 链式法则 + 梯度的实际应用

反向传播不是一个新数学概念，而是**用链式法则高效计算梯度的算法**。

**具体过程：**

**前向传播（从左到右）：**

$$x \xrightarrow{W_1} h_1 \xrightarrow{W_2} h_2 \xrightarrow{W_3} y \rightarrow \text{Loss} = 5.2$$

记住中间值 $h_1, h_2$（激活值，需要显存存储）。

**反向传播（从右到左）：**

$$\frac{\partial L}{\partial W_3} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W_3} \quad \leftarrow \text{先算最后一层}$$

$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial h_2} \cdot \frac{\partial h_2}{\partial W_2} \quad \leftarrow \text{往前传一层}$$

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1} \quad \leftarrow \text{传到第一层}$$

> $\frac{\partial L}{\partial W_3}$ 的意思是：其他所有层的参数都不动，只看 $W_3$ 里的参数变化对 Loss 的影响有多大。

**数值示例（2 层网络）：**

前向：

$$h = W_1 \times x = 2 \times 3 = 6$$

$$y = W_2 \times h = 0.5 \times 6 = 3$$

$$\text{Loss} = (y - \text{target})^2 = (3 - 1)^2 = 4$$

反向（链式法则）：

$$\frac{\partial L}{\partial y} = 2(y - 1) = 2(3 - 1) = 4$$

$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W_2} = 4 \times h = 4 \times 6 = 24$$

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial h} \cdot \frac{\partial h}{\partial W_1} = 4 \times W_2 \times x = 4 \times 0.5 \times 3 = 6$$

更新参数（学习率 $\eta = 0.01$）：

$$W_{2,\text{new}} = 0.5 - 0.01 \times 24 = 0.26$$

$$W_{1,\text{new}} = 2 - 0.01 \times 6 = 1.94$$

**对于矩阵：** 每个位置独立算，互不影响。

$$W_2 = \begin{bmatrix} 0.5 & 0.3 \\ 0.1 & -0.2 \end{bmatrix}, \quad \frac{\partial L}{\partial W_2} = \begin{bmatrix} 24 & -8 \\ 12 & 5 \end{bmatrix}$$

$$W_{2,\text{new}} = \begin{bmatrix} 0.5 - 0.01 \times 24 & 0.3 - 0.01 \times (-8) \\ 0.1 - 0.01 \times 12 & -0.2 - 0.01 \times 5 \end{bmatrix} = \begin{bmatrix} 0.26 & 0.38 \\ -0.02 & -0.25 \end{bmatrix}$$

在框架里就一行：`W₂ = W₂ - lr * grad`，底层并行算完所有元素。GPU 擅长的就是这个——几百万个元素同时算。

---

## 四、总结：数学概念在 LLM 训练中的位置

```
输入文本
  ↓
Tokenizer → token IDs
  ↓
Embedding 查表 → 向量                    ← 矩阵运算
  ↓
Attention(Q×Kᵀ/√d × V) → 上下文向量      ← 矩阵运算
  ↓
FFN(W₁, W₂) → 输出向量                   ← 矩阵运算
  ↓
Softmax → 概率分布                        ← 概率分布
  ↓
CrossEntropy Loss = -log P(正确token)     ← 最大似然估计
  ↓
反向传播: ∂Loss/∂每个参数                  ← 链式法则 + 梯度
  ↓
Adam 更新参数                              ← 梯度下降
  ↓
重复几万次 → 模型学会了
```

| 数学概念 | 在 LLM 中干什么 |
|---------|----------------|
| 矩阵运算 | 模型每一层的核心计算（Attention、FFN） |
| SVD | LoRA 的理论基础（低秩近似） |
| 概率分布 | 模型输出就是词表上的概率分布 |
| 贝叶斯 | 根据上文更新对下一个词的预测 |
| 最大似然估计 | 训练目标：找到最可能生成训练数据的参数 |
| 链式法则 | 反向传播的数学基础 |
| 梯度 | 告诉优化器参数该往哪个方向调整 |
