# 机器学习基础概念与算法

---

## 一、损失函数（Loss Function）

损失函数衡量模型预测值和真实值之间的"差距"。训练的目标就是让损失函数的值尽可能小。

### 1. MSE（均方误差）— 回归任务用

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{真实}} - y_{\text{预测}})^2$$

直觉：每个样本的预测误差取平方（让大误差惩罚更重），然后求平均。

```
真实值:  [3.0,  5.0,  7.0]
预测值:  [2.5,  5.5,  6.0]
误差:    [0.5, -0.5,  1.0]
平方:    [0.25, 0.25, 1.0]
MSE = (0.25 + 0.25 + 1.0) / 3 = 0.5
```

为什么用平方而不是绝对值？
- 平方可导，方便梯度下降优化
- 对大误差惩罚更重（误差 2 → 惩罚 4，误差 3 → 惩罚 9）

### 2. 交叉熵（Cross-Entropy）— 分类任务用

二分类交叉熵：

$$L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y \cdot \log(p) + (1-y) \cdot \log(1-p) \right]$$

其中 $y$ = 真实标签（0 或 1），$p$ = 模型预测为 1 的概率。

直觉拆解：
- 当 $y=1$（真实是正类）：$L = -\log(p)$，$p$ 越接近 1 损失越小
- 当 $y=0$（真实是负类）：$L = -\log(1-p)$，$p$ 越接近 0 损失越小

```
例子：真实标签 y=1

模型预测 p=0.9  → L = -log(0.9)  = 0.105  ← 很自信且正确，损失小
模型预测 p=0.5  → L = -log(0.5)  = 0.693  ← 不确定，损失中等
模型预测 p=0.1  → L = -log(0.1)  = 2.303  ← 很自信但错了，损失巨大！
```

多分类交叉熵（LLM 用的就是这个）：

$$L = -\sum_{i} y_i \cdot \log(p_i)$$

$y_i$ 是 one-hot 向量（只有正确类别那一位是 1），所以实际上就是：$L = -\log(p_{\text{正确类别}})$

这就是 note.md 里提到的 next token prediction 的损失函数。

---

## 二、梯度下降（Gradient Descent）

### 核心思想

把损失函数想象成一个山谷地形，参数就是你的位置。梯度下降就是"沿着最陡的下坡方向走一步"，反复走，直到走到谷底（损失最小）。

参数更新公式：

$$w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial L}{\partial w}$$

其中 $\eta$ = 学习率（步长），$\frac{\partial L}{\partial w}$ = 损失对参数的偏导数（梯度）。

### 为什么是减号？

梯度指向函数增大最快的方向，我们要最小化损失，所以反方向走 → 减号。

### 学习率的影响

```
lr 太大:   ↗ ↘ ↗ ↘ ↗    在谷底来回震荡，甚至发散
lr 合适:   ↘ ↘ ↘ → 收敛   稳步下降到谷底
lr 太小:   ↘...↘...↘...   能收敛但极慢，可能卡在局部最小值
```

### 三种变体

```
1. 批量梯度下降（Batch GD）
   每次用全部数据算梯度 → 准确但慢

2. 随机梯度下降（SGD）
   每次只用 1 个样本算梯度 → 快但噪声大，路径抖动

3. 小批量梯度下降（Mini-batch SGD）← 实际最常用
   每次用一小批数据（如 32/64/128 个样本）→ 兼顾速度和稳定性
```

### 一个具体例子：用梯度下降拟合 $y = wx + b$

```
假设真实关系：y = 3x + 1

初始参数：w=0, b=0
数据点：(1, 4), (2, 7), (3, 10)

第 1 步：
  预测: [0, 0, 0]    真实: [4, 7, 10]
  MSE = (16 + 49 + 100) / 3 = 55
  ∂L/∂w = (2/3) × [1×(0-4) + 2×(0-7) + 3×(0-10)] = (2/3)×(-48) = -32
  ∂L/∂b = (2/3) × [(0-4) + (0-7) + (0-10)] = (2/3)×(-21) = -14

  lr = 0.01
  w = 0 - 0.01×(-32) = 0.32
  b = 0 - 0.01×(-14) = 0.14

第 2 步：
  预测: [0.46, 0.78, 1.10]   ← 比上一步更接近真实值了
  ...

经过几百步迭代，w → 3, b → 1，拟合成功。
```

---

## 三、过拟合与欠拟合

### 欠拟合（Underfitting）

模型太简单，连训练数据的规律都没学到。

```
真实关系：y = x² + 2x + 1（抛物线）

用 y = wx + b（直线）去拟合 → 怎么调参都拟合不好

训练集误差：大
测试集误差：大
```

原因：模型容量不够（用直线拟合曲线）

### 过拟合（Overfitting）

模型太复杂，把训练数据的噪声也"记住"了，在新数据上表现差。

```
5 个数据点，大致呈直线趋势，但有噪声

欠拟合（直线）:     过拟合（高次多项式）:     刚好（带正则化）:

  .  /  .             . ╱╲ .                    .  / .
 . /                  .╱  ╲                    . /
  / .  .              ╱ .  ╲.                   / .  .

训练误差：中等       训练误差：≈0              训练误差：较小
测试误差：中等       测试误差：巨大！           测试误差：较小
```

### 判断方法

```
                    训练误差    测试误差
欠拟合              高          高
过拟合              低          高        ← 两者差距大是关键信号
刚好                低          低（接近）
```

### 解决过拟合的方法

1. 增加数据量 — 最直接有效
2. 正则化 — 限制模型复杂度（下一节详讲）
3. Dropout — 训练时随机关闭部分神经元
4. 早停（Early Stopping）— 测试误差开始上升时停止训练
5. 数据增强 — 对现有数据做变换生成更多样本

---

## 四、正则化（Regularization）

核心思想：在损失函数里加一个"惩罚项"，让模型参数不要太大，从而限制模型复杂度，防止过拟合。

### L2 正则化（Ridge / 权重衰减）

$$L_{\text{total}} = L_{\text{原始}} + \lambda \sum_{i} w_i^2$$

$\lambda$ = 正则化强度（超参数）

效果：让所有参数趋向于较小的值，但不会变成 0。

直觉：参数越大，模型越"激进"（对输入的微小变化反应剧烈）。L2 惩罚大参数，让模型更"平滑"。

```
无正则化：  w = [10.5, -8.3, 0.01, 15.2]   ← 参数很极端
L2 正则化： w = [2.1,  -1.5, 0.3,  1.8]    ← 参数更均匀、更小
```

### L1 正则化（Lasso）

$$L_{\text{total}} = L_{\text{原始}} + \lambda \sum_{i} |w_i|$$

效果：让部分参数直接变成 0 → 自动特征选择。

```
无正则化：  w = [10.5, -8.3, 0.01, 15.2]
L1 正则化： w = [3.2,   0,   0,    2.1]    ← 不重要的特征权重直接归零
```

### L1 vs L2 对比

|              | L1（Lasso） | L2（Ridge） |
|:-------------|:-----------|:-----------|
| 惩罚项 | $\sum \|w_i\|$ | $\sum w_i^2$ |
| 效果 | 产生稀疏解（很多 0） | 参数整体缩小 |
| 适用场景 | 特征选择，高维稀疏数据 | 一般防过拟合 |
| 几何直觉 | 菱形约束（角上容易为 0） | 圆形约束（均匀缩小） |

### 为什么 L1 能产生 0 而 L2 不能？

**梯度对比：**

- L1 的梯度：$\frac{\partial |w|}{\partial w} = \pm 1$（常数），不管 $w$ 多小，推力都一样大 → 能推到 0
- L2 的梯度：$\frac{\partial w^2}{\partial w} = 2w$，$w$ 越小推力越小 → 无限趋近 0 但到不了

**L1 在 $w=0$ 处不可导怎么办？** 实际用**次梯度（subgradient）**：

$$\text{sign}(w) = \begin{cases} +1 & w > 0 \\ 0 & w = 0 \\ -1 & w < 0 \end{cases}$$

$w = 0$ 时梯度定义为 0，权重就停在 0 不动了。

**L1 推到 0 的过程：**

假设 $w = 0.05$，$\lambda = 0.1$，$\eta = 0.1$，原始梯度 $= 0$

$$w_{\text{new}} = 0.05 - 0.1 \times (0 + 0.1 \times \text{sign}(0.05)) = 0.05 - 0.01 = 0.04$$

```
继续几轮：0.04 → 0.03 → 0.02 → 0.01 → 0.00  ← 到了 0
到 0 之后：sign(0) = 0，推力消失，w 就停住了
```

**对比 L2 为什么推不到 0：**

同样 $w = 0.05$，$\lambda = 0.1$，$\eta = 0.1$，原始梯度 $= 0$

$$w_{\text{new}} = 0.05 - 0.1 \times (2 \times 0.1 \times 0.05) = 0.05 - 0.001 = 0.049$$

```
继续：0.049 → 0.048 → ... → 0.0001 → 0.00009 → ...
越小推力越弱，永远在趋近 0，但到不了
```

**直觉类比：**

| | L1 | L2 |
|:--|:---|:---|
| 推力 | 恒定（不管 $w$ 多小都是 $\lambda$） | 和 $w$ 成正比（越小越弱） |
| 类比 | 恒定摩擦力（直接停住） | 弹簧（越近拉力越弱） |
| 结果 | 能精确等于 0 | 无限趋近 0 但到不了 |

---

## 五、线性回归（Linear Regression）

### 模型

$$y = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$$

矩阵形式：$y = Xw + b$

最简单的机器学习模型：输入特征的加权求和。

### 训练目标

最小化 MSE：

$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### 两种求解方式

**方式一：正规方程（解析解）**

$$w = (X^T X)^{-1} X^T y$$

一步算出最优解，不需要迭代。但当特征很多或数据量很大时，矩阵求逆计算量太大。

**方式二：梯度下降（迭代求解）**

$$\frac{\partial L}{\partial w} = -\frac{2}{n} X^T(y - Xw)$$

$$\frac{\partial L}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)$$

$$w = w - \eta \cdot \frac{\partial L}{\partial w}, \quad b = b - \eta \cdot \frac{\partial L}{\partial b}$$

### 梯度推导过程

#### 从损失函数开始

线性回归的目标：最小化 MSE

$$L = \frac{1}{n} \sum(y_i - \hat{y}_i)^2, \quad \hat{y}_i = w \cdot x_i + b$$

#### 正规方程的推导

写成矩阵形式（$X$ 已包含偏置列）：

$$\hat{y} = Xw, \quad L = \frac{1}{n}(y - Xw)^T(y - Xw)$$

展开：

$$L = \frac{1}{n}(y^Ty - y^TXw - w^TX^Ty + w^TX^TXw)$$

对 $w$ 求导，令导数 $= 0$：

$$\frac{\partial L}{\partial w} = \frac{1}{n}(-2X^Ty + 2X^TXw) = 0$$

$$\Rightarrow X^TXw = X^Ty \Rightarrow w = (X^TX)^{-1}X^Ty$$

> **用到的矩阵求导规则：**
> - $\frac{\partial(a^Tw)}{\partial w} = a$
> - $\frac{\partial(w^TAw)}{\partial w} = 2Aw$（$A$ 对称时）

#### 梯度下降的推导

不解方程，而是对 $w$ 和 $b$ 分别求偏导，然后沿梯度方向更新。

**对 $w$ 求导：** 用链式法则，令 $u_i = y_i - w \cdot x_i - b$

$$\frac{\partial L}{\partial w} = \frac{1}{n} \sum 2 \cdot u_i \cdot \frac{\partial u_i}{\partial w} = \frac{1}{n} \sum 2(y_i - w \cdot x_i - b)(-x_i) = -\frac{2}{n} \sum x_i(y_i - \hat{y}_i)$$

写成矩阵形式：$= -\frac{2}{n} X^T(y - Xw)$

**对 $b$ 求导：**

$$\frac{\partial L}{\partial b} = \frac{1}{n} \sum 2(y_i - w \cdot x_i - b)(-1) = -\frac{2}{n} \sum(y_i - \hat{y}_i)$$

**沿梯度反方向更新：**

$$w = w - \eta \cdot \frac{\partial L}{\partial w}, \quad b = b - \eta \cdot \frac{\partial L}{\partial b}$$

#### 具体数字示例

```
数据：x = [1, 2, 3],  y = [2, 4, 6]
初始：w = 0,  b = 0,  η = 0.1

第1轮：
  ŷ = [0, 0, 0]
  误差 = y - ŷ = [2, 4, 6]

  ∂L/∂w = -(2/3)(1·2 + 2·4 + 3·6) = -(2/3)(28) = -18.67
  ∂L/∂b = -(2/3)(2 + 4 + 6)       = -(2/3)(12) = -8

  w = 0 - 0.1 × (-18.67) = 1.867
  b = 0 - 0.1 × (-8)     = 0.8

第2轮：
  ŷ = [2.667, 4.534, 6.401]  ← 比上一步更接近真实值了
  误差 = [-0.667, -0.534, -0.401]
  梯度变小了 → w 和 b 微调 ...

最终收敛：w ≈ 2, b ≈ 0  （y = 2x，完美拟合）
```

#### 两种方式的关系

| | 正规方程 | 梯度下降 |
|:--|:--------|:--------|
| 做法 | 直接解 $\frac{\partial L}{\partial w} = 0$ | 沿 $\frac{\partial L}{\partial w}$ 方向迭代 |
| 速度 | 特征少时一步到位 | 需要多轮迭代 |
| 瓶颈 | 矩阵求逆 $O(n^3)$ | 需要调学习率 |
| 适用 | 特征数 $< \sim 10000$ | 特征多、数据量大 |


### 手动实现思路（numpy）

```
初始化 w = 随机小值, b = 0
循环 1000 次:
    ŷ = X @ w + b                    # 前向传播
    loss = mean((y - ŷ)²)            # 计算损失
    dw = -(2/n) * X.T @ (y - ŷ)     # 计算梯度
    db = -(2/n) * sum(y - ŷ)
    w = w - lr * dw                  # 更新参数
    b = b - lr * db
```

### 完整代码实现

```python
import numpy as np

# ============ 线性回归手动实现 ============

# 1. 造数据：y = 3x₁ + 5x₂ + 7 + 噪声
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)                    # 100个样本，2个特征
y = 3 * X[:, 0] + 5 * X[:, 1] + 7 + np.random.randn(n_samples) * 0.5

# 2. 初始化参数
w = np.zeros(2)   # 权重
b = 0.0           # 偏置
lr = 0.05         # 学习率
epochs = 500

# 3. 梯度下降训练
for epoch in range(epochs):
    # 前向传播：预测
    y_pred = X @ w + b

    # 计算损失（MSE）
    loss = np.mean((y - y_pred) ** 2)

    # 计算梯度
    dw = -(2 / n_samples) * X.T @ (y - y_pred)    # shape: (2,)
    db = -(2 / n_samples) * np.sum(y - y_pred)     # 标量

    # 更新参数
    w = w - lr * dw
    b = b - lr * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | w: [{w[0]:.3f}, {w[1]:.3f}] | b: {b:.3f}")

print(f"\n最终结果: w = [{w[0]:.3f}, {w[1]:.3f}], b = {b:.3f}")
print(f"真实参数: w = [3.000, 5.000], b = 7.000")
```

运行输出示例：
```
Epoch   0 | Loss: 72.5431 | w: [0.614, 0.978] | b: 1.378
Epoch 100 | Loss:  0.2753 | w: [2.975, 4.978] | b: 6.988
Epoch 200 | Loss:  0.2413 | w: [2.993, 4.993] | b: 6.998
Epoch 300 | Loss:  0.2407 | w: [2.996, 4.996] | b: 7.000
Epoch 400 | Loss:  0.2406 | w: [2.997, 4.997] | b: 7.000

最终结果: w = [2.997, 4.997], b = 7.000
真实参数: w = [3.000, 5.000], b = 7.000
```

### 加上正则化 → Ridge 回归

$$L = \text{MSE} + \lambda \sum_{i} w_i^2$$

梯度变化：

$$\frac{\partial L}{\partial w} = -\frac{2}{n} X^T(y - \hat{y}) + 2\lambda w \quad \leftarrow \text{多了 } 2\lambda w \text{ 这一项}$$

$\lambda$ 越大，参数被压得越小，模型越简单。

---

## 六、逻辑回归（Logistic Regression）

虽然名字里有"回归"，但它是一个分类算法。

### 核心思想

线性回归的输出范围是 $(-\infty, +\infty)$，不适合表示概率。
解决方案：在线性回归外面套一个 Sigmoid 函数，把输出压缩到 $(0, 1)$。

$$z = wx + b \quad \leftarrow \text{线性部分}$$

$$p = \sigma(z) = \frac{1}{1 + e^{-z}} \quad \leftarrow \text{Sigmoid 压缩到 } (0,1)$$

Sigmoid 函数图像：

```
  1 |            ___________
    |          /
    |        /
0.5 |------/------------------
    |    /
    |  /
  0 |/___________
    -6  -3   0   3   6
```

### 决策边界

$p \geq 0.5$ → 预测为正类（1），$p < 0.5$ → 预测为负类（0）

等价于：$z = wx + b \geq 0$ → 正类

所以逻辑回归的决策边界是一条直线（或超平面）。

### 损失函数：二分类交叉熵

$$L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y \cdot \log(p) + (1-y) \cdot \log(1-p) \right]$$

为什么不用 MSE？因为 Sigmoid + MSE 的损失曲面不是凸的，梯度下降容易卡在局部最小值。交叉熵 + Sigmoid 是凸的，保证收敛到全局最优。

### 交叉熵的推导来源

交叉熵来自**最大似然估计（MLE）**：

1. 假设样本服从伯努利分布：$P(y|\hat{y}) = \hat{y}^y(1-\hat{y})^{1-y}$
2. 对所有样本取似然函数：$\prod_i \hat{y}_i^{y_i}(1-\hat{y}_i)^{1-y_i}$
3. 取对数得到对数似然：$\sum_i [y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$
4. 加负号（最大化似然 → 最小化损失），就得到了交叉熵公式

本质上，**最小化交叉熵 = 最大化模型对正确标签的预测概率**。

### 从二分类到多分类

在二分类交叉熵中，$y$ 只能是 0 或 1，因为它对应的是二分类问题（是/否、正/负）。

如果标签不止两类，就要用**多分类交叉熵（Categorical Cross-Entropy）**：

$$L = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

其中：
- $C$：类别总数
- $y_c$：one-hot 编码（正确类别为 1，其余为 0）
- $\hat{y}_c$：模型对第 $c$ 类的预测概率（通常经过 softmax）

其实二分类交叉熵就是 $C=2$ 时的特例。

### 梯度推导

$$\frac{\partial L}{\partial w} = \frac{1}{n} X^T(p - y)$$

$$\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n}(p_i - y_i)$$

形式上和线性回归的梯度几乎一样！区别只是 $\hat{y}$ 换成了 $p$（经过 Sigmoid 的输出）。

### 手动实现思路

```
初始化 w, b = 0
循环 1000 次:
    z = X @ w + b
    p = 1 / (1 + exp(-z))              # Sigmoid
    loss = -mean(y*log(p) + (1-y)*log(1-p))
    dw = (1/n) * X.T @ (p - y)
    db = (1/n) * sum(p - y)
    w = w - lr * dw
    b = b - lr * db
```

### 完整代码实现

```python
import numpy as np

# ============ 逻辑回归手动实现 ============

# 1. 造二分类数据
np.random.seed(42)
n_samples = 200

# 正类：中心在 (2, 2)
X_pos = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
# 负类：中心在 (-2, -2)
X_neg = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])

X = np.vstack([X_pos, X_neg])                  # (200, 2)
y = np.array([1] * 100 + [0] * 100)            # 标签

# 打乱顺序
shuffle_idx = np.random.permutation(n_samples)
X, y = X[shuffle_idx], y[shuffle_idx]

# 2. Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))   # clip 防溢出

# 3. 初始化参数
w = np.zeros(2)
b = 0.0
lr = 0.1
epochs = 1000

# 4. 梯度下降训练
for epoch in range(epochs):
    # 前向传播
    z = X @ w + b
    p = sigmoid(z)                              # 预测概率

    # 计算交叉熵损失
    eps = 1e-8                                  # 防止 log(0)
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    # 计算梯度（和线性回归形式几乎一样，只是 y_pred 换成了 p）
    dw = (1 / n_samples) * X.T @ (p - y)
    db = (1 / n_samples) * np.sum(p - y)

    # 更新参数
    w = w - lr * dw
    b = b - lr * db

    if epoch % 200 == 0:
        # 计算准确率
        predictions = (p >= 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")

# 5. 最终结果
z = X @ w + b
p = sigmoid(z)
predictions = (p >= 0.5).astype(int)
accuracy = np.mean(predictions == y)
print(f"\n最终准确率: {accuracy:.2%}")
print(f"学到的参数: w = [{w[0]:.3f}, {w[1]:.3f}], b = {b:.3f}")
print(f"决策边界: {w[0]:.3f}*x₁ + {w[1]:.3f}*x₂ + {b:.3f} = 0")
```

运行输出示例：
```
Epoch    0 | Loss: 0.6931 | Accuracy: 50.00%
Epoch  200 | Loss: 0.1198 | Accuracy: 98.50%
Epoch  400 | Loss: 0.0812 | Accuracy: 99.00%
Epoch  600 | Loss: 0.0647 | Accuracy: 99.00%
Epoch  800 | Loss: 0.0551 | Accuracy: 99.50%

最终准确率: 99.50%
学到的参数: w = [1.523, 1.487], b = -0.012
决策边界: 1.523*x₁ + 1.487*x₂ + -0.012 = 0
```

### 多分类扩展 → Softmax 回归

把 Sigmoid 换成 Softmax，交叉熵变成多分类版本。这是二分类 → 多分类的自然推广：

**从二分类说起：** Sigmoid 用于二分类（0 或 1），它把一个数压缩到 $(0, 1)$，表示"是正类的概率"：

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

对应的损失函数是二分类交叉熵：

$$L = -[y \log(\hat{p}) + (1-y)\log(1-\hat{p})]$$

#### 二分类交叉熵的 MLE 推导

**第一步：建立概率模型**

逻辑回归假设 $\hat{p} = P(y=1 | x) = \sigma(z)$，那么：
- $y=1$ 的概率是 $\hat{p}$
- $y=0$ 的概率是 $1-\hat{p}$

用一个式子把两种情况合并：

$$P(y|x) = \hat{p}^{y} \cdot (1-\hat{p})^{1-y}$$

验证：当 $y=1$ 时得到 $\hat{p}$，当 $y=0$ 时得到 $1-\hat{p}$。✓

**第二步：最大似然——希望概率越大越好**

有 $n$ 个独立样本，联合似然是：

$$L(\theta) = \prod_{i=1}^{n} \hat{p}_i^{y_i} \cdot (1-\hat{p}_i)^{1-y_i}$$

我们想最大化这个式子（让模型尽量"猜对"）。

**第三步：取对数（log-likelihood）**

连乘很难优化，取 $\log$ 变连加：

$$\log L(\theta) = \sum_{i=1}^{n} \left[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \right]$$

**第四步：最大化 → 最小化（加负号）**

机器学习习惯最小化损失函数，所以加个负号：

$$\boxed{L = -\left[ y \log \hat{p} + (1-y) \log(1-\hat{p}) \right]}$$

这就是二分类交叉熵！

#### 当 $K=2$ 时，Softmax 等价于 Sigmoid

两个类别，分别有得分 $z_1$ 和 $z_2$，Softmax 给出：

$$P(y=1) = \frac{e^{z_1}}{e^{z_1} + e^{z_2}}, \quad P(y=0) = \frac{e^{z_2}}{e^{z_1} + e^{z_2}}$$

化简 $P(y=1)$，分子分母同除以 $e^{z_2}$：

$$P(y=1) = \frac{e^{z_1}/e^{z_2}}{(e^{z_1}+e^{z_2})/e^{z_2}} = \frac{e^{z_1-z_2}}{1 + e^{z_1-z_2}}$$

令 $z = z_1 - z_2$（两个得分的差值），就得到：

$$P(y=1) = \frac{e^z}{1+e^z} = \frac{1}{1+e^{-z}}$$

这正好就是 Sigmoid！✓


---

## 七、决策树（Decision Tree）

### 核心思想

通过一系列"if-else"规则对数据进行递归划分，每次选择最优特征和阈值进行分裂，直到满足停止条件。

### 关键概念

- **信息熵（ID3）：** $H(D) = -\sum p_i \cdot \log_2(p_i)$，衡量数据集的混乱程度
- **信息增益：** $IG = H(D) - H(D|\text{特征A})$，选择使信息增益最大的特征
- **基尼系数（CART）：** $Gini = 1 - \sum p_i^2$，越小越纯净

信息熵在问：“我猜下一个样本的类别，平均会有多惊讶？”（惊讶越大，越混乱）
信息增益在问：“按这个特征分类后，我的惊讶度降低了多少？”（降低越多，特征越好）
基尼系数在问：“我随便抓两个样本，它们不是一伙的概率有多大？”（概率越大，越混乱）

### 构建过程

```
1. 计算所有特征的信息增益（或基尼系数）
2. 选择增益最大的特征作为分裂节点
3. 递归对子集重复上述步骤
4. 直到：叶节点样本数<阈值 / 树达到最大深度 / 信息增益<阈值
```

**Python 实现：**
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, criterion='gini')
model.fit(X_train, y_train)
```

**优缺点：**
- ✅ 可解释性极强，无需特征归一化，可处理非线性
- ❌ 容易过拟合，对噪声敏感

---

## 八、随机森林（Random Forest）

### 核心思想

集成学习方法，通过 **Bagging（自举聚合）** 训练多棵决策树，最终用投票（分类）或平均（回归）来综合结果。

### 关键步骤

```
1. 从训练集中有放回地随机抽取 n 个样本（Bootstrap）
2. 在每个节点，随机选取 √p 个特征（p 为总特征数）进行最优分裂
3. 重复构建 T 棵树（通常 100~500 棵）
4. 预测时：分类取投票多数，回归取均值
```

**特征重要性：** 通过统计每个特征在所有树中带来的信息增益平均值来评估。

**Python 实现：**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_features='sqrt')
model.fit(X_train, y_train)
print(model.feature_importances_)  # 特征重要性
```

### 完整代码示例：乳腺癌分类

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 第一步：准备数据
data = load_breast_cancer()
X = data.data      # 病人的特征（肿瘤半径、纹理等 30 个指标）
y = data.target    # 确诊结果（0=恶性，1=良性）

# 80% 训练集，20% 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 第二步：建立随机森林模型（100 棵决策树）
forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 第三步：训练
forest_model.fit(X_train, y_train)

# 第四步：预测
predictions = forest_model.predict(X_test)

# 第五步：评估
accuracy = accuracy_score(y_test, predictions)
print(f”随机森林的准确率: {accuracy * 100:.2f}%”)
```

**`fit()` 背后发生了什么？**

1. **Bagging（样本随机）：** 循环 100 次，每次从训练集中有放回地随机抽取样本，分配给一棵树。
2. **特征随机：** 每棵树在每个节点，从 30 个特征中只随机选取约 $\sqrt{30} \approx 5.4$ 个特征。
3. **寻找最纯净的划分：** 在选出的特征中，计算信息增益或基尼系数，选最优特征和阈值分裂。
4. **投票：** 预测时，100 棵树同时给出结论，少数服从多数。
**优缺点：**
- ✅ 抗过拟合能力强，可处理高维数据，能评估特征重要性
- ❌ 模型较大，可解释性弱于单棵决策树，训练较慢

---

## 九、朴素贝叶斯（Naive Bayes）

### 核心思想

基于贝叶斯定理，并假设所有特征之间**条件独立**（这一"朴素"假设在实际中很少成立，但效果往往不错）。

### 贝叶斯定理

$$P(\text{类别}C \mid \text{特征}X) = \frac{P(\text{特征}X \mid \text{类别}C) \times P(\text{类别}C)}{P(\text{特征}X)} \propto \text{似然度} \times \text{先验概率}$$

### 三种常用变体

| 类型 | 适用场景 | 特征分布假设 |
|------|---------|-------------|
| 高斯NB | 连续特征 | 正态分布 |
| 多项式NB | 文本分类（词频） | 多项分布 |
| 伯努利NB | 二值特征 | 伯努利分布 |

**Python 实现：**
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB
model = GaussianNB()
model.fit(X_train, y_train)
```

**优缺点：**
- ✅ 极快、数据量小时表现好，天然支持增量学习
- ❌ 特征独立性假设过强，概率估计可能不准

---

## 十、支持向量机（SVM）

### 核心思想

找一个超平面把两类数据分开，而且要让离超平面最近的点（支持向量）到超平面的距离最大化。

```
两类数据点：○ 和 ×

                    间隔（margin）
                   ←───────→
  ○  ○             |         |
    ○    ○         |         |        ×    ×
  ○   ○            |         |     ×    ×
    ○              |         |        ×
                   |         |
              支持向量    支持向量
              （离边界最近的点）
```

### 数学表达

超平面：$w \cdot x + b = 0$

分类规则：

$$w \cdot x + b \geq +1 \rightarrow \text{正类}$$
$$w \cdot x + b \leq -1 \rightarrow \text{负类}$$

间隔宽度 $= \frac{2}{\|w\|}$

目标：最大化间隔 = 最小化 $\|w\|^2$

### 优化问题

$$\min \frac{1}{2} \|w\|^2 \quad \text{s.t.} \quad y_i (w \cdot x_i + b) \geq 1, \;\forall i$$

这是一个有约束的凸优化问题。

### 软间隔 SVM（实际使用的版本）

现实数据往往不能完美线性分割，所以引入松弛变量 $\xi$，允许部分样本违反间隔约束：

$$\min \frac{1}{2} \|w\|^2 + C \sum_{i} \xi_i$$

$$\text{s.t.} \quad y_i(w \cdot x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

$C$ 是惩罚参数：
- $C$ 大 → 不容忍误分类 → 间隔小，可能过拟合
- $C$ 小 → 容忍一些误分类 → 间隔大，更泛化

### Hinge Loss（合页损失）

SVM 的损失函数可以写成：

$$L = \frac{1}{n} \sum_{i=1}^{n} \max\left(0,\; 1 - y_i(w \cdot x_i + b)\right) + \lambda \|w\|^2$$

$$\underbrace{\qquad\qquad\qquad\qquad\qquad}_{\text{hinge loss}} \quad + \quad \underbrace{\qquad}_{\text{正则化}}$$

直觉：
- $y_i(w \cdot x_i + b) \geq 1$ → 分类正确且在间隔外 → 损失 $= 0$
- $y_i(w \cdot x_i + b) < 1$ → 在间隔内或分类错误 → 损失 $= 1 - y_i f(x_i)$

### 手动实现思路（用梯度下降解 Hinge Loss 版本）

```
标签必须是 +1 和 -1（不是 0 和 1）

初始化 w, b = 0
循环 1000 次:
    对每个样本 (x_i, y_i):
        if y_i × (w·x_i + b) ≥ 1:
            # 分类正确且在间隔外，只有正则化梯度
            dw = 2λw
            db = 0
        else:
            # 在间隔内或分类错误
            dw = 2λw - y_i × x_i
            db = -y_i
        w = w - lr * dw
        b = b - lr * db
```

### 完整代码实现

```python
import numpy as np

# ============ SVM 手动实现（Hinge Loss + 梯度下降）============

# 1. 造二分类数据（注意 SVM 标签是 +1/-1，不是 0/1）
np.random.seed(42)
n_samples = 200

X_pos = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
X_neg = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])

X = np.vstack([X_pos, X_neg])
y = np.array([1] * 100 + [-1] * 100)           # SVM 用 +1/-1

shuffle_idx = np.random.permutation(n_samples)
X, y = X[shuffle_idx], y[shuffle_idx]

# 2. 初始化参数
w = np.zeros(2)
b = 0.0
lr = 0.001
lambda_reg = 0.01     # 正则化强度
epochs = 1000

# 3. 梯度下降训练
for epoch in range(epochs):
    total_loss = 0

    for i in range(n_samples):
        # 计算 margin
        margin = y[i] * (X[i] @ w + b)

        if margin >= 1:
            # 分类正确且在间隔外 → 只有正则化梯度
            dw = 2 * lambda_reg * w
            db = 0
        else:
            # 在间隔内或分类错误 → hinge loss 梯度 + 正则化
            dw = 2 * lambda_reg * w - y[i] * X[i]
            db = -y[i]
            total_loss += 1 - margin

        w = w - lr * dw
        b = b - lr * db

    # 总损失 = hinge loss + 正则化
    total_loss = total_loss / n_samples + lambda_reg * np.sum(w ** 2)

    if epoch % 200 == 0:
        predictions = np.sign(X @ w + b)
        accuracy = np.mean(predictions == y)
        print(f"Epoch {epoch:4d} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.2%}")

# 4. 最终结果
predictions = np.sign(X @ w + b)
accuracy = np.mean(predictions == y)
print(f"\n最终准确率: {accuracy:.2%}")
print(f"学到的参数: w = [{w[0]:.3f}, {w[1]:.3f}], b = {b:.3f}")

# 5. 找出支持向量（离决策边界最近的点）
margins = y * (X @ w + b)
support_vectors_idx = np.where((margins > 0.9) & (margins < 1.1))[0]
print(f"支持向量数量: {len(support_vectors_idx)}")
```

运行输出示例：
```
Epoch    0 | Loss: 1.0025 | Accuracy: 50.50%
Epoch  200 | Loss: 0.0312 | Accuracy: 99.50%
Epoch  400 | Loss: 0.0198 | Accuracy: 100.00%
Epoch  600 | Loss: 0.0167 | Accuracy: 100.00%
Epoch  800 | Loss: 0.0152 | Accuracy: 100.00%

最终准确率: 100.00%
学到的参数: w = [0.892, 0.871], b = 0.015
支持向量数量: 3
```

### 核技巧（Kernel Trick）— 处理非线性

当数据线性不可分时，SVM 可以用核函数把数据映射到高维空间，在高维空间里线性可分：

```
原始空间（2D，线性不可分）:        映射到高维后（3D，线性可分）:

    ×  ×                                    ×  ×
  × ○○○ ×                              ×          ×
  × ○○○ ×          ──映射──→         ○○○ ← 被"抬起来"了
    ×  ×                              ○○○
                                    ×          ×
                                        ×  ×
```

常用核函数：

| 核函数 | 公式 | 说明 |
|--------|------|------|
| 线性核 | $K(x,y) = x \cdot y$ | 就是普通 SVM |
| 多项式核 | $K(x,y) = (x \cdot y + 1)^d$ | |
| RBF 核 | $K(x,y) = e^{-\gamma \|x-y\|^2}$ | 最常用，能处理复杂边界 |

---

## 十一、卷积神经网络（CNN）

### 核心思想

专为处理**网格状数据**（如图像）设计，通过局部感受野、权值共享、空间下采样三大思想大幅减少参数量。

### 核心层结构

```
输入图像 → [卷积层 → 激活(ReLU)] × N → 池化层 → 全连接层 → 输出
```

### 卷积操作

$$\text{输出大小} = \frac{\text{输入尺寸} - \text{卷积核尺寸} + 2 \times \text{Padding}}{\text{Stride}} + 1$$

例：输入 $32 \times 32$，卷积核 $3 \times 3$，$\text{Padding}=1$，$\text{Stride}=1$ → 输出 $= \frac{32 - 3 + 2}{1} + 1 = 32 \times 32$（尺寸不变）

### 经典架构演进

| 架构 | 年份 | 特点 |
|------|------|------|
| LeNet-5 | 1998 | 开创性 CNN |
| AlexNet | 2012 | 深层+ReLU+Dropout |
| VGG16 | 2014 | 全用3×3卷积堆叠 |
| ResNet | 2015 | 残差连接，解决梯度消失 |
| EfficientNet | 2019 | 复合缩放策略 |

### PyTorch 实现示例

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

---

## 十二、循环神经网络（RNN）

### 核心思想

处理**序列数据**（文本、时序），通过隐藏状态将前一时刻的信息传递到下一时刻。

### 基本公式

$$h_t = \tanh(W_h \cdot h_{t-1} + W_x \cdot x_t + b)$$

$$y_t = W_y \cdot h_t$$

### 梯度消失问题

随着序列变长，梯度在反向传播中会指数级衰减，导致难以学习长距离依赖。

### 改进方案

**LSTM（长短期记忆网络）：** 引入三个门控机制

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{遗忘门：决定遗忘多少旧信息}$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{输入门：决定写入多少新信息}$$

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{输出门：决定输出多少信息}$$

$$C_t = f_t \odot C_{t-1} + i_t \odot \tanh(W_C \cdot [h_{t-1}, x_t]) \quad \text{细胞态}$$

**GRU（门控循环单元）：** LSTM的简化版，只有重置门和更新门，参数更少。

### PyTorch 实现

```python
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 取最后时间步
```

---

## 十三、自编码器（AutoEncoder）

### 核心思想

通过**编码器**将输入压缩为低维潜在表示（瓶颈层），再通过**解码器**重建原始输入，迫使模型学习数据的核心特征。

### 结构

$$x \xrightarrow{\text{编码器}} z \text{（低维）} \xrightarrow{\text{解码器}} \hat{x}$$

目标：最小化重建误差 $\|x - \hat{x}\|^2$

### 主要变体

| 变体 | 特点 | 应用 |
|------|------|------|
| 降噪AE (DAE) | 输入加噪，重建原始 | 去噪、鲁棒特征 |
| 稀疏AE (SAE) | 对隐层加稀疏约束 | 特征提取 |
| 变分AE (VAE) | 潜在空间建模为概率分布 | 图像生成 |
| 收缩AE (CAE) | 对雅可比矩阵加正则 | 流形学习 |

### PyTorch 实现（VAE）

```python
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 400)
        self.mu_layer = nn.Linear(400, 20)
        self.logvar_layer = nn.Linear(400, 20)
        self.decoder = nn.Sequential(
            nn.Linear(20, 400), nn.ReLU(),
            nn.Linear(400, 784), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)  # 重参数化技巧

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        mu, logvar = self.mu_layer(h), self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
```

---

## 十四、深度玻尔兹曼机（DBM）

### 核心思想

一种**生成式概率模型**，由多层随机二值神经元组成，通过无监督方式学习数据的深层概率分布。

### 结构特点

```
可见层 (v) ↔ 隐藏层1 (h¹) ↔ 隐藏层2 (h²) ↔ ...
（层内无连接，层间无向全连接）
```

### 能量函数

$$E(v,h) = -v^T W^{(1)} h^{(1)} - h^{(1)T} W^{(2)} h^{(2)} - b^T v - c^T h$$

**训练方法：** 对比散度（Contrastive Divergence, CD）算法近似求解最大似然。

**与 RBM 的关系：** RBM（受限玻尔兹曼机）是 DBM 的基本构建单元（只有一个隐藏层），多个 RBM 逐层堆叠可预训练 DBM。

**应用场景：** 图像生成、特征无监督预训练、推荐系统协同过滤。

---

## 十五、学习范式总览

### 有监督学习（Supervised Learning）

训练数据包含输入 $X$ 和对应标签 $y$，模型学习 $f: X \rightarrow y$ 的映射关系。

```
标注数据 → 特征工程 → 模型选择 → 训练(最小化损失) → 验证/调参 → 测试评估
```

| 任务类型 | 代表算法 |
|---------|---------|
| 二分类 | LR、SVM、决策树 |
| 多分类 | 随机森林、神经网络、softmax |
| 回归 | 线性回归、SVR、XGBoost |
| 序列标注 | CRF、BiLSTM-CRF |

**关键问题：**
- **过拟合：** 正则化(L1/L2)、Dropout、数据增强、早停
- **欠拟合：** 增加模型复杂度、增加特征
- **评估指标：** 准确率、精确率、召回率、F1、AUC-ROC

### 无监督学习（Unsupervised Learning）

训练数据**只有输入 $X$，没有标签**，模型自行发现数据的内在结构。

**① 聚类：**
```python
# K-Means
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
labels = km.fit_predict(X)

# DBSCAN（密度聚类，可发现任意形状的簇）
from sklearn.cluster import DBSCAN
labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
```

**② 降维：**
```python
# PCA（线性降维，保留最大方差方向）
from sklearn.decomposition import PCA
X_reduced = PCA(n_components=2).fit_transform(X)

# t-SNE（非线性降维，专用于可视化）
from sklearn.manifold import TSNE
X_2d = TSNE(n_components=2).fit_transform(X)
```

**③ 密度估计/生成：**
- 高斯混合模型（GMM）：用多个高斯分布的加权和拟合数据分布
- 自编码器、VAE、GAN：学习数据流形，生成新样本

### 强化学习（Reinforcement Learning）

智能体（Agent）在环境（Environment）中通过**试错**，学习最大化累积奖励的策略（Policy）。

```
Agent（智能体）
  ↓ 动作 aₜ
Environment（环境）
  ↓ 状态 sₜ₊₁ + 奖励 rₜ
Agent 更新策略 π(a|s)
```

**Bellman方程：**

$$Q(s,a) = r + \gamma \cdot \max_{a'} Q(s', a') \quad (\gamma \text{为折扣因子，} 0 < \gamma < 1)$$

| 类别 | 算法 | 特点 |
|------|------|------|
| 基于价值 | Q-Learning、DQN | 离散动作空间 |
| 基于策略 | REINFORCE、PPO | 连续动作空间 |
| Actor-Critic | A3C、SAC | 结合价值与策略 |

---

## 十六、算法对比与知识图谱

### 传统算法对比

```
算法        类型    损失函数      输出        决策边界    适用场景
─────────────────────────────────────────────────────────────────
线性回归    回归    MSE           连续值      —          预测房价、温度
逻辑回归    分类    交叉熵        概率(0~1)   线性        垃圾邮件、情感分析
SVM         分类    Hinge Loss    +1/-1       线性/非线性  文本分类、图像识别
决策树      分类    信息增益/基尼  类别        非线性      可解释性要求高的场景
随机森林    分类    集成投票      类别        非线性      高维数据、特征选择
朴素贝叶斯  分类    后验概率      概率        线性        文本分类、小数据集
```

### 它们和深度学习的关系

```
线性回归 → 神经网络去掉激活函数的单层版本
逻辑回归 → 单层神经网络 + Sigmoid 激活
SVM      → 启发了 max-margin 思想，但被深度学习在大多数任务上超越

深度学习本质上就是把这些简单模型堆叠起来：
  输入 → [线性变换 + 激活函数] × N层 → 输出
```

### 知识图谱总结

```
机器学习
├── 有监督学习
│   ├── 传统ML：逻辑回归、决策树、随机森林、贝叶斯、SVM
│   └── 深度学习：CNN（图像）、RNN/LSTM（序列）、Transformer（NLP）
├── 无监督学习
│   ├── 聚类：K-Means、DBSCAN、GMM
│   ├── 降维：PCA、t-SNE、AE
│   └── 生成模型：VAE、GAN、DBM
└── 强化学习
    ├── 基于价值：Q-Learning、DQN
    ├── 基于策略：PPO、TRPO
    └── Actor-Critic：A3C、SAC
```

这些基础算法理解透了，再看神经网络就是"同样的套路，只是层数多了"。
