# 机器学习与深度学习算法全览

---

## 一、基础分类算法

### 1. 逻辑回归 (Logistic Regression)

**核心思想：** 虽然名字带"回归"，但它是一种二分类（可扩展到多分类）算法。它用 Sigmoid 函数将线性输出压缩到 (0, 1) 区间，表示属于某类的概率。

**数学原理：**

```
线性部分：  z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
Sigmoid：   σ(z) = 1 / (1 + e^(-z))
预测：      ŷ = 1 if σ(z) ≥ 0.5 else 0
```

**损失函数（交叉熵）：**
```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**训练方式：** 梯度下降法更新参数 w 和 b。

**Python 实现：**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

**优缺点：**
- ✅ 简单高效，可解释性强，输出概率值
- ❌ 只能处理线性可分问题，对特征工程依赖大

---

### 2. 决策树 (Decision Tree)

**核心思想：** 通过一系列"if-else"规则对数据进行递归划分，每次选择最优特征和阈值进行分裂，直到满足停止条件。

**关键概念：**

- **信息熵（ID3）：** `H(D) = -Σ pᵢ·log₂(pᵢ)`，衡量数据集的混乱程度
- **信息增益：** `IG = H(D) - H(D|特征A)`，选择使信息增益最大的特征
- **基尼系数（CART）：** `Gini = 1 - Σ pᵢ²`，越小越纯净

**构建过程：**
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

### 3. 随机森林 (Random Forest)

**核心思想：** 集成学习方法，通过 **Bagging（自举聚合）** 训练多棵决策树，最终用投票（分类）或平均（回归）来综合结果。

**关键步骤：**
```
1. 从训练集中有放回地随机抽取 n 个样本（Bootstrap）
2. 在每个节点，随机选取 √p 个特征（p为总特征数）进行最优分裂
3. 重复构建 T 棵树（通常100~500棵）
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

**优缺点：**
- ✅ 抗过拟合能力强，可处理高维数据，能评估特征重要性
- ❌ 模型较大，可解释性弱于单棵决策树，训练较慢

---

### 4. 朴素贝叶斯 (Naive Bayes)

**核心思想：** 基于贝叶斯定理，并假设所有特征之间**条件独立**（这一"朴素"假设在实际中很少成立，但效果往往不错）。

**贝叶斯定理：**
```
P(类别C | 特征X) = P(特征X | 类别C) × P(类别C) / P(特征X)
                 ∝ 似然度 × 先验概率
```

**三种常用变体：**

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

### 5. 支持向量机 (SVM)

**核心思想：** 在特征空间中找到一个**最优超平面**，使得两类之间的**间隔（Margin）最大化**。

**关键概念：**

- **支持向量：** 距离超平面最近的那些样本点
- **硬间隔：** 完全线性可分时，最大化 `2/||w||`
- **软间隔：** 允许部分误分类，引入松弛变量 ξ 和惩罚参数 C
- **核函数：** 将低维不可分数据映射到高维空间

```
常用核函数：
  线性核：  K(x, z) = xᵀz
  RBF核：   K(x, z) = exp(-γ||x-z||²)   ← 最常用
  多项式核：K(x, z) = (xᵀz + c)ᵈ
```

**优化目标（对偶问题）：**
```
最大化：Σαᵢ - ½ΣΣαᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
约束：  0 ≤ αᵢ ≤ C，Σαᵢyᵢ = 0
```

**Python 实现：**
```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)
```

**优缺点：**
- ✅ 高维空间效果好，泛化能力强，对噪声鲁棒
- ❌ 大数据集训练慢，核函数和参数选择难

---

## 二、深度学习算法

### 6. 卷积神经网络 (CNN)

**核心思想：** 专为处理**网格状数据**（如图像）设计，通过局部感受野、权值共享、空间下采样三大思想大幅减少参数量。

**核心层结构：**

```
输入图像 → [卷积层 → 激活(ReLU)] × N → 池化层 → 全连接层 → 输出
```

**卷积操作：**
```
特征图输出大小 = (输入尺寸 - 卷积核尺寸 + 2×Padding) / Stride + 1

例：输入32×32，卷积核3×3，Padding=1，Stride=1
    输出 = (32 - 3 + 2) / 1 + 1 = 32×32  （尺寸不变）
```

**经典架构演进：**

| 架构 | 年份 | 特点 |
|------|------|------|
| LeNet-5 | 1998 | 开创性 CNN |
| AlexNet | 2012 | 深层+ReLU+Dropout |
| VGG16 | 2014 | 全用3×3卷积堆叠 |
| ResNet | 2015 | 残差连接，解决梯度消失 |
| EfficientNet | 2019 | 复合缩放策略 |

**PyTorch 实现示例：**
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

### 7. 循环神经网络 (RNN)

**核心思想：** 处理**序列数据**（文本、时序），通过隐藏状态将前一时刻的信息传递到下一时刻。

**基本公式：**
```
hₜ = tanh(Wₕ·hₜ₋₁ + Wₓ·xₜ + b)    # 隐藏状态更新
yₜ = Wᵧ·hₜ                          # 输出
```

**梯度消失问题：** 随着序列变长，梯度在反向传播中会指数级衰减，导致难以学习长距离依赖。

**改进方案：**

**LSTM（长短期记忆网络）：** 引入三个门控机制
```
遗忘门：fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)     # 决定遗忘多少旧信息
输入门：iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)     # 决定写入多少新信息
输出门：oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)     # 决定输出多少信息
细胞态：Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙tanh(WC·[hₜ₋₁,xₜ])
```

**GRU（门控循环单元）：** LSTM的简化版，只有重置门和更新门，参数更少。

**PyTorch 实现：**
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

### 8. 深度玻尔兹曼机 (DBM)

**核心思想：** 一种**生成式概率模型**，由多层随机二值神经元组成，通过无监督方式学习数据的深层概率分布。

**结构特点：**
```
可见层 (v) ↔ 隐藏层1 (h¹) ↔ 隐藏层2 (h²) ↔ ...
（层内无连接，层间无向全连接）
```

**能量函数：**
```
E(v,h) = -vᵀW¹h¹ - h¹ᵀW²h² - bᵀv - cᵀh
```

**训练方法：** 对比散度（Contrastive Divergence, CD）算法近似求解最大似然。

**与 RBM 的关系：** RBM（受限玻尔兹曼机）是 DBM 的基本构建单元（只有一个隐藏层），多个 RBM 逐层堆叠可预训练 DBM。

**应用场景：** 图像生成、特征无监督预训练、推荐系统协同过滤。

---

### 9. 自编码器 (AutoEncoder, AE)

**核心思想：** 通过**编码器**将输入压缩为低维潜在表示（瓶颈层），再通过**解码器**重建原始输入，迫使模型学习数据的核心特征。

**结构：**
```
输入x → [编码器] → 潜在向量z（低维） → [解码器] → 重建x̂
目标：最小化重建误差 ||x - x̂||²
```

**主要变体：**

| 变体 | 特点 | 应用 |
|------|------|------|
| 降噪AE (DAE) | 输入加噪，重建原始 | 去噪、鲁棒特征 |
| 稀疏AE (SAE) | 对隐层加稀疏约束 | 特征提取 |
| 变分AE (VAE) | 潜在空间建模为概率分布 | 图像生成 |
| 收缩AE (CAE) | 对雅可比矩阵加正则 | 流形学习 |

**PyTorch 实现（VAE）：**
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

## 三、学习范式

### 10. 有监督学习 (Supervised Learning)

**定义：** 训练数据包含输入 X 和对应标签 y，模型学习 f: X → y 的映射关系。

**核心流程：**
```
标注数据 → 特征工程 → 模型选择 → 训练(最小化损失) → 验证/调参 → 测试评估
```

**常见任务与算法：**

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

---

### 11. 无监督学习 (Unsupervised Learning)

**定义：** 训练数据**只有输入 X，没有标签**，模型自行发现数据的内在结构。

**三大任务方向：**

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

**常用评估方法（无真实标签时）：**
- 轮廓系数（Silhouette Score）：越接近1越好
- 戴维斯-鲍丁指数（DB Index）：越小越好

---

### 12. 强化学习 (Reinforcement Learning)

**核心思想：** 智能体（Agent）在环境（Environment）中通过**试错**，学习最大化累积奖励的策略（Policy）。

**核心要素：**
```
Agent（智能体）
  ↓ 动作 aₜ
Environment（环境）
  ↓ 状态 sₜ₊₁ + 奖励 rₜ
Agent 更新策略 π(a|s)
```

**关键概念：**
- **状态 S：** 环境当前的描述
- **动作 A：** 智能体可执行的操作集合
- **奖励 R：** 执行动作后从环境获得的即时反馈
- **策略 π(a|s)：** 在状态s下选择动作a的概率
- **值函数 V(s)：** 从状态s开始，按策略π执行的期望累积奖励
- **Q值 Q(s,a)：** 在状态s执行动作a后的期望累积奖励

**Bellman方程：**
```
Q(s,a) = r + γ·max Q(s', a')    （γ为折扣因子，0<γ<1）
```

**主流算法分类：**

| 类别 | 算法 | 特点 |
|------|------|------|
| 基于价值 | Q-Learning、DQN | 离散动作空间 |
| 基于策略 | REINFORCE、PPO | 连续动作空间 |
| Actor-Critic | A3C、SAC | 结合价值与策略 |

**DQN 核心代码：**
```python
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# 训练关键：经验回放(Replay Buffer) + 目标网络(Target Network)
# Q_target = r + γ * max(Q_target_net(s'))
# Loss = MSE(Q_online(s,a), Q_target)
```

**DQN 两大关键创新：**
- **经验回放：** 打破样本时序相关性，提高数据利用率
- **目标网络：** 定期更新，稳定训练目标，防止振荡

---

## 四、知识图谱总结

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
