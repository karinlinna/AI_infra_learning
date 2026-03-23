## claydeapi
https://www.yuque.com/nicaisadasd/fwextu/ekk2q8nrf3ow4k9q
《中转api使用claude code教程》
征集买家秀晒图活动，评价20字以上+2张效果图，赠送一张Claude Code Max 请求【100-200次】59a7fb4943a24d97b4fba76db2940e66
q邮箱+大写短密码

https://docs.packyapi.com/docs/register/1-register.html https://www.packyapi.com/console/topup packy karinlinna 数字长 这个目前比较好用 https://www.packyapi.com

或者试试dockcoding https://api.duckcoding.ai 

car :
karinlinna 小短字母 1天
https://claude.hk.cn/pastel/#/claude-carlist

https://claudeonline.top/list 15天 

autoDL 手机号 微信 80块

## open claw
https://zhuanlan.zhihu.com/p/2000850539936765122 openclaw 

## 一些配置
⏺ venv 目录已经不存在了。需要重新创建。运行以下命令：

  deactivate                                                                    
  rm -rf venv     
                                                                                
  然后确认你的 Python 3.12 路径：
  python3.12 --version
  如果能输出版本号，就用它创建新的虚拟环境：

  python3.12 -m venv .venv 
  source .venv/bin/activate
  pip install torch transformers
  cd /Users/lijialin/Work/Github/robotlearning/llm_learning

## 自学视频
https://space.bilibili.com/271305784/lists/1827462?type=season
https://huggingface.co/ q邮箱
transformers库
预训练（背后原理）、指令微调（有什么方法，适用什么场景）
显存与规模的估算，8b模型 3080显卡估算，怎么设置batchsize？要不要梯度累计？是否需要梯度检查点？
混合精度训练、deepspeed框架，如何做并行？
推理阶段：量化，模型规模变小，跑的更快、kvcache、投机采样、加速推理、降低部署成本
大模型的评估：怎么知道这个模型好？指标都有什么ppl？

深度学习基础：
梯度下降，损失函数，残差连接，学习率，卷积神经网络了解，

强化学习，上手dpo、ppo、grpo


## 算法工程师

深度学习及机器学习相关算法的开发和应用
深度学习相关技术栈及算法的预研和选型
算法相关的数据处理

熟悉linux系统，熟悉Python，熟悉tensorflow等深度学习框架
熟悉逻辑回归、随机森林、决策树、贝叶斯、SVM等基础分类算法
熟悉CNN、RNN、DBM、AE等人工智能算法及其原理
熟悉有监督学习、无监督学习、强化学习等原理及实现
具备极强的编程能力、学习能力和抗压能力


深入理解NLP基础理论与模型原理，熟悉Transformer、BERT、GPT等主流架构；熟练掌握Prompt构建、SFT（有监督微调）、RHLF等关键方法与流程；
精通PyTorch、DeepSpeed、vllm等框架，能够独立完成算法和模型的开发和优化，具备实战经验者优先；

1、跟踪、验证和探索前沿AI算法，提升模型性能
2、进行模型理论和工程创新，应用在更广泛的场景
3、开发概念验证系统，评估性能提升效果
4、发表论文，申请专利，参与学术交流
1、计算机科学、人工智能或相关领域硕士及以上学历
2、扎实的机器学习/深度学习算法基础
3、能够形式化推导神经网络的正向和反向过程
4、熟练掌握PyTorch或TensorFlow等深度学习框架
5、有AI领域顶会论文发表或大厂研究院工作经验者优先
加分项（非必需）：
对量子计算有初步了解或兴趣
有高性能计算或并行计算经验
有模型性能优化经验



  第一阶段：数学与编程基础                                                                                 
                 
  数学（必须扎实）：                                                                                       
  - 线性代数：矩阵运算、特征值分解、SVD（推荐：3Blue1Brown 线性代数系列）                                  
  - 概率统计：贝叶斯、分布、最大似然估计
  - 微积分：链式法则、梯度 → 这是理解反向传播的核心

  编程：
  - Python 熟练掌握（NumPy、Pandas）
  - 数据结构与算法基础

  第二阶段：机器学习基础

  - 吴恩达 Coursera《Machine Learning》课程
  - 关键概念：损失函数、梯度下降、过拟合/欠拟合、正则化
  - 手动实现一遍线性回归、逻辑回归、SVM

  第三阶段：深度学习核心

  理论：
  - 吴恩达《Deep Learning Specialization》5 门课
  - 重点掌握：
    - 前向传播 & 反向传播的数学推导（面试必考）
    - CNN、RNN/LSTM 原理
    - Batch Norm、Dropout、各种优化器（Adam 等）

  框架：
  - PyTorch 为主（业界主流）
  - 从零实现一个简单的神经网络，再用 PyTorch 重写
  - 推荐：李沐《动手学深度学习》(d2l.ai)

  第四阶段：NLP 与 Transformer

  这是岗位的核心要求：

  学习顺序：

  Word2Vec / GloVe（词向量基础）
          ↓
  RNN / LSTM / Seq2Seq + Attention
          ↓
  Transformer（精读论文 "Attention Is All You Need"）
          ↓
  BERT（编码器架构，理解 MLM 预训练）
          ↓
  GPT 系列（解码器架构，自回归生成）
          ↓
  LLaMA / Qwen 等开源大模型

  关键：Transformer 必须彻底搞懂
  - Self-Attention 的矩阵运算
  - Multi-Head Attention 为什么有效
  - 位置编码原理
  - 手推一遍前向和反向过程

  第五阶段：大模型训练与微调

  对应岗位要求的 SFT、RLHF：

  ┌────────────────────┬───────────────────────────┐
  │        技术        │          学什么           │
  ├────────────────────┼───────────────────────────┤
  │ Prompt Engineering │ 提示词设计、few-shot、CoT │
  ├────────────────────┼───────────────────────────┤
  │ SFT（有监督微调）  │ LoRA、QLoRA、全参微调     │
  ├────────────────────┼───────────────────────────┤
  │ RLHF               │ 奖励模型、PPO 算法        │
  ├────────────────────┼───────────────────────────┤
  │ DPO                │ RLHF 的简化替代方案       │
  └────────────────────┴───────────────────────────┘

  实践项目：
  - 用 LoRA 微调一个 LLaMA/Qwen 模型（你笔记里的代码就是这个）
  - 部署用 vLLM 做推理加速

  第六阶段：工程能力

  ┌──────────────────┬──────────────────┐
  │       工具       │       用途       │
  ├──────────────────┼──────────────────┤
  │ DeepSpeed / FSDP │ 分布式训练       │
  ├──────────────────┼──────────────────┤
  │ vLLM             │ 高效推理         │
  ├──────────────────┼──────────────────┤
  │ Hugging Face     │ 模型和数据集生态 │
  ├──────────────────┼──────────────────┤
  │ Weights & Biases │ 实验管理         │
  └──────────────────┴──────────────────┘

  第七阶段：论文与学术（进阶）

  - 精读 20-30 篇经典论文（Transformer、BERT、GPT、LoRA、RLHF 等）
  - 跟踪 arXiv 每周新论文
  - 复现 1-2 篇论文的实验
  - 目标顶会：ACL、EMNLP、NeurIPS、ICML

  ---
  实际建议

  1. 不要跳过数学，"能形式化推导正向和反向过程"是硬性要求
  2. 边学边做项目，比如你现在的这份笔记/代码就是很好的实践
  3. 先不用管加分项（量子计算等），把核心能力建立起来
  4. 预期路径：基础扎实的情况下，持续学习，逐步积累项目和论文经验

  你目前的笔记已经覆盖了第五阶段的内容，建议回头补第三、四阶段的理论基础，特别是 Transformer 的数学推导

