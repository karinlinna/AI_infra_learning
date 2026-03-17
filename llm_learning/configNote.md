## claydeapi
https://www.yuque.com/nicaisadasd/fwextu/ekk2q8nrf3ow4k9q
《中转api使用claude code教程》
征集买家秀晒图活动，评价20字以上+2张效果图，赠送一张Claude Code Max 请求【100-200次】59a7fb4943a24d97b4fba76db2940e66
q邮箱+大写短密码

https://docs.packyapi.com/docs/register/1-register.html https://www.packyapi.com/console/topup packy karinlinna 数字长 这个目前比较好用
或者试试dockcoding

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

  ---                                                                           
  1. 并发安全：JsonMetadataStore 锁策略不完整                                   
                                                                                
  save、delete、importAll 方法直接操作 dataMap 后调用 persist()，但只有         
  persist() 内部加了 writeLock。这意味着两个并发的 save 调用可能交错写入        
  dataMap，然后各自 persist 时序列化出不一致的快照。

  findAll、findByIds 也没有加读锁，和 reload 并发时会读到空数据。

  JsonOperationRecordStore 有同样的问题。

  ---
  2. BackupFeedback 线程安全问题

  BackupFeedback 是普通 POJO（@Data），但它在异步线程中被写入（setResult、setPro
  gress），在主线程中被读取（isCompleted、isInProgress）。没有任何 volatile
  或同步机制，主线程可能永远看不到异步线程写入的 result = 1，导致 pollUntilDone
  死循环。

  result 和 progress 字段应该用 volatile 修饰，或者用 AtomicInteger。

  ---
  3. FileMagicNumberValidator 逻辑 bug

  isValidFileType 方法（FileMagicNumberValidator.java:28-42）：

  for (String number : magicNumber) {
      if (fileBytes.length < number.length() / 2) {
          return true;   // 文件太短直接放行？应该 return false
      } else {
          String fileHeader = bytesToHex(fileBytes, number.length() / 2);
          return fileHeader.equalsIgnoreCase(number);  // 只检查第一个就 return
  了
      }
  }

  两个问题：
  - 文件字节数不够时直接返回 true（放行），应该返回 false（拒绝）
  - 循环体内直接 return，只检查了 magicNumber 列表的第一个值。对于 mp4、mov
  等有多个合法 magic number 的格式，永远只匹配第一个，其余的永远匹配不到

  ---
  4. FileService.getFileSHA512 把文件名混入 hash

  FileService.java:278-298，hash 计算把 fileName、bucket、isCloud、defineKey
  都混入了 SHA-512。这意味着同一个文件内容，换个文件名上传就会得到不同的
  hash，去重就失效了。字段名叫 md5 但实际算的是 SHA-512，也容易误导。

  ---
  5. 备份/还原没有使用自定义线程池

  BackupRestoreService 中 CompletableFuture.runAsync() 使用的是
  ForkJoinPool.commonPool()。这个公共池是全 JVM
  共享的，如果备份/还原任务耗时较长或并发多个，会影响其他使用 common pool
  的代码。应该传入一个专用的 ExecutorService。

  ---
  6. ZipUtil 是死代码

  ZipUtil 和 CrossPlatformZipUtil 功能高度重复，且 ZipUtil.extractFromZip
  有资源泄漏（FileInputStream 没有 close）。目前代码中只使用了
  CrossPlatformZipUtil，ZipUtil 应该删除。

  ---
  7. persist() 失败被静默吞掉

  JsonMetadataStore.persist() 和 JsonOperationRecordStore.persist()
  中，写入失败只 log.error，不抛异常。调用方（save、delete、importAll）完全不知
  道持久化失败了，内存和磁盘数据会悄悄不一致。至少应该抛出异常让调用方感知。

  ---
  8. OSS 模式下备份/还原不支持

  BackupRestoreService 的备份逻辑只打包本地 filecore 目录的物理文件。如果
  cloudDeploy=true，物理文件在 OSS 上，本地没有文件可打包，备份出来的 ZIP 只有
  metadata.json，没有实际文件内容。还原时也只处理本地文件。接口上没有任何提示或
  校验。

  ---
  9. removeFile 删除物理文件失败时元数据不删

  FileService.java:211-219：只有 deleted == true
  时才删元数据。如果物理文件因为权限等原因删除失败，元数据保留，但用户会收到
  return false。这个设计本身可以接受，但问题是 LocalFileStorage.delete
  在文件不存在时也返回 false，这意味着如果物理文件已经丢失（比如被外部删除），元
  数据就永远删不掉了，成了僵尸记录。

  ---
  10. 没有资源清理机制

  FileCoreClient 没有 close() 方法。如果 cloudDeploy=true，OSSClient 永远不会被
  shutdown，连接池泄漏。应该让 FileCoreClient 实现 Closeable，在 close() 中释放
  OSS 客户端等资源。

  ---
  以上是按严重程度排列的主要问题。第 1、2 点是并发
  bug，在生产环境多线程场景下会出问题；第 3 点是功能
  bug；其余是设计层面的改进点。需要我帮你修哪些