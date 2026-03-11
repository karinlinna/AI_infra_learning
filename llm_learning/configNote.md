## claydeapi
https://www.yuque.com/nicaisadasd/fwextu/ekk2q8nrf3ow4k9q
《中转api使用claude code教程》
征集买家秀晒图活动，评价20字以上+2张效果图，赠送一张Claude Code Max 请求【100-200次】59a7fb4943a24d97b4fba76db2940e66
q邮箱+大写短密码

https://claude888.creativeai.work karinlinna 小短字母 1天

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


负责垂直领域LLM模型算法落地，包括但不限于：领域业务数据清洗、高质量SFT数据构建和模型训练，参与设计、优化大模型指令、RLHF/RLAIF、模型评测，提升大模型训练/推理效果；
构建RAG链路：混合检索、向量数据库 Milvus/Elastic 部署、重排序优化，降低幻觉；
设计自动化评测框架，集成领域自定义指标，输出模型多维评测报告；
深入调研和关注大模型前沿技术，包括智能体、强化学习、模型加速等，并进行复现、部署、测试及持续创新和优化；
与项目/工程团队协作，服务客户完成驻场开发。

深入理解NLP基础理论与模型原理，熟悉Transformer、BERT、GPT等主流架构；熟练掌握Prompt构建、SFT（有监督微调）、RHLF等关键方法与流程；
精通PyTorch、DeepSpeed、vllm等框架，能够独立完成算法和模型的开发和优化，具备实战经验者优先；


熟悉深度学习理论和实践，熟练掌握PyTorch、TensorFlow等深度学习框架。
熟练掌握1种以上深度学习模型/时序模型的训练；
具有扎实的数学基础和强大的编程能力，熟练掌握Python、C++/Java等编程语言。
参与过大规模数据集的处理或模型预训练项目，有实际的模型调优经验。
需要熟练掌握 Python，并具备独立交付能力；熟悉大模型应用开发常见范式，至少熟练掌握其中两项：RAG/向量检索与重排、结构化信息抽取、函数调用/工具调用（Agent）、提示工程与输出约束；具备工程化能力，包括接口开发、日志与异常处理、版本管理与可维护性；对结果可追溯、可解释、可评测有明确意识。
加分项
有企业知识库问答或智能体项目经验；有时序数据/工业数据处理经验（数据质量、对齐、异常检测等）；有私有化部署与性能优化经验（容器化、GPU 推理服务、并发与缓存）；有评测体系与线上监控经验。

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


帮我大改一下：
要求
1.把文件服务改成一个单纯简单的文件服务，通过接口调用，无需spring和数据库，但请保留oss
2.以前是通过url调用，很繁琐，只想通过upload等接口进行调用。
3.以前是把file、record以表的形式存在数据库里，现在不用数据库期望以json文件的格式存在文件存放处，并且是一个多条json的文件，也就是说把以前的数据库表以json数组（比方）存在本地。或oss。和处理的文件存在一起即可。
5，请改的简洁好用，删掉不需要的类和方法。
6.请保证json文件内的每一条都是上传完或者备份完的数据

│ - JSON库：Jackson（OSS SDK间接依赖，无需额外引入）                           │
│ - 并发安全：ReentrantReadWriteLock + 原子文件写入（写临时文件后rename）      │
│ - ID生成：保留sedi-distributedID依赖                      │
│ - 备份/恢复：不删除 且和文件服务一样   一个单纯简单的，通过接口调用，无需spring和数据库，但请保留oss                                              │
│ - file-core-test：改     