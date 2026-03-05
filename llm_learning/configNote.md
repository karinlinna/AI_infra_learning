## claydeapi
https://www.yuque.com/nicaisadasd/fwextu/ekk2q8nrf3ow4k9q
《中转api使用claude code教程》
征集买家秀晒图活动，评价20字以上+2张效果图，赠送一张Claude Code Max 请求【100-200次】59a7fb4943a24d97b4fba76db2940e66
q邮箱+大写短密码

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
  python3 -m venv venv                                                           
  source venv/bin/activate                                                       
  pip install torch transformers psutil   

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

## 大模型并发
维度 大模型 Infra（训练 + 推理）
计算核心 — GPU/TPU/NPU 异构集群，张量并行 / 矩阵运算优化（CUDA/CANN）
计算形态 — 数秒–数天有状态长任务（训练周级；推理会话级 KV Cache）
网络要求 — InfiniBand/RoCE+RDMA，NCCL 集合通信，延迟 < 10 微秒
存储架构 — 并行文件系统 + 对象存储混合，TB/s 级吞吐，Checkpoint / 预取优化
调度与并行 — 模型并行 / 数据并行 / 流水线并行，ZeRO/FSDP 分片，显存精准管控
容错机制 — 长任务断点续训，梯度检查点，节点故障不丢进度
软件栈 — PyTorch/TensorFlow+DeepSpeed/vLLM+Triton+FlashAttention
成本模型 — 弹性 GPU / 存储峰值，训练占 TCO 70%+，推理按 token 计费

Plan to implement                                                            │
│                                                                              │
│ Mini LLM Infra 教学项目计划                                                  │
│                                                                              │
│ Context                                                                      │
│                                                                              │
│ 用户希望在笔记本电脑（无 GPU）上构建一个可运行的教学项目，覆盖大模型 Infra   │
│ 的 8 个核心维度。项目既要有可运行代码，也要有详细注释作为学习笔记。          │
│                                                                              │
│ 现有仓库已有 PyTorch + transformers 环境（Python 3.12 venv），以及一个简单的 │
│  gpt2_test.py。                                                              │
│                                                                              │
│ 项目结构                                                                     │
│                                                                              │
│ 在仓库根目录创建 mini_infra/ 目录：                                          │
│                                                                              │
│ mini_infra/                                                                  │
│ ├── README.md                    # 项目总览 + 8 维度对照表                   │
│ ├── requirements.txt             # 额外依赖（psutil）                        │
│ │                                                                            │
│ ├── 1_compute_core/              # 维度1: 计算核心                           │
│ │   └── tensor_ops_demo.py       # 矩阵运算/张量并行原理演示                 │
│ │                                                                            │
│ ├── 2_model/                     # 维度2: 计算形态（模型定义）               │
│ │   └── nano_gpt.py              # 从零手写 NanoGPT（~100行），含 KV Cache   │
│ │                                                                            │
│ ├── 3_parallel/                  # 维度5: 调度与并行                         │
│ │   ├── data_parallel.py         # 数据并行 (模拟多 worker)                  │
│ │   ├── tensor_parallel.py       # 张量并行 (矩阵切分)                       │
│ │   └── pipeline_parallel.py     # 流水线并行 (层切分)                       │
│ │                                                                            │
│ ├── 4_communication/             # 维度3: 网络/集合通信                      │
│ │   └── collective_ops.py        # 用 multiprocessing 模拟                   │
│ all-reduce/all-gather/reduce-scatter                                         │
│ │                                                                            │
│ ├── 5_storage/                   # 维度4: 存储架构                           │
│ │   └── checkpoint_manager.py    # Checkpoint 保存/加载/分片/预取            │
│ │                                                                            │
│ ├── 6_fault_tolerance/           # 维度6: 容错机制                           │
│ │   └── resilient_trainer.py     # 模拟节点故障 + 断点续训                   │
│ │                                                                            │
│ ├── 7_inference/                 # 维度2+7: 推理 + 软件栈                    │
│ │   └── inference_engine.py      # KV Cache 推理 + 简易 continuous batching  │
│ │                                                                            │
│ ├── 8_cost_monitor/              # 维度8: 成本模型                           │
│ │   └── monitor.py               # CPU/内存监控 + token 吞吐 + 成本估算      │
│ │                                                                            │
│ └── run_all.py                   # 一键运行所有 demo                         │
│                                                                              │
│ 各模块设计                                                                   │
│                                                                              │
│ 1. 1_compute_core/tensor_ops_demo.py                                         │
│                                                                              │
│ - 演示矩阵乘法、GEMM 运算                                                    │
│ - 对比 naive 实现 vs PyTorch 优化实现的性能差异                              │
│ - 演示张量分片（模拟 GPU 上的 tensor core 概念）                             │
│ - 半精度 (float16) vs 单精度 (float32) 对比                                  │
│                                                                              │
│ 2. 2_model/nano_gpt.py                                                       │
│                                                                              │
│ - 手写一个 ~4 层、64 维的 Transformer decoder                                │
│ - 包含：Embedding、Multi-Head Attention、FFN、LayerNorm                      │
│ - 实现 KV Cache 机制                                                         │
│ - 代码全中文注释                                                             │
│                                                                              │
│ 3. 3_parallel/                                                               │
│                                                                              │
│ - data_parallel.py: 用 Python multiprocessing 模拟 2 个                      │
│ worker，各自前向+反向，手动 all-reduce 梯度                                  │
│ - tensor_parallel.py: 将一个线性层的权重矩阵按列切分到 2                     │
│ 个"设备"，各自计算后合并                                                     │
│ - pipeline_parallel.py: 将模型层分成 2 个 stage，演示 micro-batch 流水线调度 │
│                                                                              │
│ 4. 4_communication/collective_ops.py                                         │
│                                                                              │
│ - 用 torch.distributed 的 Gloo 后端（CPU 可用）或 multiprocessing 模拟       │
│ - 实现并可视化：all-reduce、all-gather、reduce-scatter、broadcast            │
│ - 打印通信耗时和数据流向                                                     │
│                                                                              │
│ 5. 5_storage/checkpoint_manager.py                                           │
│                                                                              │
│ - 完整的 checkpoint 生命周期：save / load / 分片存储                         │
│ - 模拟大模型 checkpoint 分片（按层拆分保存）                                 │
│ - 异步预取（用线程模拟）                                                     │
│ - 存储性能统计（写入吞吐 MB/s）                                              │
│                                                                              │
│ 6. 6_fault_tolerance/resilient_trainer.py                                    │
│                                                                              │
│ - 基于 2_model/ 和 5_storage/ 组合                                           │
│ - 训练循环中随机模拟"节点故障"（抛异常）                                     │
│ - 自动从最近 checkpoint 恢复训练                                             │
│ - 梯度检查点 (gradient checkpointing) 演示以节省内存                         │
│                                                                              │
│ 7. 7_inference/inference_engine.py                                           │
│                                                                              │
│ - 加载训练好的 NanoGPT 权重                                                  │
│ - 实现 KV Cache 增量推理                                                     │
│ - 简易 continuous batching（多请求合并推理）                                 │
│ - token/s 吞吐统计                                                           │
│                                                                              │
│ 8. 8_cost_monitor/monitor.py                                                 │
│                                                                              │
│ - 实时监控：CPU 利用率、内存占用、进程耗时                                   │
│ - 训练成本估算（基于实际 GPU 价格换算）                                      │
│ - 推理成本：每 token 耗时 → 换算为 $/1M tokens                               │
│ - 输出格式化报表                                                             │
│                                                                              │
│ run_all.py                                                                   │
│                                                                              │
│ - 依次运行所有模块，每个模块前打印维度说明                                   │
│ - 支持 --module N 单独运行某个模块                                           │
│ - 汇总输出"8 维度覆盖报告"                                                   │
│                                                                              │
│ 技术约束                                                                     │
│                                                                              │
│ - 纯 CPU 运行，所有 GPU 概念用 CPU 张量 + multiprocessing 模拟               │
│ - 零外部服务依赖，只需 PyTorch + psutil                                      │
│ - 每个文件独立可运行，python mini_infra/1_compute_core/tensor_ops_demo.py    │
│ - 全中文注释，每个文件开头有该维度的知识点总结                               │
│                                                                              │
│ 依赖                                                                         │
│                                                                              │
│ 仅需在现有 venv 基础上额外安装 psutil：                                      │
│ pip install psutil                                                           │
│                                                                              │
│ 验证方式                                                                     │
│                                                                              │
│ # 安装依赖                                                                   │
│ pip install psutil                                                           │
│                                                                              │
│ # 一键运行全部 demo                                                          │
│ python mini_infra/run_all.py                                                 │
│                                                                              │
│ # 或单独运行某个模块                                                         │
│ python mini_infra/run_all.py --module 1                                      │
│ python mini_infra/1_compute_core/tensor_ops_demo.py                          │
│                                                                              │
│ 每个模块运行后会输出：                                                       │
│ - 知识点摘要                                                                 │
│ - 运行结果（数值/性能指标）                                                  │
│ - 与真实大模型 Infra 的对照说明       

## 算法工程师要求

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


