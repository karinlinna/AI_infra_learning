# Mini LLM Infra 教学项目

纯 CPU 可运行的大模型基础设施教学项目，覆盖 8 个核心维度。

## 环境要求

- Python 3.10+
- PyTorch (CPU 版本即可)
- psutil

```bash
pip install psutil
```

## 项目结构

```
mini_infra/
├── README.md                    # 本文件
├── requirements.txt             # 额外依赖
├── run_all.py                   # 一键运行所有 demo
│
├── 1_compute_core/              # 维度1: 计算核心
│   └── tensor_ops_demo.py       # GEMM、张量分片、精度对比
│
├── 2_model/                     # 维度2: 计算形态（模型定义）
│   └── nano_gpt.py              # 手写 NanoGPT + KV Cache
│
├── 3_parallel/                  # 维度5: 调度与并行
│   ├── data_parallel.py         # 数据并行 (模拟多 worker)
│   ├── tensor_parallel.py       # 张量并行 (Megatron 风格)
│   └── pipeline_parallel.py     # 流水线并行 (GPipe 风格)
│
├── 4_communication/             # 维度3: 网络/集合通信
│   └── collective_ops.py        # All-Reduce / All-Gather / Reduce-Scatter
│
├── 5_storage/                   # 维度4: 存储架构
│   └── checkpoint_manager.py    # Checkpoint 保存/加载/分片/预取
│
├── 6_fault_tolerance/           # 维度6: 容错机制
│   └── resilient_trainer.py     # 模拟故障 + 断点续训
│
├── 7_inference/                 # 维度7: 推理优化
│   └── inference_engine.py      # KV Cache 推理 + Continuous Batching
│
└── 8_cost_monitor/              # 维度8: 成本模型
    └── monitor.py               # 资源监控 + 成本估算
```

## 8 维度对照表

| 维度 | 模块 | 核心概念 | 真实系统对应 |
|------|------|---------|-------------|
| 1. 计算核心 | `1_compute_core/` | GEMM、Tensor Core、精度 | cuBLAS, CUTLASS |
| 2. 计算形态 | `2_model/` | Transformer、KV Cache | GPT, LLaMA |
| 3. 网络通信 | `4_communication/` | 集合通信原语 | NCCL, Gloo |
| 4. 存储架构 | `5_storage/` | Checkpoint 分片 | HDFS, S3 |
| 5. 调度并行 | `3_parallel/` | DP/TP/PP | Megatron-LM, DeepSpeed |
| 6. 容错机制 | `6_fault_tolerance/` | 断点续训、梯度检查点 | Elastic Training |
| 7. 推理优化 | `7_inference/` | KV Cache、批处理 | vLLM, TensorRT-LLM |
| 8. 成本模型 | `8_cost_monitor/` | FLOPs 估算、$/token | Cloud Pricing |

## 运行方式

```bash
# 一键运行全部
python mini_infra/run_all.py

# 运行单个模块（编号 1-10）
python mini_infra/run_all.py --module 1
python mini_infra/run_all.py --module 2

# 直接运行单个文件
python mini_infra/1_compute_core/tensor_ops_demo.py
python mini_infra/2_model/nano_gpt.py
```

## 设计原则

- **纯 CPU 运行**：所有 GPU 概念用 CPU 张量 + multiprocessing 模拟
- **零外部服务依赖**：只需 PyTorch + psutil
- **每个文件独立可运行**：无跨模块 import 依赖
- **全中文注释**：每个文件开头有该维度的知识点总结
