"""
Mini LLM Infra 教学项目 - 一键运行所有模块

用法:
    python mini_infra/run_all.py          # 运行全部8个模块
    python mini_infra/run_all.py --module 1   # 只运行模块1
    python mini_infra/run_all.py --module 3   # 只运行模块3
"""

import argparse
import importlib.util
import sys
import os
import time


# 8个模块的定义：(编号, 目录名, 文件名, 维度名称, 简介)
MODULES = [
    (1, "1_compute_core", "tensor_ops_demo.py",
     "计算核心 (Compute Core)",
     "矩阵运算/GEMM、张量分片、半精度vs单精度对比"),

    (2, "2_model", "nano_gpt.py",
     "计算形态 - 模型定义 (Model Architecture)",
     "从零手写 NanoGPT Transformer Decoder，含 KV Cache"),

    (3, "3_parallel", "data_parallel.py",
     "调度与并行 - 数据并行 (Data Parallelism)",
     "多 worker 模拟数据并行，手动 All-Reduce 梯度聚合"),

    (4, "3_parallel", "tensor_parallel.py",
     "调度与并行 - 张量并行 (Tensor Parallelism)",
     "Megatron-LM 风格列并行/行并行，矩阵切分计算"),

    (5, "3_parallel", "pipeline_parallel.py",
     "调度与并行 - 流水线并行 (Pipeline Parallelism)",
     "GPipe 风格 micro-batch 流水线调度"),

    (6, "4_communication", "collective_ops.py",
     "网络与集合通信 (Collective Communication)",
     "模拟 All-Reduce / All-Gather / Reduce-Scatter / Broadcast"),

    (7, "5_storage", "checkpoint_manager.py",
     "存储架构 (Storage Architecture)",
     "Checkpoint 保存/加载/分片存储/异步预取"),

    (8, "6_fault_tolerance", "resilient_trainer.py",
     "容错机制 (Fault Tolerance)",
     "模拟节点故障 + 断点续训 + 梯度检查点"),

    (9, "7_inference", "inference_engine.py",
     "推理优化 (Inference Optimization)",
     "KV Cache 增量推理 + 简易 Continuous Batching"),

    (10, "8_cost_monitor", "monitor.py",
      "成本模型与监控 (Cost Modeling & Monitoring)",
      "CPU/内存监控 + 训练/推理成本估算"),
]


def get_base_dir():
    """获取 mini_infra 目录的绝对路径"""
    return os.path.dirname(os.path.abspath(__file__))


def run_module(module_info):
    """运行单个模块"""
    num, dir_name, file_name, dimension, description = module_info
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, dir_name, file_name)

    print("\n" + "=" * 70)
    print(f"  模块 {num}: {dimension}")
    print(f"  文件: {dir_name}/{file_name}")
    print(f"  简介: {description}")
    print("=" * 70)

    if not os.path.exists(file_path):
        print(f"  [错误] 文件不存在: {file_path}")
        return False

    # 动态加载并运行模块
    module_name = f"mini_infra_module_{num}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    start_time = time.time()
    try:
        # 临时修改 sys.argv 以避免模块内部的 argparse 冲突
        old_argv = sys.argv
        sys.argv = [file_path]
        spec.loader.exec_module(module)
        sys.argv = old_argv
        elapsed = time.time() - start_time
        print(f"\n  [完成] 模块 {num} 运行耗时: {elapsed:.2f} 秒")
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  [错误] 模块 {num} 运行失败 ({elapsed:.2f}秒): {e}")
        import traceback
        traceback.print_exc()
        return False


def print_coverage_report(results):
    """打印 8 维度覆盖报告"""
    dimensions = {
        "维度1 - 计算核心":     [1],
        "维度2 - 计算形态(模型)": [2],
        "维度3 - 网络/集合通信":  [6],
        "维度4 - 存储架构":      [7],
        "维度5 - 调度与并行":    [3, 4, 5],
        "维度6 - 容错机制":      [8],
        "维度7 - 推理/软件栈":   [9],
        "维度8 - 成本模型":      [10],
    }

    print("\n" + "=" * 70)
    print("  大模型 Infra 8 维度覆盖报告")
    print("=" * 70)

    all_covered = True
    for dim_name, module_nums in dimensions.items():
        statuses = []
        for n in module_nums:
            if n in results:
                statuses.append("PASS" if results[n] else "FAIL")
            else:
                statuses.append("SKIP")

        if all(s == "PASS" for s in statuses):
            icon = "[PASS]"
        elif any(s == "FAIL" for s in statuses):
            icon = "[FAIL]"
            all_covered = False
        else:
            icon = "[SKIP]"
            all_covered = False

        module_detail = ", ".join(
            f"模块{n}:{s}" for n, s in zip(module_nums, statuses)
        )
        print(f"  {icon} {dim_name:<24} ({module_detail})")

    print("-" * 70)
    if all_covered:
        print("  全部 8 个维度已覆盖!")
    else:
        print("  部分维度未完全通过，请检查上方日志。")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Mini LLM Infra 教学项目 - 一键运行所有 demo"
    )
    parser.add_argument(
        "--module", type=int, default=None,
        help="只运行指定编号的模块 (1-10)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Mini LLM Infra 教学项目")
    print("  覆盖大模型基础设施的 8 个核心维度")
    print("  纯 CPU 运行，无需 GPU")
    print("=" * 70)

    results = {}

    if args.module is not None:
        # 运行单个模块
        target = None
        for m in MODULES:
            if m[0] == args.module:
                target = m
                break
        if target is None:
            print(f"  [错误] 未找到模块 {args.module}，可选: 1-{len(MODULES)}")
            sys.exit(1)
        success = run_module(target)
        results[args.module] = success
    else:
        # 运行全部模块
        total = len(MODULES)
        for i, m in enumerate(MODULES):
            print(f"\n{'#' * 70}")
            print(f"  进度: [{i+1}/{total}]")
            print(f"{'#' * 70}")
            success = run_module(m)
            results[m[0]] = success

    # 打印覆盖报告
    print_coverage_report(results)


if __name__ == "__main__":
    main()
