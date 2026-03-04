"""
================================================================================
模块1: 计算核心 (Compute Core) — Tensor 运算基础演示
================================================================================

知识总结:
---------
1. 大语言模型 (LLM) 的核心计算本质上是大规模矩阵乘法 (GEMM: General Matrix Multiply)。
   - Transformer 中的 QKV 投影、注意力计算、FFN 层全部依赖矩阵乘法。
   - 一次前向传播中，矩阵乘法占计算量的 90% 以上。

2. GPU Tensor Core 是专门加速矩阵乘法的硬件单元:
   - NVIDIA A100 的 Tensor Core 可以在一个时钟周期内完成 4x4 矩阵的乘加运算。
   - Tensor Core 原生支持 FP16/BF16/TF32/INT8 等低精度格式。
   - 利用低精度计算可以显著提升吞吐量 (FP16 吞吐约为 FP32 的 2 倍)。

3. 张量分片 (Tensor Sharding / Tiling):
   - 大矩阵无法一次放入计算单元，需要拆分成小块 (tile) 分批计算。
   - 分片策略直接影响缓存命中率和计算效率。
   - 这也是后续张量并行 (Tensor Parallelism) 的基础。

4. 精度与性能的权衡:
   - FP32 (32位浮点): 精度高，但计算慢、显存占用大。
   - FP16 (16位浮点): 精度略低，但计算快、显存占用减半。
   - 混合精度训练 (Mixed Precision): 用 FP16 加速计算，用 FP32 维护主权重。

本文件在纯 CPU 上模拟演示以上概念，帮助理解 LLM 基础设施的计算核心。
================================================================================
"""

import time
import numpy as np

# 尝试导入 PyTorch，用于对比优化后的矩阵运算性能
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[警告] 未安装 PyTorch，将跳过 PyTorch 相关对比演示。")
    print("       可通过 pip install torch 安装。\n")


# =============================================================================
# 第一部分: 朴素 Python 矩阵乘法 vs 优化实现
# =============================================================================

def naive_matmul(A, B):
    """
    朴素的三重循环矩阵乘法。
    时间复杂度: O(M * N * K)
    这是最直观但最慢的实现方式。
    """
    M = len(A)        # A 的行数
    K = len(A[0])     # A 的列数 = B 的行数
    N = len(B[0])     # B 的列数

    # 初始化结果矩阵为全零
    C = [[0.0] * N for _ in range(M)]

    # 三重循环: 逐元素计算
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i][j] += A[i][k] * B[k][j]
    return C


def numpy_matmul(A, B):
    """
    使用 NumPy 的矩阵乘法。
    底层调用 BLAS 库 (如 OpenBLAS / MKL)，经过高度优化。
    利用了 SIMD 指令、缓存优化、多线程等技术。
    """
    return np.matmul(A, B)


def torch_matmul(A, B):
    """
    使用 PyTorch 的矩阵乘法。
    在 CPU 上同样调用优化的 BLAS 库。
    在 GPU 上则会利用 Tensor Core 进行加速。
    """
    return torch.matmul(A, B)


def benchmark_matmul():
    """
    对比不同矩阵乘法实现的性能。
    模拟 Transformer 中一个线性层的计算。
    """
    print("=" * 70)
    print("第一部分: 矩阵乘法 (GEMM) 性能对比")
    print("=" * 70)
    print()

    # --- 小规模矩阵: 朴素 Python vs NumPy vs PyTorch ---
    # 小矩阵用于演示朴素实现 (大矩阵太慢)
    small_size = 128
    print(f"[实验1] 小规模矩阵乘法: {small_size}x{small_size}")
    print("-" * 50)

    # 生成随机矩阵 (模拟权重矩阵和输入)
    A_list = [[np.random.randn() for _ in range(small_size)] for _ in range(small_size)]
    B_list = [[np.random.randn() for _ in range(small_size)] for _ in range(small_size)]
    A_np = np.array(A_list, dtype=np.float32)
    B_np = np.array(B_list, dtype=np.float32)

    # 1) 朴素 Python 实现
    start = time.perf_counter()
    C_naive = naive_matmul(A_list, B_list)
    naive_time = time.perf_counter() - start
    print(f"  朴素 Python 三重循环:  {naive_time:.4f} 秒")

    # 2) NumPy 优化实现
    start = time.perf_counter()
    C_np = numpy_matmul(A_np, B_np)
    numpy_time = time.perf_counter() - start
    print(f"  NumPy (BLAS 优化):     {numpy_time:.6f} 秒")

    # 计算加速比
    speedup_np = naive_time / numpy_time if numpy_time > 0 else float('inf')
    print(f"  NumPy 加速比:          {speedup_np:.1f}x")

    # 3) PyTorch 实现
    if HAS_TORCH:
        A_torch = torch.tensor(A_np)
        B_torch = torch.tensor(B_np)

        # 预热 (第一次调用可能有额外开销)
        _ = torch_matmul(A_torch, B_torch)

        start = time.perf_counter()
        C_torch = torch_matmul(A_torch, B_torch)
        torch_time = time.perf_counter() - start
        print(f"  PyTorch (CPU):         {torch_time:.6f} 秒")

        speedup_torch = naive_time / torch_time if torch_time > 0 else float('inf')
        print(f"  PyTorch 加速比:        {speedup_torch:.1f}x")

    # 验证结果正确性: 朴素实现与 NumPy 结果对比
    C_naive_np = np.array(C_naive, dtype=np.float32)
    max_diff = np.max(np.abs(C_naive_np - C_np))
    print(f"  结果验证 (最大误差):   {max_diff:.6e}")

    print()

    # --- 大规模矩阵: 模拟真实 Transformer 线性层 ---
    # 典型 LLM 尺寸: hidden_dim=4096, FFN 中间维度=11008 (LLaMA-7B)
    M, K, N = 512, 1024, 1024  # 缩小版本，适合 CPU 演示
    print(f"[实验2] 大规模矩阵乘法: ({M}x{K}) @ ({K}x{N})")
    print(f"  模拟场景: batch_size={M}, hidden_dim={K}, output_dim={N}")
    print("-" * 50)

    A_large = np.random.randn(M, K).astype(np.float32)
    B_large = np.random.randn(K, N).astype(np.float32)

    # NumPy
    start = time.perf_counter()
    C_large_np = numpy_matmul(A_large, B_large)
    numpy_large_time = time.perf_counter() - start

    # 计算 FLOPS (浮点运算次数)
    flops = 2 * M * K * N  # 矩阵乘法的浮点运算次数 = 2*M*K*N
    gflops_np = flops / numpy_large_time / 1e9
    print(f"  NumPy:    {numpy_large_time:.4f} 秒, {gflops_np:.2f} GFLOPS")

    if HAS_TORCH:
        A_large_torch = torch.from_numpy(A_large)
        B_large_torch = torch.from_numpy(B_large)

        # 预热
        _ = torch_matmul(A_large_torch, B_large_torch)

        start = time.perf_counter()
        C_large_torch = torch_matmul(A_large_torch, B_large_torch)
        torch_large_time = time.perf_counter() - start
        gflops_torch = flops / torch_large_time / 1e9
        print(f"  PyTorch:  {torch_large_time:.4f} 秒, {gflops_torch:.2f} GFLOPS")

    # 参考数据
    print()
    print("  [参考] 真实 GPU 性能:")
    print("    NVIDIA A100 FP32:  ~19.5 TFLOPS")
    print("    NVIDIA A100 FP16:  ~312  TFLOPS (Tensor Core)")
    print("    NVIDIA H100 FP16:  ~990  TFLOPS (Tensor Core)")
    print()


# =============================================================================
# 第二部分: 张量分片 (Tensor Sharding / Tiling)
# =============================================================================

def tiled_matmul(A, B, tile_size=64):
    """
    分块矩阵乘法 (Tiled Matrix Multiplication)。

    核心思想:
    - 将大矩阵拆分成小块 (tile)，逐块计算并累加。
    - 这模拟了 GPU Tensor Core 的工作方式:
      硬件一次只能处理固定大小的矩阵块 (如 16x16)。
    - 分块还能提高缓存命中率，因为小块可以完全放入 L1/L2 缓存。

    在分布式训练中，这个概念扩展为张量并行:
    - 将权重矩阵沿行或列切分到多个 GPU 上。
    - 每个 GPU 只计算一部分，最后通过 AllReduce 汇总。
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "矩阵维度不匹配"

    C = np.zeros((M, N), dtype=A.dtype)

    # 按 tile_size 大小分块遍历
    for i in range(0, M, tile_size):
        for j in range(0, N, tile_size):
            for k in range(0, K, tile_size):
                # 提取子矩阵 (tile)
                i_end = min(i + tile_size, M)
                j_end = min(j + tile_size, N)
                k_end = min(k + tile_size, K)

                # 子矩阵乘法并累加到结果
                A_tile = A[i:i_end, k:k_end]
                B_tile = B[k:k_end, j:j_end]
                C[i:i_end, j:j_end] += np.matmul(A_tile, B_tile)

    return C


def simulate_tensor_sharding():
    """
    模拟张量分片，展示分块计算的原理和正确性。
    """
    print("=" * 70)
    print("第二部分: 张量分片 (Tensor Sharding / Tiling)")
    print("=" * 70)
    print()

    M, K, N = 512, 512, 512
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    # 完整矩阵乘法 (参考结果)
    start = time.perf_counter()
    C_full = np.matmul(A, B)
    full_time = time.perf_counter() - start

    print(f"矩阵大小: ({M}x{K}) @ ({K}x{N})")
    print(f"完整矩阵乘法: {full_time:.4f} 秒")
    print()

    # 不同分块大小的对比
    tile_sizes = [32, 64, 128, 256]
    print(f"{'分块大小':<12} {'耗时(秒)':<14} {'最大误差':<16} {'分块数量'}")
    print("-" * 60)

    for ts in tile_sizes:
        start = time.perf_counter()
        C_tiled = tiled_matmul(A, B, tile_size=ts)
        tiled_time = time.perf_counter() - start

        max_diff = np.max(np.abs(C_full - C_tiled))
        # 计算总共需要多少个分块
        num_tiles = (
            ((M + ts - 1) // ts) *
            ((N + ts - 1) // ts) *
            ((K + ts - 1) // ts)
        )
        print(f"  {ts:<10} {tiled_time:<14.4f} {max_diff:<16.6e} {num_tiles}")

    print()

    # 可视化分片过程
    print("分片示意图 (以 4x4 矩阵, tile_size=2 为例):")
    print()
    print("  原始矩阵 A (4x4):          分成 4 个 2x2 的 tile:")
    print("  ┌─────────────────┐        ┌────────┬────────┐")
    print("  │ a00 a01 a02 a03 │        │ A[0,0] │ A[0,1] │")
    print("  │ a10 a11 a12 a13 │   =>   │ (2x2)  │ (2x2)  │")
    print("  │ a20 a21 a22 a23 │        ├────────┼────────┤")
    print("  │ a30 a31 a32 a33 │        │ A[1,0] │ A[1,1] │")
    print("  └─────────────────┘        │ (2x2)  │ (2x2)  │")
    print("                             └────────┴────────┘")
    print()
    print("  C[i,j] = Σ_k A_tile[i,k] @ B_tile[k,j]")
    print("  每个 tile 的乘法可以独立计算，适合并行执行。")
    print()


# =============================================================================
# 第三部分: 精度对比 — FP16 vs FP32
# =============================================================================

def compare_precision():
    """
    对比不同浮点精度的计算速度和精度损失。
    在真实 LLM 训练中，混合精度 (Mixed Precision) 是标配技术。
    """
    print("=" * 70)
    print("第三部分: 浮点精度对比 — FP16 vs FP32")
    print("=" * 70)
    print()

    # 基本精度信息
    print("[1] 浮点数格式基础信息:")
    print("-" * 50)
    print(f"  {'格式':<10} {'位数':<8} {'指数位':<8} {'尾数位':<8} {'范围'}")
    print(f"  {'FP32':<10} {'32':<8} {'8':<8} {'23':<8} ±3.4e38")
    print(f"  {'FP16':<10} {'16':<8} {'5':<8} {'10':<8} ±6.5e4")
    print(f"  {'BF16':<10} {'16':<8} {'8':<8} {'7':<8}  ±3.4e38")
    print()

    # 精度损失演示
    print("[2] 精度损失演示:")
    print("-" * 50)

    # 用 NumPy 模拟不同精度的矩阵乘法
    M, K, N = 512, 512, 512
    A_fp32 = np.random.randn(M, K).astype(np.float32)
    B_fp32 = np.random.randn(K, N).astype(np.float32)

    # FP32 计算 (参考结果)
    C_fp32 = np.matmul(A_fp32, B_fp32)

    # FP16 计算
    A_fp16 = A_fp32.astype(np.float16)
    B_fp16 = B_fp32.astype(np.float16)
    C_fp16 = np.matmul(A_fp16, B_fp16).astype(np.float32)  # 转回 FP32 用于对比

    # 计算精度差异
    abs_diff = np.abs(C_fp32 - C_fp16)
    rel_diff = abs_diff / (np.abs(C_fp32) + 1e-8)

    print(f"  矩阵大小: ({M}x{K}) @ ({K}x{N})")
    print(f"  FP16 vs FP32 绝对误差:")
    print(f"    平均值:   {np.mean(abs_diff):.4f}")
    print(f"    最大值:   {np.max(abs_diff):.4f}")
    print(f"    中位数:   {np.median(abs_diff):.4f}")
    print(f"  FP16 vs FP32 相对误差:")
    print(f"    平均值:   {np.mean(rel_diff):.6f} ({np.mean(rel_diff)*100:.4f}%)")
    print(f"    最大值:   {np.max(rel_diff):.6f} ({np.max(rel_diff)*100:.4f}%)")
    print()

    # 性能对比
    print("[3] 计算性能对比 (NumPy, CPU):")
    print("-" * 50)

    # FP32 性能
    num_runs = 5
    times_fp32 = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = np.matmul(A_fp32, B_fp32)
        times_fp32.append(time.perf_counter() - start)
    avg_fp32 = np.mean(times_fp32)

    # FP16 性能
    times_fp16 = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = np.matmul(A_fp16, B_fp16)
        times_fp16.append(time.perf_counter() - start)
    avg_fp16 = np.mean(times_fp16)

    flops = 2 * M * K * N
    print(f"  FP32: {avg_fp32:.4f} 秒 (平均 {num_runs} 次), {flops/avg_fp32/1e9:.2f} GFLOPS")
    print(f"  FP16: {avg_fp16:.4f} 秒 (平均 {num_runs} 次), {flops/avg_fp16/1e9:.2f} GFLOPS")
    print(f"  FP16/FP32 速度比: {avg_fp32/avg_fp16:.2f}x")
    print()

    # 显存占用对比
    print("[4] 显存/内存占用对比:")
    print("-" * 50)
    mem_fp32 = A_fp32.nbytes + B_fp32.nbytes
    mem_fp16 = A_fp16.nbytes + B_fp16.nbytes
    print(f"  FP32 矩阵占用: {mem_fp32 / 1024:.1f} KB")
    print(f"  FP16 矩阵占用: {mem_fp16 / 1024:.1f} KB")
    print(f"  内存节省: {(1 - mem_fp16/mem_fp32)*100:.0f}%")
    print()

    # PyTorch 精度对比
    if HAS_TORCH:
        print("[5] PyTorch 精度对比:")
        print("-" * 50)
        A_t32 = torch.randn(M, K, dtype=torch.float32)
        B_t32 = torch.randn(K, N, dtype=torch.float32)
        A_t16 = A_t32.half()  # 转为 FP16
        B_t16 = B_t32.half()

        C_t32 = torch.matmul(A_t32, B_t32)
        C_t16 = torch.matmul(A_t16, B_t16).float()  # 转回 FP32 对比

        diff_torch = torch.abs(C_t32 - C_t16)
        print(f"  PyTorch FP16 vs FP32 绝对误差:")
        print(f"    平均值: {diff_torch.mean().item():.4f}")
        print(f"    最大值: {diff_torch.max().item():.4f}")
        print()

    # 溢出演示
    print("[6] FP16 溢出风险演示:")
    print("-" * 50)
    print(f"  FP16 最大值: {np.finfo(np.float16).max}")
    print(f"  FP32 最大值: {np.finfo(np.float32).max}")

    # 模拟大数值溢出
    large_val = np.float16(60000.0)
    result = large_val * np.float16(2.0)
    print(f"  FP16: 60000 * 2 = {result}  (溢出为 inf!)")

    small_val = np.float16(0.00001)
    print(f"  FP16: 0.00001 存储为 {small_val} (精度丢失)")
    print()
    print("  这就是为什么训练中需要 Loss Scaling 技术:")
    print("  将梯度放大后用 FP16 计算，再缩小回来，避免下溢。")
    print()


# =============================================================================
# 第四部分: 总结与真实 LLM 基础设施对比
# =============================================================================

def print_comparison_with_real_infra():
    """
    将本演示的概念映射到真实 LLM 基础设施。
    """
    print("=" * 70)
    print("总结: 本演示 vs 真实 LLM 基础设施")
    print("=" * 70)
    print()

    comparisons = [
        ("矩阵乘法 (GEMM)",
         "朴素 Python 三重循环 / NumPy BLAS",
         "GPU Tensor Core, cuBLAS, CUTLASS\n"
         "                                     "
         "单次 GEMM 可达数百 TFLOPS"),

        ("张量分片 (Tiling)",
         "NumPy 子矩阵切片 + 循环累加",
         "硬件自动 tiling (Tensor Core 16x16)\n"
         "                                     "
         "软件层: Triton / CUDA kernel 手动 tiling"),

        ("精度 (Precision)",
         "NumPy FP16/FP32 对比",
         "混合精度训练 (AMP): FP16 前向/反向\n"
         "                                     "
         "FP32 主权重, Loss Scaling, BF16"),

        ("并行计算",
         "单线程串行模拟",
         "多 GPU 张量并行: Megatron-LM 列/行切分\n"
         "                                     "
         "数据并行 + 流水线并行 (3D 并行)"),

        ("内存优化",
         "标准 Python/NumPy 内存管理",
         "显存池化, 梯度检查点, ZeRO 优化器\n"
         "                                     "
         "激活重计算, PagedAttention"),

        ("计算规模",
         f"矩阵大小 ~512x512, ~{2*512*512*512/1e6:.0f}M FLOPS",
         "LLaMA-70B 单次前向: ~140T FLOPS\n"
         "                                     "
         "训练总计算: ~数千 PetaFLOP-days"),
    ]

    for concept, demo, real in comparisons:
        print(f"  [{concept}]")
        print(f"    本演示:  {demo}")
        print(f"    真实系统: {real}")
        print()

    print("-" * 70)
    print("  核心启示:")
    print("    1. LLM 的计算核心就是矩阵乘法，理解 GEMM 就理解了计算瓶颈。")
    print("    2. 分块 (Tiling) 是从硬件到软件都在使用的通用优化策略。")
    print("    3. 低精度计算是 LLM 训练/推理加速的关键技术之一。")
    print("    4. CPU 上的演示虽然规模小，但核心原理与 GPU 上完全一致。")
    print("    5. 后续模块将在此基础上构建模型并行、通信、存储等基础设施。")
    print("-" * 70)
    print()


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         LLM 基础设施教学项目 — 模块1: 计算核心演示                ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # 第一部分: 矩阵乘法性能对比
    benchmark_matmul()

    # 第二部分: 张量分片
    simulate_tensor_sharding()

    # 第三部分: 精度对比
    compare_precision()

    # 第四部分: 总结
    print_comparison_with_real_infra()

    print("演示完毕。")
    print()
