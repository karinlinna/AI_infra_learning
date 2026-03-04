"""
张量并行 (Tensor Parallelism) - 模拟 Megatron-LM 风格的张量并行

=== 知识点总结 ===

1. 什么是张量并行？
   - 将单个算子（如线性层的权重矩阵）切分到多个设备上
   - 每个设备只持有权重的一部分，计算也只做一部分
   - 通过通信操作（AllReduce / AllGather）合并各设备的部分结果
   - 目标：单个层太大放不下一个 GPU 时，拆分到多个 GPU

2. Megatron-LM 的两种切分方式

   (a) 列并行 (Column Parallel Linear):
       - 权重矩阵 W 按列切分: W = [W1 | W2]  (每个设备一列分片)
       - 输入 X 不切分（每个设备都有完整输入）
       - 每个设备计算: Y_i = X @ W_i  (得到输出的一部分列)
       - 合并方式: Y = [Y1 | Y2]  (沿列拼接，即 AllGather)
       - 适用场景: MLP 的第一个线性层、注意力机制的 QKV 投影

   (b) 行并行 (Row Parallel Linear):
       - 权重矩阵 W 按行切分: W = [W1; W2]  (每个设备一行分片)
       - 输入 X 按列切分: X = [X1 | X2]  (每个设备持有输入的一部分)
       - 每个设备计算: Y_i = X_i @ W_i  (得到部分求和结果)
       - 合并方式: Y = Y1 + Y2  (AllReduce 求和)
       - 适用场景: MLP 的第二个线性层、注意力机制的输出投影

3. Megatron-LM 中 MLP 的张量并行设计
   - MLP: Y = GeLU(X @ A) @ B
   - A 用列并行切分 -> 每个设备独立计算 GeLU (无需通信)
   - B 用行并行切分 -> AllReduce 得到最终输出
   - 精妙之处：列并行的输出正好是行并行的输入，中间无需额外通信!
   - 一个完整 MLP 块只需要 1 次 AllReduce (在行并行的输出处)

4. 通信分析
   - 列并行: 前向 AllGather (或无需通信如果后接行并行), 反向 AllReduce
   - 行并行: 前向 AllReduce, 反向 AllGather (或无需通信如果前接列并行)
   - Megatron-LM MLP: 前向 1 次 AllReduce, 反向 1 次 AllReduce
   - Megatron-LM Attention: 前向 1 次 AllReduce, 反向 1 次 AllReduce

5. 实际框架对应
   - Megatron-LM: NVIDIA 的大模型训练框架，首创张量并行方案
   - Megatron-Core: Megatron-LM 的核心库，提供 ColumnParallelLinear, RowParallelLinear
   - PyTorch FSDP + TP: PyTorch 原生支持的张量并行 (torch.distributed.tensor)
   - DeepSpeed: 通过 Inference 引擎支持张量并行

6. 本示例做了什么
   - 用 NumPy 模拟 2 个设备上的列并行和行并行线性层
   - 手动切分权重矩阵并分配到"设备"
   - 验证：切分计算的结果 == 完整矩阵的计算结果
   - 演示 Megatron 风格 MLP 的完整流程
"""

import numpy as np


# ============================================================
# 列并行线性层 (Column Parallel Linear)
# ============================================================
class ColumnParallelLinear:
    """
    列并行线性层: Y = X @ W + b
    将权重 W 按列切分到多个设备上

    完整权重 W: (input_dim, output_dim)
    设备 i 的权重 W_i: (input_dim, output_dim // num_devices)

    每个设备计算输出的一部分列，最后拼接得到完整输出
    """

    def __init__(self, W_full, b_full, num_devices):
        """
        初始化：将完整权重按列切分
        W_full: (input_dim, output_dim) - 完整权重矩阵
        b_full: (output_dim,) - 完整偏置向量
        num_devices: 设备数量
        """
        self.num_devices = num_devices
        self.input_dim = W_full.shape[0]
        self.output_dim = W_full.shape[1]

        # 按列切分权重: 每个设备得到 output_dim // num_devices 列
        self.W_shards = np.array_split(W_full, num_devices, axis=1)
        self.b_shards = np.array_split(b_full, num_devices)

        print(f"  [列并行] 完整权重形状: {W_full.shape}")
        for i, ws in enumerate(self.W_shards):
            print(f"  [列并行] 设备 {i} 权重形状: {ws.shape}")

    def forward(self, x):
        """
        前向传播:
        1. 每个设备用自己的权重分片计算部分输出
        2. 拼接所有设备的输出（AllGather 操作）
        """
        # 每个"设备"独立计算自己的部分
        partial_outputs = []
        for i in range(self.num_devices):
            # Y_i = X @ W_i + b_i
            y_i = x @ self.W_shards[i] + self.b_shards[i]
            partial_outputs.append(y_i)
            print(f"    设备 {i}: 输入 {x.shape} @ 权重 {self.W_shards[i].shape} "
                  f"-> 输出 {y_i.shape}")

        # AllGather: 沿列拼接所有设备的输出
        y_combined = np.concatenate(partial_outputs, axis=-1)
        print(f"    AllGather 后输出形状: {y_combined.shape}")

        return y_combined, partial_outputs


# ============================================================
# 行并行线性层 (Row Parallel Linear)
# ============================================================
class RowParallelLinear:
    """
    行并行线性层: Y = X @ W + b
    将权重 W 按行切分到多个设备上

    完整权重 W: (input_dim, output_dim)
    设备 i 的权重 W_i: (input_dim // num_devices, output_dim)

    输入也需要按列切分，每个设备计算部分结果，最后 AllReduce 求和
    """

    def __init__(self, W_full, b_full, num_devices):
        """
        初始化：将完整权重按行切分
        W_full: (input_dim, output_dim) - 完整权重矩阵
        b_full: (output_dim,) - 完整偏置向量
        num_devices: 设备数量
        """
        self.num_devices = num_devices
        self.input_dim = W_full.shape[0]
        self.output_dim = W_full.shape[1]

        # 按行切分权重: 每个设备得到 input_dim // num_devices 行
        self.W_shards = np.array_split(W_full, num_devices, axis=0)
        # 偏置不切分，只在一个设备上加（或每个设备加 b/N 后 AllReduce）
        self.b_full = b_full

        print(f"  [行并行] 完整权重形状: {W_full.shape}")
        for i, ws in enumerate(self.W_shards):
            print(f"  [行并行] 设备 {i} 权重形状: {ws.shape}")

    def forward(self, x_shards):
        """
        前向传播:
        1. 每个设备用自己的输入分片和权重分片计算部分结果
        2. AllReduce 求和得到最终输出

        x_shards: list，每个元素是一个设备上的输入分片
        """
        # 每个"设备"独立计算部分结果
        partial_outputs = []
        for i in range(self.num_devices):
            # Y_i = X_i @ W_i （部分矩阵乘法结果）
            y_i = x_shards[i] @ self.W_shards[i]
            partial_outputs.append(y_i)
            print(f"    设备 {i}: 输入 {x_shards[i].shape} @ 权重 {self.W_shards[i].shape} "
                  f"-> 部分输出 {y_i.shape}")

        # AllReduce: 求和所有设备的部分结果 + 加偏置
        y_combined = sum(partial_outputs) + self.b_full
        print(f"    AllReduce (求和) 后输出形状: {y_combined.shape}")

        return y_combined, partial_outputs


# ============================================================
# GeLU 激活函数
# ============================================================
def gelu(x):
    """GeLU 激活函数的近似实现"""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


# ============================================================
# Megatron-LM 风格的 MLP 张量并行
# ============================================================
class MegatronParallelMLP:
    """
    Megatron-LM 风格的 MLP 张量并行:
    Y = GeLU(X @ A) @ B

    A 使用列并行 -> GeLU 各设备独立计算(无需通信) -> B 使用行并行
    整个 MLP 前向传播只需要 1 次 AllReduce!
    """

    def __init__(self, W_A, b_A, W_B, b_B, num_devices):
        """
        初始化 Megatron MLP
        W_A: 第一层权重 (input_dim, hidden_dim)
        b_A: 第一层偏置 (hidden_dim,)
        W_B: 第二层权重 (hidden_dim, output_dim)
        b_B: 第二层偏置 (output_dim,)
        """
        self.num_devices = num_devices

        # A 层：列并行 (按列切分隐藏层维度)
        self.W_A_shards = np.array_split(W_A, num_devices, axis=1)
        self.b_A_shards = np.array_split(b_A, num_devices)

        # B 层：行并行 (按行切分，行数 = 隐藏层维度)
        self.W_B_shards = np.array_split(W_B, num_devices, axis=0)
        self.b_B = b_B

    def forward(self, x):
        """
        前向传播流程:
        1. 列并行: 每个设备计算 X @ A_i + b_A_i -> 得到隐藏层的一部分
        2. GeLU: 每个设备独立计算 GeLU（因为 GeLU 是逐元素操作，不需要其他设备的数据!）
        3. 行并行: 每个设备计算 GeLU_output_i @ B_i -> 得到部分结果
        4. AllReduce: 求和得到最终输出
        """
        device_outputs = []

        for i in range(self.num_devices):
            # 列并行的前向: Y_i = X @ A_i + b_A_i
            hidden_i = x @ self.W_A_shards[i] + self.b_A_shards[i]

            # GeLU 激活（各设备独立，无需通信!）
            activated_i = gelu(hidden_i)

            # 行并行的前向: Z_i = activated_i @ B_i
            output_i = activated_i @ self.W_B_shards[i]
            device_outputs.append(output_i)

        # 唯一的通信操作: AllReduce 求和
        final_output = sum(device_outputs) + self.b_B
        return final_output, device_outputs


# ============================================================
# 验证函数
# ============================================================
def verify_results(name, sharded_result, full_result, tolerance=1e-10):
    """验证切分计算的结果是否与完整计算一致"""
    max_diff = np.max(np.abs(sharded_result - full_result))
    match = max_diff < tolerance
    status = "通过" if match else "失败"
    print(f"\n  [{status}] {name}")
    print(f"    最大绝对差: {max_diff:.2e}")
    if not match:
        print(f"    期望结果:\n    {full_result}")
        print(f"    实际结果:\n    {sharded_result}")
    return match


# ============================================================
# 主函数
# ============================================================
def main():
    np.random.seed(42)
    num_devices = 2  # 模拟 2 个设备

    print("=" * 70)
    print("张量并行 (Tensor Parallelism) 模拟演示 - Megatron-LM 风格")
    print("=" * 70)

    # ============================
    # 实验 1: 列并行线性层
    # ============================
    print(f"\n{'='*70}")
    print("实验 1: 列并行线性层 (Column Parallel Linear)")
    print(f"{'='*70}")
    print("\n原理: Y = X @ W, 将 W 按列切分为 [W1 | W2]")
    print("每个设备计算 Y_i = X @ W_i, 然后 AllGather 拼接\n")

    # 完整权重
    input_dim, output_dim = 4, 6  # output_dim 能被 num_devices 整除
    W_col = np.random.randn(input_dim, output_dim)
    b_col = np.random.randn(output_dim)

    # 输入数据
    batch_size = 3
    x = np.random.randn(batch_size, input_dim)

    # 完整计算（参考结果）
    y_full = x @ W_col + b_col
    print(f"完整计算: {x.shape} @ {W_col.shape} -> {y_full.shape}")

    # 列并行计算
    print(f"\n列并行计算 ({num_devices} 个设备):")
    col_parallel = ColumnParallelLinear(W_col, b_col, num_devices)
    y_sharded, _ = col_parallel.forward(x)

    # 验证
    verify_results("列并行 vs 完整计算", y_sharded, y_full)

    # ============================
    # 实验 2: 行并行线性层
    # ============================
    print(f"\n{'='*70}")
    print("实验 2: 行并行线性层 (Row Parallel Linear)")
    print(f"{'='*70}")
    print("\n原理: Y = X @ W, 将 W 按行切分为 [W1; W2], X 按列切分为 [X1 | X2]")
    print("每个设备计算 Y_i = X_i @ W_i, 然后 AllReduce 求和\n")

    # 完整权重
    input_dim_row, output_dim_row = 6, 4  # input_dim 能被 num_devices 整除
    W_row = np.random.randn(input_dim_row, output_dim_row)
    b_row = np.random.randn(output_dim_row)

    # 输入数据 (这里使用上一步列并行的输出作为输入，模拟 Megatron 的设计)
    x_row = np.random.randn(batch_size, input_dim_row)

    # 完整计算
    y_full_row = x_row @ W_row + b_row
    print(f"完整计算: {x_row.shape} @ {W_row.shape} -> {y_full_row.shape}")

    # 行并行计算
    print(f"\n行并行计算 ({num_devices} 个设备):")
    row_parallel = RowParallelLinear(W_row, b_row, num_devices)

    # 将输入按列切分（模拟从列并行层接收的分片输出）
    x_row_shards = np.array_split(x_row, num_devices, axis=-1)
    print(f"\n  输入切分:")
    for i, xs in enumerate(x_row_shards):
        print(f"    设备 {i} 输入形状: {xs.shape}")

    y_sharded_row, _ = row_parallel.forward(x_row_shards)

    # 验证
    verify_results("行并行 vs 完整计算", y_sharded_row, y_full_row)

    # ============================
    # 实验 3: Megatron-LM MLP 张量并行
    # ============================
    print(f"\n{'='*70}")
    print("实验 3: Megatron-LM 风格 MLP 张量并行")
    print(f"{'='*70}")
    print("""
    MLP 结构: Y = GeLU(X @ A) @ B

    张量并行拆分策略:
    ┌─────────────────────────────────────────────────────┐
    │  X ──┬── [X @ A1] ── GeLU ── [GeLU_out1 @ B1] ──┬── AllReduce ── Y  │
    │      └── [X @ A2] ── GeLU ── [GeLU_out2 @ B2] ──┘                    │
    │      列并行(A)       各设备独立     行并行(B)       唯一通信点         │
    └─────────────────────────────────────────────────────┘
    """)

    # MLP 维度
    mlp_input_dim = 8
    mlp_hidden_dim = 12  # 必须能被 num_devices 整除
    mlp_output_dim = 8

    # 初始化权重
    W_A = np.random.randn(mlp_input_dim, mlp_hidden_dim)
    b_A = np.random.randn(mlp_hidden_dim)
    W_B = np.random.randn(mlp_hidden_dim, mlp_output_dim)
    b_B = np.random.randn(mlp_output_dim)

    # 输入数据
    x_mlp = np.random.randn(batch_size, mlp_input_dim)

    # 完整 MLP 计算（参考结果）
    hidden_full = gelu(x_mlp @ W_A + b_A)
    y_mlp_full = hidden_full @ W_B + b_B
    print(f"完整 MLP 计算:")
    print(f"  输入: {x_mlp.shape}")
    print(f"  隐藏层 (GeLU后): {hidden_full.shape}")
    print(f"  输出: {y_mlp_full.shape}")

    # Megatron 张量并行 MLP
    print(f"\nMegatron 张量并行 MLP ({num_devices} 个设备):")
    megatron_mlp = MegatronParallelMLP(W_A, b_A, W_B, b_B, num_devices)
    y_mlp_sharded, device_outputs = megatron_mlp.forward(x_mlp)

    print(f"\n  各设备的部分结果:")
    for i, do in enumerate(device_outputs):
        print(f"    设备 {i} 输出形状: {do.shape}, 范数: {np.linalg.norm(do):.4f}")
    print(f"  AllReduce 后最终输出形状: {y_mlp_sharded.shape}")

    # 验证
    verify_results("Megatron MLP 张量并行 vs 完整计算", y_mlp_sharded, y_mlp_full)

    # ============================
    # 通信量分析
    # ============================
    print(f"\n{'='*70}")
    print("通信量分析")
    print(f"{'='*70}")

    # 以 Megatron MLP 为例
    # 前向传播: 1次 AllReduce, 数据量 = batch_size * output_dim
    # 反向传播: 1次 AllReduce, 数据量 = batch_size * input_dim
    fwd_comm = batch_size * mlp_output_dim * 8  # 8 bytes per float64
    bwd_comm = batch_size * mlp_input_dim * 8

    print(f"\n  Megatron MLP 通信量 (本示例):")
    print(f"    前向传播 AllReduce: {batch_size} x {mlp_output_dim} x 8B = {fwd_comm} bytes")
    print(f"    反向传播 AllReduce: {batch_size} x {mlp_input_dim} x 8B = {bwd_comm} bytes")
    print(f"    每个 MLP 块总通信: {fwd_comm + bwd_comm} bytes")
    print(f"\n  对比：如果不用张量并行，完整权重通信量:")
    full_param_bytes = (mlp_input_dim * mlp_hidden_dim + mlp_hidden_dim * mlp_output_dim) * 8
    print(f"    A 权重: {mlp_input_dim}x{mlp_hidden_dim}x8B = {mlp_input_dim * mlp_hidden_dim * 8} bytes")
    print(f"    B 权重: {mlp_hidden_dim}x{mlp_output_dim}x8B = {mlp_hidden_dim * mlp_output_dim * 8} bytes")

    # ============================
    # 与真实框架的对比说明
    # ============================
    print(f"\n{'='*70}")
    print("与真实基础设施的对比")
    print(f"{'='*70}")
    print("""
    本示例 (模拟)                        真实框架
    ─────────────────────────────────────────────────────────────────
    np.array_split 切分权重          ->  Megatron-LM ColumnParallelLinear
    np.concatenate 拼接输出          ->  NCCL AllGather
    sum() 聚合部分结果               ->  NCCL AllReduce
    顺序模拟各设备计算               ->  各 GPU 真正并行计算
    NumPy float64                    ->  CUDA float16/bfloat16 + 混合精度

    关键区别:
    1. 真实张量并行需要高带宽互连(NVLink/NVSwitch)，因为通信在每一层发生
    2. 张量并行通常限制在单机内(8 GPU)，跨机使用数据并行或流水线并行
    3. Megatron-LM 还对 Attention 层做了张量并行:
       - Q/K/V 投影: 按 head 维度切分（列并行）
       - 输出投影: 行并行
       - 每个 Attention 块也只需要 1 次 AllReduce
    4. 序列并行(Sequence Parallelism): 对 LayerNorm/Dropout 也做并行
       - 在非张量并行的部分按序列维度切分
       - 与张量并行配合，进一步减少显存
    """)

    # ============================
    # 总结
    # ============================
    print(f"\n{'='*70}")
    print("总结: 所有验证结果")
    print(f"{'='*70}")
    all_passed = True
    all_passed &= verify_results("列并行线性层", y_sharded, y_full)
    all_passed &= verify_results("行并行线性层", y_sharded_row, y_full_row)
    all_passed &= verify_results("Megatron MLP", y_mlp_sharded, y_mlp_full)

    if all_passed:
        print(f"\n  所有验证均通过! 张量并行的切分计算与完整计算完全等价。")


if __name__ == "__main__":
    main()
