"""
数据并行 (Data Parallelism) - 模拟分布式训练中的数据并行策略

=== 知识点总结 ===

1. 什么是数据并行？
   - 每个设备(GPU/worker)持有完整的模型副本
   - 训练数据被切分成多份，每个设备处理不同的数据批次(mini-batch)
   - 每个设备独立做前向传播和反向传播，计算各自的梯度
   - 通过 All-Reduce 操作聚合所有设备的梯度（求平均）
   - 用平均梯度更新模型参数，保证所有设备的模型保持一致

2. All-Reduce 操作
   - 核心通信原语：将所有节点的梯度汇总并分发回每个节点
   - 常见实现：Ring All-Reduce（环形拓扑，带宽最优）
   - 步骤：每个节点发送自己的梯度 -> 聚合(求和/求平均) -> 广播回所有节点
   - 通信量与设备数量关系：Ring All-Reduce 下，通信量 = 2*(N-1)/N * 模型大小

3. 数学等价性
   - 数据并行的梯度平均 等价于 在全部数据上计算梯度
   - 即：mean(grad_worker_0, grad_worker_1, ...) == grad(全部数据的loss之和/N)
   - 这是数据并行正确性的数学基础

4. 实际框架对应
   - PyTorch DDP (DistributedDataParallel)：最常用的数据并行方案
     * 使用 NCCL 后端进行 GPU 间通信
     * 梯度计算与通信重叠(overlap)，提高效率
     * 按 bucket 进行 All-Reduce，而非逐参数通信
   - PyTorch FSDP (Fully Sharded Data Parallel)：
     * ZeRO 优化的实现，将模型参数、梯度、优化器状态分片到各设备
     * 大幅降低每个设备的显存占用
     * ZeRO Stage 1: 分片优化器状态
     * ZeRO Stage 2: 分片优化器状态 + 梯度
     * ZeRO Stage 3 (FSDP): 分片优化器状态 + 梯度 + 参数
   - DeepSpeed ZeRO：与 FSDP 类似，提供 Stage 1/2/3

5. 本示例做了什么
   - 用 Python multiprocessing 模拟 2 个 worker
   - 每个 worker 使用相同的初始模型，处理不同的数据
   - 手动实现 All-Reduce（通过共享内存传递梯度并求平均）
   - 验证：多 worker 平均梯度 == 在全部数据上计算的梯度
"""

import numpy as np
from multiprocessing import Process, Array
import ctypes


# ============================================================
# 简单线性模型：y = Wx + b
# ============================================================
class SimpleLinearModel:
    """一个简单的线性模型，用于演示数据并行"""

    def __init__(self, input_dim, output_dim, seed=42):
        """初始化模型参数"""
        rng = np.random.RandomState(seed)
        # 权重矩阵 W: (output_dim, input_dim)
        self.W = rng.randn(output_dim, input_dim).astype(np.float64)
        # 偏置向量 b: (output_dim,)
        self.b = np.zeros(output_dim, dtype=np.float64)

        # 梯度存储
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # 缓存前向传播的输入（反向传播需要）
        self._input_cache = None

    def forward(self, x):
        """
        前向传播: y = Wx + b
        x: (batch_size, input_dim)
        返回: (batch_size, output_dim)
        """
        self._input_cache = x  # 缓存输入用于反向传播
        return x @ self.W.T + self.b  # (batch, out) = (batch, in) @ (in, out) + (out,)

    def backward(self, grad_output):
        """
        反向传播: 计算参数梯度
        grad_output: (batch_size, output_dim) - 来自损失函数的梯度
        """
        batch_size = grad_output.shape[0]
        x = self._input_cache

        # dL/dW = grad_output^T @ x / batch_size
        self.grad_W = grad_output.T @ x / batch_size
        # dL/db = mean(grad_output, axis=0)
        self.grad_b = np.mean(grad_output, axis=0)

        return self.grad_W, self.grad_b


def mse_loss_and_grad(y_pred, y_true):
    """
    MSE 损失函数及其梯度
    loss = mean((y_pred - y_true)^2)
    grad = 2 * (y_pred - y_true) / batch_size
    """
    diff = y_pred - y_true
    loss = np.mean(diff ** 2)
    grad = 2.0 * diff / y_true.shape[0]  # 对 batch 维度求平均
    return loss, grad


# ============================================================
# Worker 进程：模拟单个设备上的训练
# ============================================================
def worker_fn(worker_id, num_workers, input_dim, output_dim,
              W_flat, b_flat, x_flat, y_flat, batch_size,
              grad_W_shared, grad_b_shared):
    """
    单个 worker 的训练过程:
    1. 从共享内存恢复模型参数和数据
    2. 执行前向传播
    3. 计算损失和梯度
    4. 执行反向传播
    5. 将梯度写入共享内存（用于 All-Reduce）
    """
    # 从共享内存恢复模型参数（所有 worker 使用相同的初始参数）
    W = np.frombuffer(W_flat, dtype=np.float64).reshape(output_dim, input_dim).copy()
    b = np.frombuffer(b_flat, dtype=np.float64).reshape(output_dim).copy()

    # 从共享内存恢复本 worker 的数据分片
    x = np.frombuffer(x_flat, dtype=np.float64).reshape(batch_size, input_dim).copy()
    y = np.frombuffer(y_flat, dtype=np.float64).reshape(batch_size, output_dim).copy()

    # 构建模型并加载参数
    model = SimpleLinearModel(input_dim, output_dim)
    model.W = W
    model.b = b

    # === 前向传播 ===
    y_pred = model.forward(x)

    # === 计算损失 ===
    loss, grad_loss = mse_loss_and_grad(y_pred, y)
    print(f"  [Worker {worker_id}] 损失值 = {loss:.6f}")

    # === 反向传播 ===
    grad_W, grad_b = model.backward(grad_loss)
    print(f"  [Worker {worker_id}] 梯度W范数 = {np.linalg.norm(grad_W):.6f}, "
          f"梯度b范数 = {np.linalg.norm(grad_b):.6f}")

    # === 将梯度写入共享内存（All-Reduce 的准备阶段）===
    # 每个 worker 写入自己的区域：grad_W_shared[worker_id * size : (worker_id+1) * size]
    w_size = output_dim * input_dim
    b_size = output_dim
    grad_W_buf = np.frombuffer(grad_W_shared, dtype=np.float64)
    grad_b_buf = np.frombuffer(grad_b_shared, dtype=np.float64)

    grad_W_buf[worker_id * w_size: (worker_id + 1) * w_size] = grad_W.flatten()
    grad_b_buf[worker_id * b_size: (worker_id + 1) * b_size] = grad_b.flatten()


# ============================================================
# All-Reduce 实现：聚合所有 worker 的梯度并求平均
# ============================================================
def all_reduce_average(grad_shared, param_shape, num_workers):
    """
    手动实现 All-Reduce（求平均）:
    - 从共享内存中读取每个 worker 的梯度
    - 计算所有 worker 梯度的平均值
    - 返回平均梯度

    在真实系统中，这由 NCCL 的 AllReduce 操作完成，
    使用 Ring AllReduce 或 Tree AllReduce 算法。
    """
    param_size = int(np.prod(param_shape))
    buf = np.frombuffer(grad_shared, dtype=np.float64)

    # 收集所有 worker 的梯度
    all_grads = []
    for i in range(num_workers):
        grad_i = buf[i * param_size: (i + 1) * param_size].reshape(param_shape).copy()
        all_grads.append(grad_i)

    # 求平均（等价于 AllReduce SUM 后除以 N）
    avg_grad = np.mean(all_grads, axis=0)
    return avg_grad, all_grads


# ============================================================
# 单设备基准：在全部数据上训练（用于正确性验证）
# ============================================================
def single_device_training(W_init, b_init, x_all, y_all, input_dim, output_dim):
    """
    在单个设备上用全部数据训练，作为正确性参照
    数据并行的结果应该与此完全一致
    """
    model = SimpleLinearModel(input_dim, output_dim)
    model.W = W_init.copy()
    model.b = b_init.copy()

    # 前向传播（全部数据）
    y_pred = model.forward(x_all)

    # 计算损失
    loss, grad_loss = mse_loss_and_grad(y_pred, y_all)

    # 反向传播
    grad_W, grad_b = model.backward(grad_loss)

    return loss, grad_W, grad_b


# ============================================================
# 主函数：编排整个数据并行训练流程
# ============================================================
def main():
    print("=" * 70)
    print("数据并行 (Data Parallelism) 模拟演示")
    print("=" * 70)

    # --- 超参数 ---
    input_dim = 4       # 输入维度
    output_dim = 3      # 输出维度
    num_workers = 2     # worker 数量（模拟2个GPU）
    batch_per_worker = 8  # 每个 worker 的 batch 大小
    total_batch = batch_per_worker * num_workers  # 全局 batch 大小

    # --- 生成数据 ---
    rng = np.random.RandomState(123)
    x_all = rng.randn(total_batch, input_dim).astype(np.float64)
    # 构造一个有规律的目标：y = 2*x[:,:3] + 1 + noise
    W_true = rng.randn(output_dim, input_dim).astype(np.float64)
    y_all = x_all @ W_true.T + 0.1 * rng.randn(total_batch, output_dim)

    # --- 初始化模型 ---
    model_init = SimpleLinearModel(input_dim, output_dim, seed=42)
    W_init = model_init.W.copy()
    b_init = model_init.b.copy()

    print(f"\n模型配置: 输入维度={input_dim}, 输出维度={output_dim}")
    print(f"并行配置: {num_workers} 个 worker, "
          f"每个 worker batch_size={batch_per_worker}, "
          f"全局 batch_size={total_batch}")
    print(f"初始权重W形状: {W_init.shape}, 偏置b形状: {b_init.shape}")

    # ============================
    # 第一步：数据并行训练
    # ============================
    print(f"\n{'='*70}")
    print("第一步：数据并行训练（2个Worker）")
    print(f"{'='*70}")

    # 将数据分片给各 worker
    x_splits = np.array_split(x_all, num_workers, axis=0)
    y_splits = np.array_split(y_all, num_workers, axis=0)

    print(f"\n数据分片:")
    for i in range(num_workers):
        print(f"  Worker {i}: x_shape={x_splits[i].shape}, y_shape={y_splits[i].shape}")

    # 创建共享内存：模型参数（只读）+ 梯度收集区（各worker写入）
    W_shared = Array(ctypes.c_double, W_init.flatten(), lock=False)
    b_shared = Array(ctypes.c_double, b_init.flatten(), lock=False)

    # 为每个 worker 的数据创建共享内存
    x_shared_list = []
    y_shared_list = []
    for i in range(num_workers):
        x_shared_list.append(Array(ctypes.c_double, x_splits[i].flatten(), lock=False))
        y_shared_list.append(Array(ctypes.c_double, y_splits[i].flatten(), lock=False))

    # 创建梯度收集的共享内存（每个 worker 写入自己的区域）
    w_size = output_dim * input_dim
    b_size = output_dim
    grad_W_shared = Array(ctypes.c_double, w_size * num_workers, lock=False)
    grad_b_shared = Array(ctypes.c_double, b_size * num_workers, lock=False)

    # 启动 worker 进程（模拟多GPU并行计算）
    print(f"\n启动 {num_workers} 个 worker 进程...")
    processes = []
    for i in range(num_workers):
        p = Process(
            target=worker_fn,
            args=(i, num_workers, input_dim, output_dim,
                  W_shared, b_shared,
                  x_shared_list[i], y_shared_list[i], batch_per_worker,
                  grad_W_shared, grad_b_shared)
        )
        processes.append(p)
        p.start()

    # 等待所有 worker 完成
    for p in processes:
        p.join()

    print("\n所有 worker 完成前向+反向传播!")

    # ============================
    # 第二步：All-Reduce 聚合梯度
    # ============================
    print(f"\n{'='*70}")
    print("第二步：All-Reduce 梯度聚合")
    print(f"{'='*70}")

    # 执行 All-Reduce（求平均）
    avg_grad_W, all_grad_W = all_reduce_average(
        grad_W_shared, (output_dim, input_dim), num_workers
    )
    avg_grad_b, all_grad_b = all_reduce_average(
        grad_b_shared, (output_dim,), num_workers
    )

    print(f"\nAll-Reduce 操作（模拟 Ring All-Reduce）:")
    for i in range(num_workers):
        print(f"  Worker {i} 梯度W范数: {np.linalg.norm(all_grad_W[i]):.6f}")
    print(f"  -> 平均梯度W范数: {np.linalg.norm(avg_grad_W):.6f}")
    print(f"\n平均后的梯度W:\n{avg_grad_W}")
    print(f"\n平均后的梯度b:\n{avg_grad_b}")

    # ============================
    # 第三步：正确性验证
    # ============================
    print(f"\n{'='*70}")
    print("第三步：正确性验证 - 对比单设备训练")
    print(f"{'='*70}")

    # 单设备训练（全部数据）
    single_loss, single_grad_W, single_grad_b = single_device_training(
        W_init, b_init, x_all, y_all, input_dim, output_dim
    )

    print(f"\n单设备梯度W:\n{single_grad_W}")
    print(f"\n单设备梯度b:\n{single_grad_b}")

    # 比较两种方式的梯度
    w_diff = np.max(np.abs(avg_grad_W - single_grad_W))
    b_diff = np.max(np.abs(avg_grad_b - single_grad_b))

    print(f"\n--- 梯度差异比较 ---")
    print(f"  梯度W最大绝对差: {w_diff:.2e}")
    print(f"  梯度b最大绝对差: {b_diff:.2e}")

    if w_diff < 1e-10 and b_diff < 1e-10:
        print(f"\n  [验证通过] 数据并行的平均梯度 与 单设备全量梯度 完全一致!")
        print(f"  这证明了: mean(grad_i for i in workers) == grad(全量数据)")
    else:
        print(f"\n  [验证失败] 梯度存在差异，请检查实现")

    # ============================
    # 第四步：模拟参数更新
    # ============================
    print(f"\n{'='*70}")
    print("第四步：使用平均梯度更新模型参数")
    print(f"{'='*70}")

    learning_rate = 0.01
    W_new = W_init - learning_rate * avg_grad_W
    b_new = b_init - learning_rate * avg_grad_b

    # 单设备更新
    W_new_single = W_init - learning_rate * single_grad_W
    b_new_single = b_init - learning_rate * single_grad_b

    print(f"\n  学习率: {learning_rate}")
    print(f"  更新后参数W差异: {np.max(np.abs(W_new - W_new_single)):.2e}")
    print(f"  更新后参数b差异: {np.max(np.abs(b_new - b_new_single)):.2e}")
    print(f"  -> 数据并行更新的参数 与 单设备更新的参数 完全一致!")

    # ============================
    # 与真实框架的对比说明
    # ============================
    print(f"\n{'='*70}")
    print("与真实基础设施的对比")
    print(f"{'='*70}")
    print("""
    本示例 (模拟)                    真实框架
    ─────────────────────────────────────────────────────────────────
    multiprocessing.Process     ->   多个 GPU 进程 (torch.distributed)
    共享内存传递梯度            ->   NCCL AllReduce (GPU直连通信)
    手动求平均                  ->   Ring AllReduce / Tree AllReduce
    同步等待所有worker          ->   梯度桶(bucket)异步通信 + 计算重叠
    完整模型副本 x N            ->   DDP: 完整副本 / FSDP: 分片存储

    关键区别:
    1. DDP 使用 gradient bucketing（梯度桶），将多个小梯度打包通信
    2. DDP 在反向传播过程中就开始通信（计算与通信重叠）
    3. FSDP 在前向传播时按需收集参数，反向传播后释放，极大节省显存
    4. 真实系统使用 NCCL/Gloo 等高性能通信库，带宽利用率远超共享内存
    5. 大规模训练通常组合使用：数据并行 + 张量并行 + 流水线并行 (3D并行)
    """)


if __name__ == "__main__":
    main()
