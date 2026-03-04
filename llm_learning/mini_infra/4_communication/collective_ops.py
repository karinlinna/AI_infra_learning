"""
================================================================================
模块4: 集合通信操作 (Collective Communication Operations)
================================================================================

【知识总结】

1. 什么是集合通信？
   在分布式训练中，多个GPU/节点需要交换数据（梯度、参数、激活值等）。
   集合通信定义了一组标准的多对多通信原语，让所有参与者协调完成数据交换。

2. 核心通信原语：
   - Broadcast:    一个rank将数据广播给所有rank
   - All-Reduce:   所有rank贡献数据，执行规约（如求和），结果分发给所有rank
   - All-Gather:   每个rank持有一片数据，收集所有片到每个rank
   - Reduce-Scatter: 先规约再分散，每个rank只得到结果的一部分

3. 通信后端：
   - NCCL (NVIDIA Collective Communications Library):
     * NVIDIA专为GPU间通信优化的库
     * 支持NVLink、PCIe、InfiniBand等多种互连
     * 是PyTorch分布式GPU训练的默认后端
     * 自动选择最优拓扑（ring、tree等）

   - Gloo:
     * Facebook开发的开源集合通信库
     * 支持CPU和GPU，跨平台
     * PyTorch CPU分布式训练的默认后端
     * 支持TCP和共享内存传输

   - MPI (Message Passing Interface):
     * 高性能计算领域的经典标准
     * OpenMPI、MPICH等实现
     * 灵活但需要额外安装配置

4. 硬件互连：
   - NVLink:   GPU间直连，带宽可达900GB/s (NVLink 4.0)
   - NVSwitch:  全连接GPU拓扑交换机
   - InfiniBand: 节点间高速网络，HDR可达200Gb/s
   - RoCE:     基于以太网的RDMA，成本更低
   - PCIe:     通用总线，带宽相对较低

5. 通信算法：
   - Ring All-Reduce: 将数据分块在环形拓扑上传递，带宽最优
   - Tree All-Reduce: 树形拓扑，延迟更低
   - Recursive Halving-Doubling: 适合小消息
   - Bucket融合: 将多个小tensor合并通信，减少启动开销

6. 在LLM训练中的应用：
   - 数据并行:  All-Reduce同步梯度
   - 张量并行:  All-Reduce / All-Gather同步切分的计算结果
   - 流水线并行: 点对点Send/Recv传递激活值
   - ZeRO优化:  Reduce-Scatter分散梯度，All-Gather收集参数

本模块使用Python multiprocessing模拟4个rank的集合通信操作，
帮助理解每种操作的数据流动方式，无需GPU或torch.distributed。
================================================================================
"""

import multiprocessing as mp
import numpy as np
import time
import os
from multiprocessing import shared_memory


# ============================================================================
# 工具函数
# ============================================================================

def print_separator(title: str, char: str = "=", width: int = 70):
    """打印分隔线和标题"""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_rank_data(label: str, rank_data: dict):
    """打印每个rank的数据"""
    print(f"\n  [{label}]")
    for rank in sorted(rank_data.keys()):
        data = rank_data[rank]
        if isinstance(data, np.ndarray):
            print(f"    Rank {rank}: {data.tolist()}")
        else:
            print(f"    Rank {rank}: {data}")


def format_time(elapsed: float) -> str:
    """格式化耗时"""
    if elapsed < 0.001:
        return f"{elapsed * 1_000_000:.1f} us"
    elif elapsed < 1.0:
        return f"{elapsed * 1_000:.2f} ms"
    else:
        return f"{elapsed:.3f} s"


# ============================================================================
# 1. Broadcast - 广播操作
# ============================================================================
# 一个rank（root）将自己的数据发送给所有其他rank
# 典型场景：模型初始化时，rank 0加载权重后广播给所有rank

def broadcast_worker(rank: int, world_size: int, root: int,
                     shm_names: list, barrier: mp.Barrier, result_queue: mp.Queue):
    """
    广播操作的工作进程
    - root rank: 将数据写入共享内存
    - 其他rank: 从共享内存读取root的数据
    """
    # 每个rank连接到自己的共享内存段
    shm = shared_memory.SharedMemory(name=shm_names[rank])
    local_data = np.ndarray((4,), dtype=np.float64, buffer=shm.buf)

    # 记录操作前的数据
    before = local_data.copy()

    # 同步：确保所有rank都准备好
    barrier.wait()

    start_time = time.perf_counter()

    if rank == root:
        # root rank: 将自己的数据写入所有其他rank的共享内存
        for target in range(world_size):
            if target != root:
                target_shm = shared_memory.SharedMemory(name=shm_names[target])
                target_data = np.ndarray((4,), dtype=np.float64, buffer=target_shm.buf)
                target_data[:] = local_data[:]
                target_shm.close()

    # 同步：等待广播完成
    barrier.wait()

    elapsed = time.perf_counter() - start_time
    after = local_data.copy()

    result_queue.put((rank, before.tolist(), after.tolist(), elapsed))

    shm.close()


def demo_broadcast(world_size: int = 4):
    """演示Broadcast操作"""
    print_separator("1. Broadcast（广播）", "=")
    print("""
  【原理】一个rank（称为root）将数据发送给所有其他rank。
  【场景】模型初始化时，rank 0 加载预训练权重后广播给所有GPU。
    """)

    root = 0  # rank 0 作为广播源

    # 创建共享内存，模拟每个rank的本地数据
    shm_list = []
    shm_names = []
    for rank in range(world_size):
        data = np.array([rank * 10 + i for i in range(4)], dtype=np.float64)
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        buf = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        buf[:] = data[:]
        shm_list.append(shm)
        shm_names.append(shm.name)

    barrier = mp.Barrier(world_size)
    result_queue = mp.Queue()

    # 启动所有worker进程
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=broadcast_worker,
                       args=(rank, world_size, root, shm_names, barrier, result_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # 收集结果
    results = {}
    while not result_queue.empty():
        rank, before, after, elapsed = result_queue.get()
        results[rank] = (before, after, elapsed)

    # 打印结果
    before_data = {r: results[r][0] for r in results}
    after_data = {r: results[r][1] for r in results}
    total_time = max(results[r][2] for r in results)

    print_rank_data("操作前各rank数据", before_data)
    print_rank_data("操作后各rank数据（所有rank拥有root的数据）", after_data)
    print(f"\n  通信耗时: {format_time(total_time)}")

    # ASCII可视化
    print("""
  【数据流可视化 - Broadcast from Rank 0】

       Rank 0            Rank 1            Rank 2            Rank 3
      ┌──────┐          ┌──────┐          ┌──────┐          ┌──────┐
      │[0,1, │          │[10,11│          │[20,21│          │[30,31│
      │ 2,3] │          │12,13]│          │22,23]│          │32,33]│
      └──┬───┘          └──────┘          └──────┘          └──────┘
         │  ┌──────────────┘                  │                 │
         │  │  ┌─────────────────────────────┘                 │
         │  │  │  ┌────────────────────────────────────────────┘
         ▼  ▼  ▼  ▼
      ┌──────┐          ┌──────┐          ┌──────┐          ┌──────┐
      │[0,1, │ ──copy──>│[0,1, │          │[0,1, │          │[0,1, │
      │ 2,3] │          │ 2,3] │<──copy───│ 2,3] │<──copy───│ 2,3] │
      └──────┘          └──────┘          └──────┘          └──────┘

      Root发送数据 ───────────────────> 所有rank收到相同数据
    """)

    # 清理共享内存
    for shm in shm_list:
        shm.close()
        shm.unlink()


# ============================================================================
# 2. All-Reduce - 全局规约
# ============================================================================
# 所有rank贡献数据，执行规约操作（如求和），结果分发给所有rank
# 典型场景：数据并行训练中同步梯度

def allreduce_worker(rank: int, world_size: int,
                     shm_names: list, barrier: mp.Barrier, result_queue: mp.Queue):
    """
    All-Reduce操作的工作进程
    模拟Ring All-Reduce算法：
    1. Reduce-Scatter阶段：数据在环上传递并累加
    2. All-Gather阶段：将最终结果传播到所有rank
    这里简化为：所有rank读取所有数据并求和
    """
    # 连接到自己的共享内存
    shm = shared_memory.SharedMemory(name=shm_names[rank])
    local_data = np.ndarray((4,), dtype=np.float64, buffer=shm.buf)
    before = local_data.copy()

    barrier.wait()
    start_time = time.perf_counter()

    # 第一步：读取所有rank的数据并求和（模拟reduce）
    total = np.zeros(4, dtype=np.float64)
    for r in range(world_size):
        other_shm = shared_memory.SharedMemory(name=shm_names[r])
        other_data = np.ndarray((4,), dtype=np.float64, buffer=other_shm.buf)
        total += other_data
        other_shm.close()

    # 同步：确保所有rank都读完了原始数据
    barrier.wait()

    # 第二步：将求和结果写回自己的共享内存（模拟all部分）
    local_data[:] = total[:]

    barrier.wait()

    elapsed = time.perf_counter() - start_time
    after = local_data.copy()

    result_queue.put((rank, before.tolist(), after.tolist(), elapsed))
    shm.close()


def demo_allreduce(world_size: int = 4):
    """演示All-Reduce操作"""
    print_separator("2. All-Reduce（全局规约）", "=")
    print("""
  【原理】每个rank贡献本地数据，对所有数据执行规约（这里用求和），
         结果分发给每个rank。等价于 Reduce + Broadcast。
  【场景】数据并行训练中，每个GPU计算局部梯度后，All-Reduce求和得到全局梯度。
    """)

    shm_list = []
    shm_names = []
    for rank in range(world_size):
        # 每个rank有不同的"梯度"
        data = np.array([(rank + 1) * 1.0, (rank + 1) * 2.0,
                         (rank + 1) * 3.0, (rank + 1) * 4.0], dtype=np.float64)
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        buf = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        buf[:] = data[:]
        shm_list.append(shm)
        shm_names.append(shm.name)

    barrier = mp.Barrier(world_size)
    result_queue = mp.Queue()

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=allreduce_worker,
                       args=(rank, world_size, shm_names, barrier, result_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    results = {}
    while not result_queue.empty():
        rank, before, after, elapsed = result_queue.get()
        results[rank] = (before, after, elapsed)

    before_data = {r: results[r][0] for r in results}
    after_data = {r: results[r][1] for r in results}
    total_time = max(results[r][2] for r in results)

    print_rank_data("操作前各rank数据（局部梯度）", before_data)
    print_rank_data("操作后各rank数据（全局梯度 = 所有rank之和）", after_data)
    print(f"\n  通信耗时: {format_time(total_time)}")

    # ASCII可视化
    print("""
  【数据流可视化 - All-Reduce (SUM)】

    操作前:
       Rank 0: [1, 2, 3, 4]        局部梯度
       Rank 1: [2, 4, 6, 8]        局部梯度
       Rank 2: [3, 6, 9, 12]       局部梯度
       Rank 3: [4, 8, 12, 16]      局部梯度

    Ring All-Reduce 过程 (简化示意):

       Rank 0 ──────> Rank 1 ──────> Rank 2 ──────> Rank 3
         ▲                                             │
         └─────────────────────────────────────────────┘

       阶段1 - Reduce-Scatter (在环上累加):
         每一步，每个rank把自己的一块数据发给下一个rank并累加

       阶段2 - All-Gather (在环上传播结果):
         每一步，每个rank把收到的最终结果块转发给下一个rank

    操作后:
       Rank 0: [10, 20, 30, 40]    全局梯度（所有rank相同）
       Rank 1: [10, 20, 30, 40]    全局梯度（所有rank相同）
       Rank 2: [10, 20, 30, 40]    全局梯度（所有rank相同）
       Rank 3: [10, 20, 30, 40]    全局梯度（所有rank相同）
    """)

    for shm in shm_list:
        shm.close()
        shm.unlink()


# ============================================================================
# 3. All-Gather - 全局收集
# ============================================================================
# 每个rank持有数据的一片，操作后每个rank拥有所有片的完整数据
# 典型场景：ZeRO-3中，前向计算前All-Gather收集完整参数

def allgather_worker(rank: int, world_size: int,
                     input_shm_names: list, output_shm_names: list,
                     barrier: mp.Barrier, result_queue: mp.Queue):
    """
    All-Gather操作的工作进程
    每个rank有一小块数据(2个元素)，收集后每个rank拥有完整数据(8个元素)
    """
    # 读取自己的输入数据
    in_shm = shared_memory.SharedMemory(name=input_shm_names[rank])
    in_data = np.ndarray((2,), dtype=np.float64, buffer=in_shm.buf)
    before = in_data.copy()

    barrier.wait()
    start_time = time.perf_counter()

    # 连接到自己的输出共享内存
    out_shm = shared_memory.SharedMemory(name=output_shm_names[rank])
    out_data = np.ndarray((2 * world_size,), dtype=np.float64, buffer=out_shm.buf)

    # 从每个rank收集数据片段
    for r in range(world_size):
        r_shm = shared_memory.SharedMemory(name=input_shm_names[r])
        r_data = np.ndarray((2,), dtype=np.float64, buffer=r_shm.buf)
        out_data[r * 2: r * 2 + 2] = r_data[:]
        r_shm.close()

    barrier.wait()

    elapsed = time.perf_counter() - start_time
    after = out_data.copy()

    result_queue.put((rank, before.tolist(), after.tolist(), elapsed))

    in_shm.close()
    out_shm.close()


def demo_allgather(world_size: int = 4):
    """演示All-Gather操作"""
    print_separator("3. All-Gather（全局收集）", "=")
    print("""
  【原理】每个rank持有数据的一个片段，All-Gather将所有片段收集并拼接，
         让每个rank都拥有完整的数据。
  【场景】ZeRO-3优化中，参数被切分到多个GPU，前向计算前需要
         All-Gather收集完整参数；张量并行中也常用于收集切分的计算结果。
    """)

    # 每个rank持有2个元素（总共4个rank * 2 = 8个元素）
    chunk_size = 2
    input_shm_list = []
    input_shm_names = []
    output_shm_list = []
    output_shm_names = []

    for rank in range(world_size):
        # 输入：每个rank的数据片段
        data = np.array([rank * 10 + 1, rank * 10 + 2], dtype=np.float64)
        in_shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        buf = np.ndarray(data.shape, dtype=data.dtype, buffer=in_shm.buf)
        buf[:] = data[:]
        input_shm_list.append(in_shm)
        input_shm_names.append(in_shm.name)

        # 输出：每个rank的完整数据缓冲区
        out_data = np.zeros(chunk_size * world_size, dtype=np.float64)
        out_shm = shared_memory.SharedMemory(create=True, size=out_data.nbytes)
        out_buf = np.ndarray(out_data.shape, dtype=out_data.dtype, buffer=out_shm.buf)
        out_buf[:] = out_data[:]
        output_shm_list.append(out_shm)
        output_shm_names.append(out_shm.name)

    barrier = mp.Barrier(world_size)
    result_queue = mp.Queue()

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=allgather_worker,
                       args=(rank, world_size, input_shm_names, output_shm_names,
                             barrier, result_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    results = {}
    while not result_queue.empty():
        rank, before, after, elapsed = result_queue.get()
        results[rank] = (before, after, elapsed)

    before_data = {r: results[r][0] for r in results}
    after_data = {r: results[r][1] for r in results}
    total_time = max(results[r][2] for r in results)

    print_rank_data("操作前各rank数据（各自的片段）", before_data)
    print_rank_data("操作后各rank数据（收集到完整数据）", after_data)
    print(f"\n  通信耗时: {format_time(total_time)}")

    # ASCII可视化
    print("""
  【数据流可视化 - All-Gather】

    操作前（每个rank只有一个片段）:
       Rank 0: [1, 2]
       Rank 1: [11, 12]
       Rank 2: [21, 22]
       Rank 3: [31, 32]

    数据流动:
       Rank 0 ─[1,2]──────────> 所有rank
       Rank 1 ─[11,12]────────> 所有rank
       Rank 2 ─[21,22]────────> 所有rank
       Rank 3 ─[31,32]────────> 所有rank

       ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
       │ Rank 0 │  │ Rank 1 │  │ Rank 2 │  │ Rank 3 │
       │ [1,2]  │  │[11,12] │  │[21,22] │  │[31,32] │
       └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘
           │           │           │           │
           ▼           ▼           ▼           ▼
       ┌──────────────────────────────────────────────┐
       │          拼接所有片段 (Concatenate)            │
       └──────────────────────────────────────────────┘
           │           │           │           │
           ▼           ▼           ▼           ▼
       ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
       │ Rank 0 │  │ Rank 1 │  │ Rank 2 │  │ Rank 3 │
       │[1,2,   │  │[1,2,   │  │[1,2,   │  │[1,2,   │
       │ 11,12, │  │ 11,12, │  │ 11,12, │  │ 11,12, │
       │ 21,22, │  │ 21,22, │  │ 21,22, │  │ 21,22, │
       │ 31,32] │  │ 31,32] │  │ 31,32] │  │ 31,32] │
       └────────┘  └────────┘  └────────┘  └────────┘

    操作后（每个rank都拥有完整数据）
    """)

    for shm in input_shm_list + output_shm_list:
        shm.close()
        shm.unlink()


# ============================================================================
# 4. Reduce-Scatter - 规约分散
# ============================================================================
# 先对所有rank的数据执行规约（如求和），然后将结果均匀分散到各rank
# 典型场景：ZeRO优化器中分散梯度

def reduce_scatter_worker(rank: int, world_size: int,
                          input_shm_names: list, output_shm_names: list,
                          barrier: mp.Barrier, result_queue: mp.Queue):
    """
    Reduce-Scatter操作的工作进程
    每个rank有完整数据(8个元素)，规约后每个rank只保留结果的一个片段(2个元素)
    """
    chunk_size = 2

    # 读取自己的输入数据
    in_shm = shared_memory.SharedMemory(name=input_shm_names[rank])
    in_data = np.ndarray((chunk_size * world_size,), dtype=np.float64, buffer=in_shm.buf)
    before = in_data.copy()

    barrier.wait()
    start_time = time.perf_counter()

    # 对所有rank的数据求和
    total = np.zeros(chunk_size * world_size, dtype=np.float64)
    for r in range(world_size):
        r_shm = shared_memory.SharedMemory(name=input_shm_names[r])
        r_data = np.ndarray((chunk_size * world_size,), dtype=np.float64, buffer=r_shm.buf)
        total += r_data
        r_shm.close()

    # 只保留属于自己的那个片段（scatter部分）
    my_chunk = total[rank * chunk_size: (rank + 1) * chunk_size]

    # 写入输出共享内存
    out_shm = shared_memory.SharedMemory(name=output_shm_names[rank])
    out_data = np.ndarray((chunk_size,), dtype=np.float64, buffer=out_shm.buf)
    out_data[:] = my_chunk[:]

    barrier.wait()

    elapsed = time.perf_counter() - start_time
    after = my_chunk.copy()

    result_queue.put((rank, before.tolist(), after.tolist(), elapsed))

    in_shm.close()
    out_shm.close()


def demo_reduce_scatter(world_size: int = 4):
    """演示Reduce-Scatter操作"""
    print_separator("4. Reduce-Scatter（规约分散）", "=")
    print("""
  【原理】所有rank的数据先执行规约（求和），然后将结果的不同片段分散到不同rank。
         可以理解为 All-Reduce 的前半部分，或者 Reduce + Scatter。
  【场景】ZeRO-1/2中，梯度先Reduce-Scatter，每个rank只负责更新一部分参数的优化器状态，
         大幅减少显存占用。
    """)

    chunk_size = 2
    total_size = chunk_size * world_size  # 每个rank初始有8个元素

    input_shm_list = []
    input_shm_names = []
    output_shm_list = []
    output_shm_names = []

    for rank in range(world_size):
        # 每个rank有完整长度的数据（模拟完整梯度）
        data = np.array([(rank + 1) * (i + 1) for i in range(total_size)], dtype=np.float64)
        in_shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        buf = np.ndarray(data.shape, dtype=data.dtype, buffer=in_shm.buf)
        buf[:] = data[:]
        input_shm_list.append(in_shm)
        input_shm_names.append(in_shm.name)

        # 输出：每个rank只保留一个chunk
        out_data = np.zeros(chunk_size, dtype=np.float64)
        out_shm = shared_memory.SharedMemory(create=True, size=out_data.nbytes)
        out_buf = np.ndarray(out_data.shape, dtype=out_data.dtype, buffer=out_shm.buf)
        out_buf[:] = out_data[:]
        output_shm_list.append(out_shm)
        output_shm_names.append(out_shm.name)

    barrier = mp.Barrier(world_size)
    result_queue = mp.Queue()

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=reduce_scatter_worker,
                       args=(rank, world_size, input_shm_names, output_shm_names,
                             barrier, result_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    results = {}
    while not result_queue.empty():
        rank, before, after, elapsed = result_queue.get()
        results[rank] = (before, after, elapsed)

    before_data = {r: results[r][0] for r in results}
    after_data = {r: results[r][1] for r in results}
    total_time = max(results[r][2] for r in results)

    print_rank_data("操作前各rank数据（完整梯度）", before_data)
    print_rank_data("操作后各rank数据（只保留自己负责的片段之和）", after_data)
    print(f"\n  通信耗时: {format_time(total_time)}")

    # ASCII可视化
    print("""
  【数据流可视化 - Reduce-Scatter (SUM)】

    操作前（每个rank有完整的8个元素）:
       Rank 0: [1,  2,  3,  4,  5,  6,  7,  8 ]   (元素 = 1*(i+1))
       Rank 1: [2,  4,  6,  8,  10, 12, 14, 16]   (元素 = 2*(i+1))
       Rank 2: [3,  6,  9,  12, 15, 18, 21, 24]   (元素 = 3*(i+1))
       Rank 3: [4,  8,  12, 16, 20, 24, 28, 32]   (元素 = 4*(i+1))

    步骤1 - Reduce（对应位置求和）:
       Sum:    [10, 20, 30, 40, 50, 60, 70, 80]

    步骤2 - Scatter（将结果分片分配）:
       ┌────────────────────────────────────────────┐
       │  Sum = [10, 20,  30, 40,  50, 60,  70, 80] │
       │         ╰──┬──╯  ╰──┬──╯  ╰──┬──╯  ╰──┬──╯│
       │          chunk0  chunk1  chunk2  chunk3  │
       └────────────────────────────────────────────┘
                    │         │         │         │
                    ▼         ▼         ▼         ▼
                 Rank 0    Rank 1    Rank 2    Rank 3
                [10,20]   [30,40]   [50,60]   [70,80]

    操作后（每个rank只持有结果的一个片段，总显存占用减少到 1/N）
    """)

    for shm in input_shm_list + output_shm_list:
        shm.close()
        shm.unlink()


# ============================================================================
# 5. 通信操作的关系总结和对比
# ============================================================================

def print_ops_relationship():
    """打印各操作之间的关系"""
    print_separator("集合通信操作的关系", "=")
    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │                    集合通信操作关系图                             │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │   Broadcast         1 -> N       一对多广播                      │
  │   Reduce            N -> 1       多对一规约                      │
  │   All-Reduce        N -> N       = Reduce + Broadcast           │
  │   Scatter           1 -> N       一对多分发（每个rank收不同片段）   │
  │   Gather            N -> 1       多对一收集                      │
  │   All-Gather        N -> N       = Gather + Broadcast           │
  │   Reduce-Scatter    N -> N       = Reduce + Scatter             │
  │                                                                 │
  │   关键等式:                                                      │
  │   All-Reduce   = Reduce-Scatter + All-Gather                    │
  │                = Reduce + Broadcast                             │
  │                                                                 │
  ├─────────────────────────────────────────────────────────────────┤
  │                   在LLM训练中的使用                               │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │   数据并行 (DDP):                                                │
  │     反向传播后 All-Reduce 梯度 -> 同步全局梯度                     │
  │                                                                 │
  │   ZeRO Stage 1 (优化器状态切分):                                  │
  │     Reduce-Scatter 梯度 -> 每个rank更新局部参数                    │
  │     All-Gather 更新后的参数 -> 同步完整模型                        │
  │                                                                 │
  │   ZeRO Stage 2 (+ 梯度切分):                                     │
  │     Reduce-Scatter 梯度 -> 每个rank只存部分梯度                    │
  │                                                                 │
  │   ZeRO Stage 3 (+ 参数切分):                                     │
  │     前向: All-Gather 参数 -> 临时组装完整层                        │
  │     反向: All-Gather 参数 + Reduce-Scatter 梯度                   │
  │                                                                 │
  │   张量并行 (Tensor Parallel):                                    │
  │     All-Reduce 或 All-Gather 同步切分的计算结果                    │
  │                                                                 │
  │   流水线并行 (Pipeline Parallel):                                 │
  │     点对点 Send/Recv 传递激活值（不是集合通信）                     │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
    """)


# ============================================================================
# 6. 与真实基础设施的对比
# ============================================================================

def print_real_infra_comparison():
    """打印与真实基础设施的对比"""
    print_separator("模拟 vs 真实基础设施对比", "=")
    print("""
  ┌────────────────┬──────────────────────┬──────────────────────────┐
  │     维度        │   本模拟 (CPU进程)    │   真实生产环境             │
  ├────────────────┼──────────────────────┼──────────────────────────┤
  │ 通信后端        │ Python共享内存        │ NCCL / Gloo / MPI        │
  │ 硬件           │ CPU内存               │ GPU显存 + NVLink/IB      │
  │ 带宽           │ ~10 GB/s (内存)       │ NVLink: 900 GB/s         │
  │                │                      │ IB HDR: 200 Gb/s         │
  │ 延迟           │ ~1-10 us (进程间)     │ ~1-5 us (NVLink)         │
  │                │                      │ ~1-2 us (IB RDMA)        │
  │ 拓扑           │ 共享内存(扁平)         │ Ring / Tree / 自适应       │
  │ 规模           │ 4 进程 (单机)          │ 数千GPU (跨多节点)         │
  │ 优化           │ 无                    │ Kernel融合、流水线重叠      │
  │                │                      │ 通信计算重叠               │
  ├────────────────┼──────────────────────┼──────────────────────────┤
  │ Ring AR复杂度   │ O(N*M) 简化实现       │ O(2*M*(N-1)/N) 带宽最优   │
  │ 消息聚合        │ 无                    │ Bucket融合，减少启动开销   │
  │ 异步通信        │ 不支持                │ 支持异步操作+流水线         │
  └────────────────┴──────────────────────┴──────────────────────────┘

  【真实训练集群网络拓扑示例 - 如 NVIDIA DGX SuperPOD】

    节点内 (Intra-node):
    ┌──────────────────────────────────────────────┐
    │  DGX H100 节点 (8x H100 GPU)                 │
    │                                              │
    │   GPU0 ═══NVLink═══ GPU1                     │
    │    ║ ╲              ╱ ║                       │
    │   GPU2 ═══NVLink═══ GPU3   NVSwitch全连接     │
    │    ║ ╲              ╱ ║    带宽: 900 GB/s     │
    │   GPU4 ═══NVLink═══ GPU5                     │
    │    ║ ╲              ╱ ║                       │
    │   GPU6 ═══NVLink═══ GPU7                     │
    │                                              │
    └──────────────┬───────────────────────────────┘
                   │ InfiniBand (400 Gb/s x 8)
                   │
    ┌──────────────┴───────────────────────────────┐
    │           InfiniBand 交换机                    │
    │        连接数百个计算节点                       │
    └──────────────────────────────────────────────┘

  【通信带宽对比】

    Python 共享内存:  ████░░░░░░░░░░░░░░░░░░░░░░  ~10 GB/s
    PCIe Gen5:       ████████░░░░░░░░░░░░░░░░░░  ~64 GB/s
    IB NDR:          ████████████░░░░░░░░░░░░░░  ~100 GB/s (双向)
    NVLink 4.0:      ████████████████████████████  ~900 GB/s

  【关键优化技术】

    1. 通信与计算重叠 (Overlap):
       在GPU执行当前层反向传播的同时，NCCL异步发送上一层的梯度。
       计算:  [Layer 3 反向] [Layer 2 反向] [Layer 1 反向]
       通信:       [Layer 3 AllReduce] [Layer 2 AllReduce] [Layer 1 AR]

    2. 梯度Bucket融合:
       不是每个tensor单独通信，而是积攒到一定大小(如25MB)后批量通信，
       分摊通信启动开销(latency)，提高带宽利用率。

    3. 分层通信 (Hierarchical):
       节点内用NVLink (高带宽)，节点间用InfiniBand。
       先节点内Reduce，再节点间Reduce，最后节点内Broadcast。
    """)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有集合通信操作的演示"""
    print_separator("集合通信操作演示 (Collective Communication Operations)", "#", 70)
    print(f"  使用 {4} 个模拟rank (Python multiprocessing + 共享内存)")
    print(f"  进程ID: {os.getpid()}")

    world_size = 4

    # 依次演示4种集合通信操作
    demo_broadcast(world_size)
    demo_allreduce(world_size)
    demo_allgather(world_size)
    demo_reduce_scatter(world_size)

    # 打印操作关系总结
    print_ops_relationship()

    # 打印与真实基础设施的对比
    print_real_infra_comparison()

    print_separator("演示完成", "#", 70)
    print("  通过本模块，你应该理解了：")
    print("  1. Broadcast:       一对多广播数据")
    print("  2. All-Reduce:      全局规约 + 广播结果")
    print("  3. All-Gather:      收集所有片段到每个rank")
    print("  4. Reduce-Scatter:  规约后分散到各rank")
    print("  5. 这些操作在数据并行、ZeRO、张量并行中的应用")
    print("  6. 真实环境中NCCL/NVLink/InfiniBand的性能特点")
    print()


if __name__ == "__main__":
    # macOS需要使用'spawn'方式启动进程（避免fork相关问题）
    mp.set_start_method("spawn", force=True)
    main()
