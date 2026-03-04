"""
================================================================================
模块5: 存储架构 —— Checkpoint Manager (检查点管理器)
================================================================================

【知识总结: LLM训练中的存储系统】

1. 检查点 (Checkpoint) 的作用:
   - 训练过程中定期保存模型状态, 用于故障恢复和断点续训
   - 一个完整的检查点包含: 模型权重、优化器状态、学习率调度器、训练步数等
   - 大模型(如 LLaMA-70B)的单个检查点可达数百GB

2. 分片存储 (Sharded Checkpoint):
   - 当模型太大无法放入单个文件时, 将模型按层/参数组拆分成多个分片
   - 每个分片可以独立保存和加载, 支持并行IO
   - 实际场景: DeepSpeed ZeRO 将优化器状态分片到不同GPU/节点
   - PyTorch FSDP 原生支持 ShardedStateDictConfig

3. 预取 (Prefetch):
   - 在需要加载检查点之前, 提前异步读取到内存中
   - 避免训练主循环因IO阻塞而等待
   - 常见实现: 后台线程/进程 + 内存缓存

4. 存储后端选择:
   - 本地 NVMe SSD: 最快, 但容量有限 (~4TB/节点)
   - 分布式文件系统 (Lustre/GPFS): 高带宽共享存储, HPC集群标配
   - 对象存储 (S3/GCS/OSS): 容量无限, 适合长期归档, 延迟较高
   - HDFS: Hadoop生态, 适合大数据场景

5. 性能关键指标:
   - 写入吞吐量 (Write Throughput): 影响检查点保存耗时
   - 读取吞吐量 (Read Throughput): 影响恢复/加载速度
   - 检查点大小 (Checkpoint Size): 影响存储成本和IO时间

本模块用一个简单的 PyTorch 模型演示上述概念的核心实现。
================================================================================
"""

import os
import time
import shutil
import tempfile
import threading
from typing import Dict, Optional, Any

import torch
import torch.nn as nn


# ==============================================================================
# 工具函数
# ==============================================================================

def get_size_mb(path: str) -> float:
    """计算文件或目录的大小 (MB)"""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    # 如果是目录, 递归统计所有文件大小
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)


def format_size(size_mb: float) -> str:
    """格式化文件大小显示"""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.2f} GB"
    elif size_mb >= 1:
        return f"{size_mb:.2f} MB"
    else:
        return f"{size_mb * 1024:.2f} KB"


# ==============================================================================
# 演示用的简单模型
# ==============================================================================

def create_demo_model(hidden_size: int = 1024, num_layers: int = 6) -> nn.Module:
    """
    创建一个简单的多层线性模型, 用于演示检查点操作
    模型结构: Linear -> ReLU -> Linear -> ReLU -> ... -> Linear
    """
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(hidden_size, hidden_size))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    return model


# ==============================================================================
# CheckpointManager: 检查点管理器
# ==============================================================================

class CheckpointManager:
    """
    检查点管理器, 负责模型检查点的保存、加载、分片和预取。

    功能:
    1. save()          - 保存完整检查点 (模型 + 优化器 + 训练步数)
    2. load()          - 加载检查点并恢复状态
    3. save_sharded()  - 分片保存模型, 按层拆分到多个文件
    4. load_sharded()  - 加载并合并分片检查点
    5. async_prefetch() - 异步预取检查点到内存缓存
    """

    def __init__(self, base_dir: str):
        """
        初始化检查点管理器

        参数:
            base_dir: 检查点保存的根目录
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        # 预取缓存: 存放异步加载的检查点数据
        self._prefetch_cache: Dict[str, Any] = {}
        # 预取锁: 保证线程安全
        self._prefetch_lock = threading.Lock()

        # 性能统计
        self.stats = {
            "last_save_time": 0.0,       # 最近一次保存耗时 (秒)
            "last_load_time": 0.0,       # 最近一次加载耗时 (秒)
            "last_save_size_mb": 0.0,    # 最近一次保存的文件大小 (MB)
            "last_write_throughput": 0.0, # 写入吞吐量 (MB/s)
            "last_read_throughput": 0.0,  # 读取吞吐量 (MB/s)
        }

    # --------------------------------------------------------------------------
    # 1. 完整检查点保存
    # --------------------------------------------------------------------------
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        filename: str = "checkpoint.pt",
    ) -> str:
        """
        保存完整检查点, 包含模型权重、优化器状态和训练步数。

        参数:
            model: PyTorch 模型
            optimizer: 优化器
            step: 当前训练步数
            filename: 保存的文件名

        返回:
            保存的文件路径
        """
        save_path = os.path.join(self.base_dir, filename)

        # 构建检查点字典
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "timestamp": time.time(),
        }

        # 计时保存过程
        start = time.time()
        torch.save(checkpoint, save_path)
        elapsed = time.time() - start

        # 更新性能统计
        size_mb = get_size_mb(save_path)
        self.stats["last_save_time"] = elapsed
        self.stats["last_save_size_mb"] = size_mb
        self.stats["last_write_throughput"] = size_mb / elapsed if elapsed > 0 else 0

        return save_path

    # --------------------------------------------------------------------------
    # 2. 检查点加载
    # --------------------------------------------------------------------------
    def load(
        self,
        path: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        加载检查点并恢复模型/优化器状态。

        参数:
            path: 检查点文件路径
            model: 需要恢复权重的模型 (可选)
            optimizer: 需要恢复状态的优化器 (可选)

        返回:
            检查点字典 (包含 step、timestamp 等元数据)
        """
        # 优先从预取缓存中获取
        with self._prefetch_lock:
            if path in self._prefetch_cache:
                checkpoint = self._prefetch_cache.pop(path)
                # 缓存命中, 跳过磁盘IO
                self.stats["last_load_time"] = 0.0
                self.stats["last_read_throughput"] = float("inf")
            else:
                checkpoint = None

        # 缓存未命中, 从磁盘加载
        if checkpoint is None:
            start = time.time()
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            elapsed = time.time() - start

            size_mb = get_size_mb(path)
            self.stats["last_load_time"] = elapsed
            self.stats["last_read_throughput"] = size_mb / elapsed if elapsed > 0 else 0

        # 恢复模型权重
        if model is not None and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        # 恢复优化器状态
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    # --------------------------------------------------------------------------
    # 3. 分片保存
    # --------------------------------------------------------------------------
    def save_sharded(
        self,
        model: nn.Module,
        num_shards: int,
        shard_dir: str = "sharded_checkpoint",
    ) -> str:
        """
        将模型状态字典按层拆分成多个分片文件分别保存。

        原理:
        - 将 state_dict 的所有 key 均匀分配到 num_shards 个分片中
        - 每个分片保存为独立的 .pt 文件
        - 同时保存一个元数据文件, 记录分片信息

        参数:
            model: PyTorch 模型
            num_shards: 分片数量
            shard_dir: 分片保存的子目录名

        返回:
            分片目录的完整路径
        """
        shard_path = os.path.join(self.base_dir, shard_dir)
        os.makedirs(shard_path, exist_ok=True)

        state_dict = model.state_dict()
        keys = list(state_dict.keys())

        # 将参数 key 均匀分配到各分片
        shard_keys = [[] for _ in range(num_shards)]
        for i, key in enumerate(keys):
            shard_keys[i % num_shards].append(key)

        start = time.time()

        # 逐个分片保存
        for shard_id in range(num_shards):
            shard_state = {k: state_dict[k] for k in shard_keys[shard_id]}
            shard_file = os.path.join(shard_path, f"shard_{shard_id}.pt")
            torch.save(shard_state, shard_file)

        # 保存元数据: 记录分片数量和每个分片包含的 key
        metadata = {
            "num_shards": num_shards,
            "shard_keys": shard_keys,
        }
        torch.save(metadata, os.path.join(shard_path, "metadata.pt"))

        elapsed = time.time() - start
        size_mb = get_size_mb(shard_path)
        self.stats["last_save_time"] = elapsed
        self.stats["last_save_size_mb"] = size_mb
        self.stats["last_write_throughput"] = size_mb / elapsed if elapsed > 0 else 0

        return shard_path

    # --------------------------------------------------------------------------
    # 4. 分片加载
    # --------------------------------------------------------------------------
    def load_sharded(
        self,
        shard_dir: str,
        num_shards: int,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        加载分片检查点并合并为完整的 state_dict。

        参数:
            shard_dir: 分片目录路径
            num_shards: 分片数量
            model: 需要恢复权重的模型 (可选)

        返回:
            合并后的完整 state_dict
        """
        start = time.time()

        # 逐个分片加载并合并
        merged_state_dict = {}
        for shard_id in range(num_shards):
            shard_file = os.path.join(shard_dir, f"shard_{shard_id}.pt")
            shard_state = torch.load(shard_file, map_location="cpu", weights_only=False)
            merged_state_dict.update(shard_state)

        elapsed = time.time() - start
        size_mb = get_size_mb(shard_dir)
        self.stats["last_load_time"] = elapsed
        self.stats["last_read_throughput"] = size_mb / elapsed if elapsed > 0 else 0

        # 如果提供了模型, 直接恢复权重
        if model is not None:
            model.load_state_dict(merged_state_dict)

        return merged_state_dict

    # --------------------------------------------------------------------------
    # 5. 异步预取
    # --------------------------------------------------------------------------
    def async_prefetch(self, path: str) -> threading.Thread:
        """
        在后台线程中异步预取检查点到内存缓存。

        原理:
        - 启动一个后台线程执行磁盘读取
        - 读取完成后将数据放入缓存字典
        - 后续调用 load() 时直接从缓存获取, 避免阻塞

        参数:
            path: 检查点文件路径

        返回:
            后台加载线程对象 (可用于 join 等待完成)
        """
        def _prefetch_worker():
            """预取工作线程: 从磁盘读取检查点到内存"""
            data = torch.load(path, map_location="cpu", weights_only=False)
            with self._prefetch_lock:
                self._prefetch_cache[path] = data

        thread = threading.Thread(target=_prefetch_worker, daemon=True)
        thread.start()
        return thread

    # --------------------------------------------------------------------------
    # 性能统计输出
    # --------------------------------------------------------------------------
    def print_stats(self, label: str = ""):
        """打印最近一次操作的性能统计"""
        header = f"  性能统计 [{label}]" if label else "  性能统计"
        print(header)
        print(f"    保存耗时:     {self.stats['last_save_time']:.4f} 秒")
        print(f"    加载耗时:     {self.stats['last_load_time']:.4f} 秒")
        print(f"    检查点大小:   {format_size(self.stats['last_save_size_mb'])}")
        print(f"    写入吞吐量:   {self.stats['last_write_throughput']:.2f} MB/s")
        if self.stats["last_read_throughput"] == float("inf"):
            print(f"    读取吞吐量:   缓存命中, 无磁盘IO")
        else:
            print(f"    读取吞吐量:   {self.stats['last_read_throughput']:.2f} MB/s")


# ==============================================================================
# 演示主程序
# ==============================================================================

def main():
    print("=" * 70)
    print("  模块5: 存储架构 —— Checkpoint Manager 演示")
    print("=" * 70)

    # 创建临时目录, 演示结束后自动清理
    tmp_dir = tempfile.mkdtemp(prefix="ckpt_demo_")
    print(f"\n  临时存储目录: {tmp_dir}")

    try:
        # ------------------------------------------------------------------
        # 准备: 创建模型和优化器
        # ------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("  [准备] 创建演示模型和优化器")
        print("-" * 70)

        model = create_demo_model(hidden_size=1024, num_layers=6)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 统计模型参数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  模型结构: 6层全连接网络 (hidden_size=1024)")
        print(f"  参数量:   {num_params:,} ({num_params * 4 / 1024 / 1024:.2f} MB, float32)")

        # 模拟一步训练, 让优化器产生状态 (momentum 等)
        dummy_input = torch.randn(32, 1024)
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        print("  已执行一步前向+反向, 优化器状态已初始化")

        # ------------------------------------------------------------------
        # 演示1: 完整检查点保存与加载
        # ------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("  [演示1] 完整检查点 —— 保存与加载")
        print("-" * 70)

        manager = CheckpointManager(base_dir=tmp_dir)

        # 保存检查点
        current_step = 1000
        save_path = manager.save(model, optimizer, step=current_step)
        print(f"\n  保存检查点到: {save_path}")
        manager.print_stats("保存")

        # 创建一个新的空模型和优化器, 用于验证加载
        model_new = create_demo_model(hidden_size=1024, num_layers=6)
        optimizer_new = torch.optim.Adam(model_new.parameters(), lr=1e-3)

        # 加载检查点
        ckpt = manager.load(save_path, model=model_new, optimizer=optimizer_new)
        print(f"\n  加载检查点, 恢复到 step={ckpt['step']}")
        manager.print_stats("加载")

        # 验证加载的权重与原始权重一致
        match = all(
            torch.equal(p1, p2)
            for p1, p2 in zip(model.parameters(), model_new.parameters())
        )
        print(f"\n  权重一致性验证: {'通过' if match else '失败'}")

        # ------------------------------------------------------------------
        # 演示2: 分片检查点
        # ------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("  [演示2] 分片检查点 —— 拆分保存与合并加载")
        print("-" * 70)

        num_shards = 3
        print(f"\n  分片数量: {num_shards}")

        # 分片保存
        shard_dir = manager.save_sharded(model, num_shards=num_shards)
        print(f"  分片保存到: {shard_dir}")
        manager.print_stats("分片保存")

        # 列出分片文件
        print(f"\n  分片文件列表:")
        for fname in sorted(os.listdir(shard_dir)):
            fpath = os.path.join(shard_dir, fname)
            print(f"    {fname:25s}  {format_size(get_size_mb(fpath)):>10s}")

        # 分片加载
        model_sharded = create_demo_model(hidden_size=1024, num_layers=6)
        merged_state = manager.load_sharded(shard_dir, num_shards=num_shards, model=model_sharded)
        print(f"\n  分片加载完成, 恢复了 {len(merged_state)} 个参数张量")
        manager.print_stats("分片加载")

        # 验证分片加载的权重一致性
        match_sharded = all(
            torch.equal(p1, p2)
            for p1, p2 in zip(model.parameters(), model_sharded.parameters())
        )
        print(f"  权重一致性验证: {'通过' if match_sharded else '失败'}")

        # ------------------------------------------------------------------
        # 演示3: 异步预取
        # ------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("  [演示3] 异步预取 —— 后台加载检查点到内存")
        print("-" * 70)

        # 先保存一个检查点用于预取测试
        prefetch_path = manager.save(model, optimizer, step=2000, filename="prefetch_test.pt")
        print(f"\n  待预取文件: {prefetch_path}")

        # 启动异步预取
        print("  启动后台预取线程...")
        prefetch_start = time.time()
        thread = manager.async_prefetch(prefetch_path)

        # 模拟训练主循环在预取期间继续执行
        print("  (模拟) 训练主循环继续执行其他计算...")
        # 做一些无关的计算, 模拟训练过程
        _ = torch.randn(512, 512) @ torch.randn(512, 512)

        # 等待预取完成
        thread.join()
        prefetch_elapsed = time.time() - prefetch_start
        print(f"  预取完成, 总耗时: {prefetch_elapsed:.4f} 秒")

        # 从缓存加载 (应该几乎零延迟)
        model_prefetched = create_demo_model(hidden_size=1024, num_layers=6)
        ckpt_prefetched = manager.load(prefetch_path, model=model_prefetched)
        print(f"  从缓存加载 step={ckpt_prefetched['step']}, 无需磁盘IO")
        manager.print_stats("预取后加载")

        # 对比: 不使用预取的普通加载
        print("\n  对比: 不使用预取的普通加载...")
        model_normal = create_demo_model(hidden_size=1024, num_layers=6)
        _ = manager.load(prefetch_path, model=model_normal)
        manager.print_stats("普通加载 (无预取)")

        # ------------------------------------------------------------------
        # 演示4: 性能汇总对比
        # ------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("  [演示4] 性能汇总")
        print("-" * 70)

        # 多次保存取平均值
        save_times = []
        load_times = []
        for i in range(5):
            path_i = manager.save(model, optimizer, step=3000 + i, filename=f"bench_{i}.pt")
            save_times.append(manager.stats["last_save_time"])
            _ = manager.load(path_i, model=model_new)
            load_times.append(manager.stats["last_load_time"])

        avg_save = sum(save_times) / len(save_times)
        avg_load = sum(load_times) / len(load_times)
        ckpt_size = manager.stats["last_save_size_mb"]

        print(f"\n  基准测试 (5次平均):")
        print(f"    检查点大小:     {format_size(ckpt_size)}")
        print(f"    平均保存耗时:   {avg_save:.4f} 秒")
        print(f"    平均加载耗时:   {avg_load:.4f} 秒")
        print(f"    平均写入吞吐:   {ckpt_size / avg_save:.2f} MB/s")
        print(f"    平均读取吞吐:   {ckpt_size / avg_load:.2f} MB/s")

    finally:
        # 清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\n  已清理临时目录: {tmp_dir}")

    # ==========================================================================
    # 与真实基础设施的对比
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  与真实 LLM 训练基础设施的对比")
    print("=" * 70)
    print("""
  ┌──────────────┬──────────────────────┬────────────────────────────────┐
  │     维度     │    本模块 (教学演示)   │     真实生产环境               │
  ├──────────────┼──────────────────────┼────────────────────────────────┤
  │ 模型规模     │ ~25MB (6层线性网络)   │ 数十GB~数TB (GPT-4/LLaMA-70B) │
  │ 存储后端     │ 本地磁盘 (tmpdir)     │ Lustre/GPFS/S3/HDFS           │
  │ 分片策略     │ 按参数key均匀分配     │ 按GPU/节点拓扑+ZeRO分片       │
  │ 保存格式     │ torch.save (pickle)   │ SafeTensors / 自定义二进制     │
  │ 异步IO       │ 单线程预取            │ 多线程/多进程 + AIO/io_uring  │
  │ 一致性保证   │ 无                    │ 原子写入 + 校验和 (checksum)   │
  │ 容错机制     │ 无                    │ 多副本 + 增量检查点            │
  │ 带宽需求     │ ~100 MB/s (本地SSD)   │ 数GB/s (并行文件系统)          │
  │ 优化器状态   │ 完整保存              │ 分布式分片 (ZeRO Stage 1/2/3) │
  │ 保存频率     │ 手动触发              │ 每N步自动 + 异步写出           │
  └──────────────┴──────────────────────┴────────────────────────────────┘

  关键技术要点:
  1. SafeTensors 格式 (Hugging Face) 比 pickle 更安全, 支持零拷贝加载
  2. DeepSpeed 使用异步检查点引擎, 保存过程不阻塞训练
  3. Megatron-LM 支持分布式检查点, 每个rank只保存自己的参数分片
  4. FSDP (Fully Sharded Data Parallel) 原生支持 FULL/SHARDED/LOCAL 三种保存模式
  5. 增量检查点 (Delta Checkpoint) 只保存相对上次变化的参数, 节省IO
  6. 实际生产中通常保留最近 K 个检查点, 按策略自动清理旧文件
    """)


if __name__ == "__main__":
    main()
