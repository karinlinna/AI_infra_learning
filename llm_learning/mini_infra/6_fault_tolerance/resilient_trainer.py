"""
================================================================================
模块6: 容错训练 (Fault-Tolerant Training)
================================================================================

【知识总结】大规模训练中的容错机制

1. 检查点恢复 (Checkpoint Recovery)
   - 大规模训练任务往往持续数天甚至数周，节点故障不可避免
   - 定期保存模型权重、优化器状态、学习率调度器状态、随机数种子等
   - 故障发生后从最近的检查点恢复，避免从头开始训练
   - 关键挑战：检查点文件可能非常大（数百GB），保存/加载需要优化
   - 常用策略：异步检查点保存、分布式检查点（每个rank只保存自己的分片）

2. 弹性训练 (Elastic Training)
   - 传统分布式训练要求固定数量的节点，一个节点挂掉全部停止
   - 弹性训练允许动态增减节点数量，不中断训练过程
   - PyTorch Elastic (torchelastic / torchrun) 是主要实现
   - 核心机制：rendezvous协议，节点发现与同步
   - 节点故障时自动重启，新节点加入时自动重新分配数据

3. 梯度检查点 (Gradient Checkpointing / Activation Checkpointing)
   - 不属于"故障容错"，而是"内存容错"——用计算换内存
   - 前向传播时不保存所有中间激活值，反向传播时重新计算
   - 可将内存消耗从 O(N) 降低到 O(sqrt(N))，N为层数
   - 适用于超大模型训练（如GPT-3/4），是必不可少的技术
   - PyTorch 提供 torch.utils.checkpoint.checkpoint() 函数

4. 工业级容错实践
   - Megatron-LM：多级检查点策略（本地SSD + 远程存储），异步保存
   - DeepSpeed：ZeRO优化器自带分布式检查点，支持弹性训练
   - Google Gemini/PaLM：自研容错框架，支持数千GPU同时训练
   - 故障检测：心跳机制、NCCL超时检测、GPU ECC错误监控
   - 故障预测：基于历史数据预测可能故障的节点，提前迁移任务

本模块通过模拟节点故障、检查点保存/恢复、梯度检查点等机制，
展示大规模训练中容错的基本原理。
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
import os
import json
import time
import random
import tempfile
import shutil
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ==============================================================================
# 简单模型定义: 带梯度检查点的4层MLP
# ==============================================================================

class SimpleMLP(nn.Module):
    """
    简单的4层MLP模型，支持梯度检查点
    用于演示容错训练，不依赖外部模块
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        # 第1层: 输入投影
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        # 第2层: 中间变换
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        # 第3层: 深层特征
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        # 第4层: 输出投影
        self.layer4 = nn.Linear(hidden_dim, output_dim)

        # 是否启用梯度检查点（用计算换内存）
        self.use_gradient_checkpoint = False

    def _forward_layer(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """单层前向传播，供梯度检查点调用"""
        return layer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gradient_checkpoint:
            # 使用梯度检查点：前向传播时不保存中间激活值
            # 反向传播时会重新计算，节省内存但增加计算量
            # use_reentrant=False 是推荐的新接口
            x = gradient_checkpoint(self._forward_layer, self.layer1, x, use_reentrant=False)
            x = gradient_checkpoint(self._forward_layer, self.layer2, x, use_reentrant=False)
            x = gradient_checkpoint(self._forward_layer, self.layer3, x, use_reentrant=False)
        else:
            # 普通前向传播：保存所有中间激活值
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

        # 最后一层不需要梯度检查点（输出层很小）
        x = self.layer4(x)
        return x


# ==============================================================================
# 合成数据生成器
# ==============================================================================

class SyntheticDataGenerator:
    """
    生成随机训练数据
    模拟一个简单的分类任务
    """

    def __init__(self, input_dim: int = 64, num_classes: int = 10,
                 batch_size: int = 32, seed: int = 42):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成一个随机批次的训练数据"""
        x = torch.randn(self.batch_size, self.input_dim, generator=self.rng)
        y = torch.randint(0, self.num_classes, (self.batch_size,), generator=self.rng)
        return x, y


# ==============================================================================
# 模拟节点故障
# ==============================================================================

class NodeFailureSimulator:
    """
    模拟分布式训练中的节点故障
    在真实场景中，故障可能来自：硬件错误、网络中断、OOM、GPU掉卡等
    """

    def __init__(self, failure_rate: float = 0.20, seed: int = 123):
        """
        failure_rate: 每个训练步骤发生故障的概率 (0.0~1.0)
        """
        self.failure_rate = failure_rate
        self.rng = random.Random(seed)
        # 记录故障历史
        self.failure_log: List[Dict] = []

    def maybe_fail(self, step: int) -> None:
        """
        以一定概率抛出异常，模拟节点故障
        在真实系统中，这些异常可能来自NCCL超时、CUDA错误等
        """
        if self.rng.random() < self.failure_rate:
            # 随机选择故障类型
            failure_types = [
                ("GPU_ECC_ERROR", "模拟GPU ECC内存错误"),
                ("NCCL_TIMEOUT", "模拟NCCL通信超时"),
                ("OOM_KILLED", "模拟进程被OOM Killer终止"),
                ("NODE_CRASH", "模拟节点意外宕机"),
            ]
            fault_type, fault_desc = self.rng.choice(failure_types)

            self.failure_log.append({
                "step": step,
                "type": fault_type,
                "desc": fault_desc,
                "time": time.time(),
            })

            raise RuntimeError(f"[模拟故障] step={step} | {fault_type}: {fault_desc}")


# ==============================================================================
# 检查点管理器
# ==============================================================================

class CheckpointManager:
    """
    检查点管理器：负责保存和加载训练状态
    在真实系统中，检查点可能保存到本地SSD、NFS、S3等

    保存内容包括:
    - 模型权重 (model_state_dict)
    - 优化器状态 (optimizer_state_dict)
    - 当前训练步数 (step)
    - 损失值历史 (loss_history)
    - 随机数状态 (rng_state)
    """

    def __init__(self, checkpoint_dir: str, save_interval: int = 5):
        """
        checkpoint_dir: 检查点保存目录
        save_interval: 每隔多少步保存一次检查点
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        os.makedirs(checkpoint_dir, exist_ok=True)
        # 记录检查点操作日志
        self.save_log: List[Dict] = []

    def _checkpoint_path(self, step: int) -> str:
        """生成检查点文件路径"""
        return os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.pt")

    def _meta_path(self) -> str:
        """元信息文件路径，记录最新的检查点位置"""
        return os.path.join(self.checkpoint_dir, "latest_checkpoint.json")

    def should_save(self, step: int) -> bool:
        """判断当前步是否需要保存检查点"""
        return step > 0 and step % self.save_interval == 0

    def save(self, model: nn.Module, optimizer: optim.Optimizer,
             step: int, loss_history: List[float]) -> str:
        """
        保存检查点
        在真实系统中，这一步可能需要数分钟（模型参数数百GB）
        可以使用异步保存来避免阻塞训练
        """
        ckpt_path = self._checkpoint_path(step)

        # 构造检查点内容
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "loss_history": loss_history,
            "torch_rng_state": torch.random.get_rng_state(),
        }

        # 保存到磁盘
        torch.save(checkpoint_data, ckpt_path)

        # 更新元信息（指向最新检查点）
        meta = {"latest_step": step, "path": ckpt_path}
        with open(self._meta_path(), "w") as f:
            json.dump(meta, f)

        self.save_log.append({
            "step": step,
            "path": ckpt_path,
            "time": time.time(),
        })

        return ckpt_path

    def load_latest(self, model: nn.Module, optimizer: optim.Optimizer
                    ) -> Optional[Dict]:
        """
        加载最新的检查点
        返回恢复的状态信息，如果没有检查点则返回None
        """
        meta_path = self._meta_path()
        if not os.path.exists(meta_path):
            return None

        with open(meta_path, "r") as f:
            meta = json.load(f)

        ckpt_path = meta["path"]
        if not os.path.exists(ckpt_path):
            return None

        # 加载检查点数据
        checkpoint_data = torch.load(ckpt_path, weights_only=False)

        # 恢复模型和优化器状态
        model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        # 恢复随机数状态（保证可复现性）
        torch.random.set_rng_state(checkpoint_data["torch_rng_state"])

        return {
            "step": checkpoint_data["step"],
            "loss_history": checkpoint_data["loss_history"],
        }


# ==============================================================================
# 训练统计追踪器
# ==============================================================================

@dataclass
class TrainingStats:
    """追踪训练过程中的各项统计指标"""
    total_failures: int = 0              # 总故障次数
    successful_recoveries: int = 0       # 成功恢复次数
    total_steps_completed: int = 0       # 完成的训练步数
    total_steps_attempted: int = 0       # 尝试的训练步数（含失败）
    loss_history: List[float] = field(default_factory=list)  # 损失值历史
    failure_events: List[Dict] = field(default_factory=list)  # 故障事件记录
    recovery_events: List[Dict] = field(default_factory=list)  # 恢复事件记录
    checkpoint_events: List[Dict] = field(default_factory=list)  # 检查点事件记录


# ==============================================================================
# 核心: 容错训练器
# ==============================================================================

class ResilientTrainer:
    """
    容错训练器：演示大规模训练中的容错机制

    核心功能:
    1. 自动检查点保存与恢复
    2. 故障模拟与自动重试
    3. 梯度检查点（内存优化）
    4. 完整的训练统计追踪
    """

    def __init__(
        self,
        target_steps: int = 20,
        checkpoint_interval: int = 5,
        failure_rate: float = 0.20,
        max_retries_per_step: int = 5,
        use_gradient_checkpoint: bool = True,
        learning_rate: float = 1e-3,
    ):
        """
        target_steps: 目标训练步数
        checkpoint_interval: 检查点保存间隔
        failure_rate: 每步故障概率
        max_retries_per_step: 单步最大重试次数
        use_gradient_checkpoint: 是否使用梯度检查点
        learning_rate: 学习率
        """
        self.target_steps = target_steps
        self.max_retries_per_step = max_retries_per_step
        self.learning_rate = learning_rate

        # 创建临时目录用于存放检查点
        self.tmp_dir = tempfile.mkdtemp(prefix="resilient_trainer_ckpt_")

        # 初始化各组件
        self.model = SimpleMLP(input_dim=64, hidden_dim=128, output_dim=10)
        self.model.use_gradient_checkpoint = use_gradient_checkpoint
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.data_gen = SyntheticDataGenerator(input_dim=64, num_classes=10)
        self.failure_sim = NodeFailureSimulator(failure_rate=failure_rate)
        self.ckpt_mgr = CheckpointManager(
            checkpoint_dir=self.tmp_dir,
            save_interval=checkpoint_interval,
        )
        self.stats = TrainingStats()

        print(f"[初始化] 容错训练器已就绪")
        print(f"  - 目标步数: {target_steps}")
        print(f"  - 检查点间隔: 每{checkpoint_interval}步")
        print(f"  - 故障概率: {failure_rate*100:.0f}%/步")
        print(f"  - 梯度检查点: {'启用' if use_gradient_checkpoint else '禁用'}")
        print(f"  - 检查点目录: {self.tmp_dir}")
        print()

    def _train_one_step(self, step: int) -> float:
        """
        执行单步训练
        可能被故障模拟器中断（抛出异常）
        """
        self.stats.total_steps_attempted += 1

        # 在训练步骤开始前检查是否"发生故障"
        # 真实场景中故障可能在任意时刻发生
        self.failure_sim.maybe_fail(step)

        # 获取训练数据
        x, y = self.data_gen.get_batch()

        # 前向传播（可能使用梯度检查点）
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(x)
        loss = self.criterion(logits, y)

        # 反向传播
        loss.backward()

        # 参数更新
        self.optimizer.step()

        return loss.item()

    def _recover_from_checkpoint(self) -> int:
        """
        从最近的检查点恢复训练状态
        返回恢复到的步数，如果没有检查点则返回0
        """
        restored = self.ckpt_mgr.load_latest(self.model, self.optimizer)

        if restored is not None:
            recovered_step = restored["step"]
            self.stats.loss_history = restored["loss_history"]
            self.stats.successful_recoveries += 1
            self.stats.recovery_events.append({
                "recovered_to_step": recovered_step,
                "time": time.time(),
            })
            print(f"  [恢复] 成功从检查点恢复到 step={recovered_step}")
            return recovered_step
        else:
            # 没有可用的检查点，只能从头开始
            print(f"  [恢复] 没有可用检查点，从 step=0 重新开始")
            self.stats.loss_history = []
            return 0

    def train(self) -> TrainingStats:
        """
        主训练循环，带有完整的容错逻辑

        工作流程:
        1. 执行训练步骤
        2. 如果成功 -> 可能保存检查点 -> 继续下一步
        3. 如果失败 -> 记录故障 -> 从检查点恢复 -> 重新训练
        """
        print("=" * 70)
        print("开始容错训练")
        print("=" * 70)

        current_step = 0
        consecutive_failures = 0

        while current_step < self.target_steps:
            try:
                # 尝试执行一步训练
                loss = self._train_one_step(current_step)

                # 训练成功
                self.stats.loss_history.append(loss)
                self.stats.total_steps_completed = current_step + 1
                consecutive_failures = 0  # 重置连续失败计数

                # 打印训练进度
                print(f"  step {current_step:3d}/{self.target_steps} | "
                      f"loss={loss:.4f} | "
                      f"故障数={self.stats.total_failures} | "
                      f"恢复数={self.stats.successful_recoveries}")

                # 按间隔保存检查点
                if self.ckpt_mgr.should_save(current_step + 1):
                    ckpt_path = self.ckpt_mgr.save(
                        self.model, self.optimizer,
                        current_step + 1, self.stats.loss_history,
                    )
                    self.stats.checkpoint_events.append({
                        "step": current_step + 1,
                        "path": ckpt_path,
                    })
                    print(f"  [检查点] 已保存 step={current_step + 1}")

                current_step += 1

            except RuntimeError as e:
                # 捕获模拟的节点故障
                self.stats.total_failures += 1
                self.stats.failure_events.append({
                    "step": current_step,
                    "error": str(e),
                    "time": time.time(),
                })
                consecutive_failures += 1

                print(f"\n  !!! 故障发生 !!! step={current_step} | {e}")

                # 防止无限重试（现实中也会有重试上限）
                if consecutive_failures > self.max_retries_per_step:
                    print(f"  [致命] 连续失败{consecutive_failures}次，"
                          f"超过最大重试次数{self.max_retries_per_step}，训练终止")
                    break

                # 从检查点恢复
                recovered_step = self._recover_from_checkpoint()
                current_step = recovered_step
                print()

        print()
        print("=" * 70)
        print("训练结束")
        print("=" * 70)

        return self.stats

    def cleanup(self) -> None:
        """清理临时检查点目录"""
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
            print(f"[清理] 已删除临时检查点目录: {self.tmp_dir}")


# ==============================================================================
# 梯度检查点效果对比
# ==============================================================================

def compare_gradient_checkpointing():
    """
    对比使用/不使用梯度检查点的内存消耗差异

    注意: 在CPU上内存差异不如GPU明显,
    真实效果需要在GPU上用大模型才能体现
    """
    print("\n" + "=" * 70)
    print("梯度检查点 (Gradient Checkpointing) 效果对比")
    print("=" * 70)

    import tracemalloc

    results = {}
    for use_ckpt in [False, True]:
        label = "启用梯度检查点" if use_ckpt else "禁用梯度检查点(基准)"

        # 创建模型
        model = SimpleMLP(input_dim=64, hidden_dim=256, output_dim=10)
        model.use_gradient_checkpoint = use_ckpt
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 测量内存：执行一次前向+反向
        tracemalloc.start()
        x = torch.randn(64, 64)
        y = torch.randint(0, 10, (64,))

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results[label] = {"current_kb": current / 1024, "peak_kb": peak / 1024}
        print(f"\n  [{label}]")
        print(f"    当前内存: {current / 1024:.1f} KB")
        print(f"    峰值内存: {peak / 1024:.1f} KB")

    # 对比说明
    labels = list(results.keys())
    peak_base = results[labels[0]]["peak_kb"]
    peak_ckpt = results[labels[1]]["peak_kb"]
    diff = peak_base - peak_ckpt
    print(f"\n  峰值内存差异: {diff:+.1f} KB")
    print(f"  说明: CPU上小模型差异不大；在GPU上训练数十亿参数模型时，")
    print(f"        梯度检查点可节省约60%-70%的显存，代价是约30%的额外计算时间")


# ==============================================================================
# 结果展示
# ==============================================================================

def print_results(stats: TrainingStats):
    """打印训练结果摘要"""
    print("\n" + "=" * 70)
    print("训练结果摘要")
    print("=" * 70)

    # 基本统计
    print(f"\n  [训练进度]")
    print(f"    完成步数:   {stats.total_steps_completed}")
    print(f"    尝试步数:   {stats.total_steps_attempted} (含失败重试)")
    print(f"    总故障次数: {stats.total_failures}")
    print(f"    成功恢复:   {stats.successful_recoveries}")
    overhead = (stats.total_steps_attempted / max(stats.total_steps_completed, 1) - 1) * 100
    print(f"    重试开销:   {overhead:.1f}% 额外计算")

    # 损失曲线（简易文本可视化）
    if stats.loss_history:
        print(f"\n  [损失曲线] (共{len(stats.loss_history)}步)")
        max_loss = max(stats.loss_history)
        min_loss = min(stats.loss_history)
        bar_width = 40

        for i, loss in enumerate(stats.loss_history):
            if max_loss > min_loss:
                normalized = (loss - min_loss) / (max_loss - min_loss)
            else:
                normalized = 0.5
            bar_len = int(normalized * bar_width)
            bar = "#" * bar_len + "." * (bar_width - bar_len)
            # 标记检查点保存位置
            ckpt_marker = " [ckpt]" if any(
                e["step"] == i + 1 for e in stats.checkpoint_events
            ) else ""
            print(f"    step {i:3d} | {loss:.4f} |{bar}|{ckpt_marker}")

        print(f"\n    初始损失: {stats.loss_history[0]:.4f}")
        print(f"    最终损失: {stats.loss_history[-1]:.4f}")

    # 故障与恢复日志
    if stats.failure_events:
        print(f"\n  [故障/恢复日志]")
        for event in stats.failure_events:
            print(f"    step {event['step']:3d} - 故障: {event['error']}")
        for event in stats.recovery_events:
            print(f"    -> 恢复到 step={event['recovered_to_step']}")


def print_comparison_with_real_infra():
    """与真实工业级基础设施的对比"""
    print("\n" + "=" * 70)
    print("与真实工业级容错系统的对比")
    print("=" * 70)

    comparisons = [
        ("检查点保存",
         "本模块: 同步保存到本地临时目录",
         "工业级: 异步保存到分布式存储(S3/HDFS)，支持增量检查点"),
        ("故障检测",
         "本模块: 简单的随机异常模拟",
         "工业级: 心跳监测 + NCCL watchdog + GPU ECC监控 + 网络健康检查"),
        ("故障恢复",
         "本模块: 单进程重载检查点",
         "工业级: 多节点协调恢复，rendezvous重新同步"),
        ("弹性训练",
         "本模块: 未实现（需要多进程环境）",
         "工业级: PyTorch Elastic / Kubernetes弹性调度，动态增减worker"),
        ("梯度检查点",
         "本模块: 简单的4层MLP，CPU上效果不明显",
         "工业级: 数百层Transformer，GPU显存节省60-70%"),
        ("检查点优化",
         "本模块: 完整保存所有参数",
         "工业级: 分片保存(每个rank只存自己的部分)，异步写入，流水线化"),
        ("故障预测",
         "本模块: 无",
         "工业级: 基于历史数据的机器学习预测模型，提前迁移任务"),
        ("Megatron-LM",
         "本模块: 未涉及",
         "工业级: 多级检查点(内存->本地SSD->远程存储)，异步IO，分布式barrier"),
        ("DeepSpeed",
         "本模块: 未涉及",
         "工业级: ZeRO分布式检查点，弹性训练支持，自动故障恢复"),
    ]

    for title, ours, theirs in comparisons:
        print(f"\n  [{title}]")
        print(f"    {ours}")
        print(f"    {theirs}")

    print()


# ==============================================================================
# 主程序入口
# ==============================================================================

def main():
    """主程序：运行容错训练演示"""
    print("=" * 70)
    print(" 模块6: 容错训练演示 (Fault-Tolerant Training Demo)")
    print(" 模拟大规模训练中的节点故障、检查点恢复、梯度检查点")
    print("=" * 70)
    print()

    # 固定随机种子，保证可复现
    torch.manual_seed(42)
    random.seed(42)

    # ---------- 第1部分: 容错训练 ----------
    trainer = ResilientTrainer(
        target_steps=20,          # 目标训练20步
        checkpoint_interval=5,    # 每5步保存检查点
        failure_rate=0.20,        # 每步20%概率发生故障
        max_retries_per_step=5,   # 单步最多重试5次
        use_gradient_checkpoint=True,  # 启用梯度检查点
        learning_rate=1e-3,
    )

    try:
        # 执行训练
        stats = trainer.train()

        # 打印训练结果
        print_results(stats)

    finally:
        # 无论训练是否成功，都清理临时文件
        trainer.cleanup()

    # ---------- 第2部分: 梯度检查点对比 ----------
    compare_gradient_checkpointing()

    # ---------- 第3部分: 工业级对比 ----------
    print_comparison_with_real_infra()

    print("=" * 70)
    print(" 演示完毕!")
    print(" 关键收获:")
    print("   1. 检查点是大规模训练的生命线——没有检查点，故障意味着从头再来")
    print("   2. 故障恢复需要保存完整状态（模型+优化器+RNG）才能精确续训")
    print("   3. 梯度检查点用计算换内存，是训练超大模型的必备技术")
    print("   4. 工业级容错系统远比本演示复杂，涉及分布式协调、异步IO、故障预测等")
    print("=" * 70)


if __name__ == "__main__":
    main()
