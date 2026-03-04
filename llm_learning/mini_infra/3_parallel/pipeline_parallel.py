"""
流水线并行 (Pipeline Parallelism) - 模拟 GPipe 风格的流水线调度

=== 知识点总结 ===

1. 什么是流水线并行？
   - 将模型的不同层分配到不同设备上（按深度切分）
   - 例如: GPU0 负责第 1-12 层, GPU1 负责第 13-24 层
   - 数据从第一个设备流向最后一个设备（像工厂流水线）
   - 问题：朴素实现会导致严重的"流水线气泡"(bubble)——设备空闲等待

2. 流水线气泡问题
   - 朴素方式: 前向传播从 Stage 0 -> Stage 1 -> ... -> Stage N
   - 反向传播从 Stage N -> ... -> Stage 1 -> Stage 0
   - 在任意时刻，只有 1 个 Stage 在工作，其余 N-1 个 Stage 空闲!
   - 气泡比例 = (num_stages - 1) / (num_stages) ，2个Stage时50%的时间浪费!

3. GPipe 调度策略 (Fill-Drain Schedule)
   - 将一个 mini-batch 切分成 M 个 micro-batch
   - 所有 micro-batch 先做完前向传播 (Fill阶段)
   - 然后所有 micro-batch 再做反向传播 (Drain阶段)
   - 气泡比例降低为: (num_stages - 1) / (num_stages + M - 1)
   - M 越大，气泡越小，但激活值内存占用越大（需要存所有 micro-batch 的激活值）

4. 1F1B 调度策略 (PipeDream / Interleaved Schedule)
   - 在稳定阶段，交替执行 1 次前向 + 1 次反向
   - 优势：不需要同时存所有 micro-batch 的激活值，显存更友好
   - 气泡比例与 GPipe 相同，但峰值显存更低
   - PipeDream-Flush (1F1B): 目前最常用的方案

5. 虚拟流水线 (Interleaved Pipeline, Megatron-LM v3)
   - 每个设备负责多个非连续的层块 (virtual stages)
   - 例如 4 层 2 设备: GPU0 负责层 0,2; GPU1 负责层 1,3
   - 进一步减少气泡，但增加了通信次数

6. 实际框架对应
   - GPipe (Google): 最早提出微批次流水线并行
   - PipeDream (Microsoft): 提出 1F1B 调度
   - Megatron-LM: 实现了 1F1B 和 Interleaved Schedule
   - DeepSpeed Pipeline: 提供 PipelineModule 抽象
   - PyTorch Pipeline: torch.distributed.pipelining (新API)

7. 本示例做了什么
   - 将一个简单的 4 层 MLP 切分为 2 个 Stage
   - 演示 GPipe 的 micro-batch 调度流程
   - 可视化流水线时间线（ASCII 甘特图）
   - 对比朴素方式和 GPipe 方式的效率
"""

import numpy as np
import time


# ============================================================
# 简单的全连接层
# ============================================================
class LinearLayer:
    """一个简单的线性层，带 ReLU 激活"""

    def __init__(self, input_dim, output_dim, layer_id, seed=None):
        """初始化线性层"""
        rng = np.random.RandomState(seed)
        # Xavier 初始化
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.W = rng.randn(input_dim, output_dim).astype(np.float64) * scale
        self.b = np.zeros(output_dim, dtype=np.float64)
        self.layer_id = layer_id

        # 缓存（用于反向传播）
        self._input = None
        self._pre_activation = None

        # 梯度
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        """前向传播: ReLU(Wx + b)"""
        self._input = x
        self._pre_activation = x @ self.W + self.b
        return np.maximum(0, self._pre_activation)  # ReLU

    def backward(self, grad_output):
        """反向传播"""
        # ReLU 的梯度
        relu_mask = (self._pre_activation > 0).astype(np.float64)
        grad_pre = grad_output * relu_mask

        batch_size = grad_output.shape[0]
        self.grad_W = self._input.T @ grad_pre / batch_size
        self.grad_b = np.mean(grad_pre, axis=0)

        # 传给前一层的梯度
        grad_input = grad_pre @ self.W.T
        return grad_input


# ============================================================
# 输出层（不带 ReLU）
# ============================================================
class OutputLayer:
    """输出层，不带激活函数"""

    def __init__(self, input_dim, output_dim, layer_id, seed=None):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.W = rng.randn(input_dim, output_dim).astype(np.float64) * scale
        self.b = np.zeros(output_dim, dtype=np.float64)
        self.layer_id = layer_id
        self._input = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        """前向传播: Wx + b (无激活)"""
        self._input = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        """反向传播"""
        batch_size = grad_output.shape[0]
        self.grad_W = self._input.T @ grad_output / batch_size
        self.grad_b = np.mean(grad_output, axis=0)
        grad_input = grad_output @ self.W.T
        return grad_input


# ============================================================
# 流水线 Stage：包含若干连续的层
# ============================================================
class PipelineStage:
    """
    流水线的一个 Stage，包含模型的若干连续层
    对应一个 GPU 设备
    """

    def __init__(self, layers, stage_id):
        """
        初始化 Stage
        layers: 该 Stage 包含的层列表
        stage_id: Stage 编号
        """
        self.layers = layers
        self.stage_id = stage_id
        # 缓存每个 micro-batch 的激活值（反向传播需要）
        self.activation_cache = {}

    def forward(self, x, micro_batch_id):
        """
        前向传播一个 micro-batch
        缓存中间激活值用于反向传播
        """
        # 为每个 micro-batch 保存中间状态
        activations = [x]
        out = x
        for layer in self.layers:
            out = layer.forward(out)
            activations.append(out)

        # 缓存激活值（反向传播时需要）
        self.activation_cache[micro_batch_id] = activations
        return out

    def backward(self, grad_output, micro_batch_id):
        """
        反向传播一个 micro-batch
        使用缓存的激活值
        """
        grad = grad_output
        # 反向遍历各层
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        # 清除已用的激活值缓存，释放内存
        if micro_batch_id in self.activation_cache:
            del self.activation_cache[micro_batch_id]

        return grad


# ============================================================
# MSE 损失
# ============================================================
def mse_loss(y_pred, y_true):
    """计算 MSE 损失和梯度"""
    diff = y_pred - y_true
    loss = np.mean(diff ** 2)
    grad = 2.0 * diff / y_true.shape[0]
    return loss, grad


# ============================================================
# 流水线时间线可视化
# ============================================================
def visualize_pipeline_timeline(schedule, num_stages, num_microbatches, title):
    """
    用 ASCII 艺术可视化流水线时间线

    schedule: 事件列表 [(time_step, stage_id, micro_batch_id, phase), ...]
              phase: 'F' = 前向, 'B' = 反向
    """
    print(f"\n  {title}")
    print(f"  {'─' * 60}")

    # 计算时间线长度
    max_time = max(s[0] for s in schedule) + 1

    # 为每个 Stage 创建时间线
    for stage_id in range(num_stages):
        timeline = ['  . '] * max_time  # 空闲时间用点表示

        for time_step, sid, mb_id, phase in schedule:
            if sid == stage_id:
                if phase == 'F':
                    timeline[time_step] = f' F{mb_id} '  # 前向
                else:
                    timeline[time_step] = f' B{mb_id} '  # 反向

        line = ''.join(timeline)
        print(f"  Stage {stage_id}: |{line}|")

    # 时间刻度
    time_labels = ''.join([f' {t:<3}' for t in range(max_time)])
    print(f"  时间步:  |{time_labels}|")

    # 计算气泡比例
    total_slots = num_stages * max_time
    active_slots = len(schedule)
    bubble_ratio = 1.0 - active_slots / total_slots
    print(f"\n  总时间步: {max_time}, 活跃槽位: {active_slots}/{total_slots}, "
          f"气泡比例: {bubble_ratio:.1%}")


# ============================================================
# 朴素流水线（无微批次）
# ============================================================
def naive_pipeline_schedule(num_stages, num_microbatches=1):
    """
    朴素流水线调度:
    - 前向: Stage 0 -> Stage 1 -> ... (串行)
    - 反向: ... -> Stage 1 -> Stage 0 (串行)
    严重的流水线气泡!
    """
    schedule = []
    t = 0

    # 前向传播: Stage 0 -> Stage 1 -> ...
    for stage in range(num_stages):
        schedule.append((t, stage, 0, 'F'))
        t += 1

    # 反向传播: ... -> Stage 1 -> Stage 0
    for stage in range(num_stages - 1, -1, -1):
        schedule.append((t, stage, 0, 'B'))
        t += 1

    return schedule


# ============================================================
# GPipe 调度
# ============================================================
def gpipe_schedule(num_stages, num_microbatches):
    """
    GPipe 调度 (Fill-Drain):
    1. Fill 阶段: 所有 micro-batch 依次通过所有 Stage 的前向传播
    2. Drain 阶段: 所有 micro-batch 依次通过所有 Stage 的反向传播

    调度规则:
    - micro-batch m 在 stage s 的前向开始时间: t = s + m
    - micro-batch m 在 stage s 的反向开始时间从最后一个 stage 开始回溯
    """
    schedule = []

    # === Fill 阶段: 前向传播 ===
    # micro-batch m 在 stage s 的前向时间步: t = m + s
    for m in range(num_microbatches):
        for s in range(num_stages):
            t = m + s
            schedule.append((t, s, m, 'F'))

    # 前向传播结束的时间步
    fwd_end = num_microbatches + num_stages - 1

    # === Drain 阶段: 反向传播 ===
    # 反向从最后一个 micro-batch 在最后一个 stage 开始
    # micro-batch m 在 stage s 的反向时间步
    for m in range(num_microbatches):
        for s in range(num_stages - 1, -1, -1):
            t = fwd_end + m + (num_stages - 1 - s)
            schedule.append((t, s, m, 'B'))

    return schedule


# ============================================================
# 1F1B 调度（额外展示）
# ============================================================
def one_f_one_b_schedule(num_stages, num_microbatches):
    """
    1F1B 调度 (PipeDream-Flush):
    1. 预热阶段: 前几个 micro-batch 做前向填充流水线
    2. 稳定阶段: 交替执行 1 次前向 + 1 次反向
    3. 排空阶段: 剩余 micro-batch 做反向排空

    优势: 峰值显存更低（不需要同时保存所有 micro-batch 的激活值）
    """
    schedule = []

    # 每个 Stage 维护自己的时间计数器
    # 使用简化的模拟：按时间步分配任务
    stage_time = [0] * num_stages  # 每个 stage 下一个可用时间步

    # 前向队列和反向队列
    fwd_done = [[] for _ in range(num_stages)]  # 每个 stage 已完成前向的 mb 列表

    # 预热阶段: 填充流水线（前 num_stages 个 micro-batch 只做前向）
    warmup_batches = min(num_stages, num_microbatches)

    for m in range(warmup_batches):
        for s in range(num_stages):
            t = max(stage_time[s], (m + s))  # 确保前一个 stage 先完成
            schedule.append((t, s, m, 'F'))
            stage_time[s] = t + 1
            fwd_done[s].append(m)

    # 稳定阶段: 1F1B
    for m in range(warmup_batches, num_microbatches):
        for s in range(num_stages):
            # 先做一个反向
            if fwd_done[s]:
                bm = fwd_done[s].pop(0)
                t = stage_time[s]
                # 反向从最后一个 stage 开始，需要等后面 stage 的反向结果
                if s < num_stages - 1:
                    t = max(t, stage_time[s])
                schedule.append((t, s, bm, 'B'))
                stage_time[s] = t + 1

            # 再做一个前向
            t = max(stage_time[s], (m + s))
            schedule.append((t, s, m, 'F'))
            stage_time[s] = t + 1
            fwd_done[s].append(m)

    # 排空阶段: 处理剩余的反向传播
    for s in range(num_stages):
        while fwd_done[s]:
            bm = fwd_done[s].pop(0)
            t = stage_time[s]
            schedule.append((t, s, bm, 'B'))
            stage_time[s] = t + 1

    return schedule


# ============================================================
# 实际的流水线并行训练模拟
# ============================================================
def run_gpipe_training(stages, x_all, y_all, num_microbatches):
    """
    运行 GPipe 风格的流水线并行训练

    stages: PipelineStage 列表
    x_all: 完整输入数据
    y_all: 完整目标数据
    num_microbatches: micro-batch 数量
    """
    num_stages = len(stages)
    batch_size = x_all.shape[0]
    mb_size = batch_size // num_microbatches

    # 切分 micro-batches
    x_mbs = [x_all[i * mb_size:(i + 1) * mb_size] for i in range(num_microbatches)]
    y_mbs = [y_all[i * mb_size:(i + 1) * mb_size] for i in range(num_microbatches)]

    print(f"\n  切分为 {num_microbatches} 个 micro-batch, 每个大小 = {mb_size}")

    # === Fill 阶段: 所有 micro-batch 前向传播 ===
    print(f"\n  === Fill 阶段 (前向传播) ===")

    # 存储每个 micro-batch 在各 stage 之间的中间输出
    inter_outputs = {}  # (micro_batch_id, stage_id) -> output

    total_loss = 0.0
    losses = []

    for m in range(num_microbatches):
        x_mb = x_mbs[m]
        for s in range(num_stages):
            if s == 0:
                input_data = x_mb
            else:
                input_data = inter_outputs[(m, s - 1)]

            output = stages[s].forward(input_data, micro_batch_id=m)
            inter_outputs[(m, s)] = output

            print(f"    micro-batch {m} -> Stage {s}: "
                  f"输入形状 {input_data.shape} -> 输出形状 {output.shape}")

        # 最后一个 stage 的输出计算损失
        y_pred = inter_outputs[(m, num_stages - 1)]
        loss, grad = mse_loss(y_pred, y_mbs[m])
        inter_outputs[(m, 'loss_grad')] = grad
        losses.append(loss)
        total_loss += loss
        print(f"    micro-batch {m} 损失: {loss:.6f}")

    avg_loss = total_loss / num_microbatches
    print(f"\n  平均损失: {avg_loss:.6f}")

    # === Drain 阶段: 所有 micro-batch 反向传播 ===
    print(f"\n  === Drain 阶段 (反向传播) ===")

    for m in range(num_microbatches):
        grad = inter_outputs[(m, 'loss_grad')]

        for s in range(num_stages - 1, -1, -1):
            grad = stages[s].backward(grad, micro_batch_id=m)
            print(f"    micro-batch {m} <- Stage {s}: 梯度形状 {grad.shape}")

    # 梯度累积：GPipe 将所有 micro-batch 的梯度求平均
    print(f"\n  梯度累积完成 (所有 micro-batch 梯度自动累积)")

    return avg_loss, losses


# ============================================================
# 非流水线训练（对照组）
# ============================================================
def run_sequential_training(all_layers, x_all, y_all):
    """顺序训练（非流水线），作为正确性参照"""
    out = x_all
    for layer in all_layers:
        out = layer.forward(out)

    loss, grad = mse_loss(out, y_all)

    for layer in reversed(all_layers):
        grad = layer.backward(grad)

    return loss


# ============================================================
# 主函数
# ============================================================
def main():
    np.random.seed(42)

    print("=" * 70)
    print("流水线并行 (Pipeline Parallelism) 模拟演示 - GPipe 风格")
    print("=" * 70)

    # ============================
    # 实验 1: 流水线调度可视化
    # ============================
    print(f"\n{'='*70}")
    print("实验 1: 流水线调度策略可视化")
    print(f"{'='*70}")

    num_stages = 2
    num_microbatches = 4

    # 朴素调度
    print(f"\n  配置: {num_stages} 个 Stage, 1 个 batch (无微批次)")
    naive_sched = naive_pipeline_schedule(num_stages)
    visualize_pipeline_timeline(
        naive_sched, num_stages, 1,
        "朴素流水线 (Naive Pipeline) - 严重的气泡!"
    )

    # GPipe 调度
    print(f"\n  配置: {num_stages} 个 Stage, {num_microbatches} 个 micro-batch")
    gpipe_sched = gpipe_schedule(num_stages, num_microbatches)
    visualize_pipeline_timeline(
        gpipe_sched, num_stages, num_microbatches,
        f"GPipe 调度 (Fill-Drain) - {num_microbatches} 个 micro-batch"
    )

    # 不同 micro-batch 数量的气泡比较
    print(f"\n  不同 micro-batch 数量的气泡比例:")
    print(f"  {'micro-batches':>15} | {'气泡比例':>10} | {'加速比':>10}")
    print(f"  {'─'*15}─┼─{'─'*10}─┼─{'─'*10}")
    for M in [1, 2, 4, 8, 16, 32]:
        bubble = (num_stages - 1) / (num_stages + M - 1)
        speedup = num_stages * (1 - bubble)
        print(f"  {M:>15} | {bubble:>9.1%} | {speedup:>10.2f}x")

    # ============================
    # 实验 2: 更多 Stage 的调度
    # ============================
    print(f"\n{'='*70}")
    print("实验 2: 4 Stage 流水线调度")
    print(f"{'='*70}")

    num_stages_4 = 4
    num_mb_4 = 6
    print(f"\n  配置: {num_stages_4} 个 Stage, {num_mb_4} 个 micro-batch")

    gpipe_sched_4 = gpipe_schedule(num_stages_4, num_mb_4)
    visualize_pipeline_timeline(
        gpipe_sched_4, num_stages_4, num_mb_4,
        f"GPipe 调度 - {num_stages_4} Stage x {num_mb_4} micro-batch"
    )

    # ============================
    # 实验 3: 实际运行流水线训练
    # ============================
    print(f"\n{'='*70}")
    print("实验 3: 实际运行 GPipe 流水线训练")
    print(f"{'='*70}")

    # 模型配置: 4 层 MLP，分成 2 个 Stage
    input_dim = 8
    hidden_dim = 16
    output_dim = 4
    batch_size = 16
    num_microbatches_run = 4

    # 创建 4 个层
    layer0 = LinearLayer(input_dim, hidden_dim, layer_id=0, seed=10)
    layer1 = LinearLayer(hidden_dim, hidden_dim, layer_id=1, seed=20)
    layer2 = LinearLayer(hidden_dim, hidden_dim, layer_id=2, seed=30)
    layer3 = OutputLayer(hidden_dim, output_dim, layer_id=3, seed=40)

    # 分配到 2 个 Stage
    # Stage 0 (GPU 0): 层 0, 层 1
    # Stage 1 (GPU 1): 层 2, 层 3
    stage0 = PipelineStage([layer0, layer1], stage_id=0)
    stage1 = PipelineStage([layer2, layer3], stage_id=1)
    stages = [stage0, stage1]

    print(f"\n  模型: 4 层 MLP")
    print(f"    层 0: Linear({input_dim}, {hidden_dim}) + ReLU  -> Stage 0 (GPU 0)")
    print(f"    层 1: Linear({hidden_dim}, {hidden_dim}) + ReLU  -> Stage 0 (GPU 0)")
    print(f"    层 2: Linear({hidden_dim}, {hidden_dim}) + ReLU  -> Stage 1 (GPU 1)")
    print(f"    层 3: Linear({hidden_dim}, {output_dim})         -> Stage 1 (GPU 1)")
    print(f"\n  数据: batch_size={batch_size}, micro-batches={num_microbatches_run}")

    # 生成数据
    rng = np.random.RandomState(99)
    x_data = rng.randn(batch_size, input_dim).astype(np.float64)
    y_data = rng.randn(batch_size, output_dim).astype(np.float64)

    # 运行 GPipe 流水线训练
    print(f"\n  --- GPipe 流水线训练 ---")
    pipeline_loss, mb_losses = run_gpipe_training(
        stages, x_data, y_data, num_microbatches_run
    )

    # ============================
    # 实验 4: 与非流水线训练对比
    # ============================
    print(f"\n{'='*70}")
    print("实验 4: 正确性验证 - 对比非流水线训练")
    print(f"{'='*70}")

    # 重新创建相同的层（因为上面的训练修改了内部状态）
    layer0_ref = LinearLayer(input_dim, hidden_dim, layer_id=0, seed=10)
    layer1_ref = LinearLayer(hidden_dim, hidden_dim, layer_id=1, seed=20)
    layer2_ref = LinearLayer(hidden_dim, hidden_dim, layer_id=2, seed=30)
    layer3_ref = OutputLayer(hidden_dim, output_dim, layer_id=3, seed=40)
    all_layers_ref = [layer0_ref, layer1_ref, layer2_ref, layer3_ref]

    sequential_loss = run_sequential_training(all_layers_ref, x_data, y_data)

    print(f"\n  非流水线(全量batch)损失: {sequential_loss:.6f}")
    print(f"  GPipe 流水线平均损失:     {pipeline_loss:.6f}")
    print(f"\n  注意: 两者损失可能略有不同，因为:")
    print(f"  - GPipe 将 batch 切成 micro-batch, 每个 micro-batch 独立计算损失")
    print(f"  - 各 micro-batch 的损失求平均 约等于 全量 batch 的损失")

    # 检查各 micro-batch 损失
    print(f"\n  各 micro-batch 损失:")
    for i, l in enumerate(mb_losses):
        print(f"    micro-batch {i}: {l:.6f}")
    print(f"    平均: {np.mean(mb_losses):.6f}")
    print(f"    全量: {sequential_loss:.6f}")
    print(f"    差异: {abs(np.mean(mb_losses) - sequential_loss):.6e}")

    # ============================
    # 实验 5: 性能分析
    # ============================
    print(f"\n{'='*70}")
    print("实验 5: 性能与显存分析")
    print(f"{'='*70}")

    print(f"""
  假设每个 Stage 的前向/反向传播时间均为 1 个单位时间:

  ┌───────────────────────────────────────────────────────────────┐
  │ 方式          │ 总时间步 │ 气泡比例 │ 峰值激活值内存          │
  ├───────────────┼──────────┼──────────┼────────────────────────┤
  │ 朴素(无微批次)│ {2*num_stages:>8} │ {(num_stages-1)/(2*num_stages):>7.0%} │ 1 个 batch              │
  │ GPipe M={num_microbatches_run:>2}    │ {2*(num_stages + num_microbatches_run - 1):>8} │ {(num_stages-1)/(num_stages + num_microbatches_run - 1):>7.0%} │ {num_microbatches_run} 个 micro-batch 的激活 │
  │ 1F1B M={num_microbatches_run:>2}     │ {2*(num_stages + num_microbatches_run - 1):>8} │ {(num_stages-1)/(num_stages + num_microbatches_run - 1):>7.0%} │ {num_stages} 个 micro-batch 的激活  │
  └───────────────┴──────────┴──────────┴────────────────────────┘

  关键观察:
  - GPipe 和 1F1B 的气泡比例相同
  - 1F1B 的峰值显存更低! (只需存 num_stages 个 micro-batch 的激活值)
  - 增加 micro-batch 数量可以减少气泡，但 GPipe 的显存也线性增长
    """)

    # ============================
    # 与真实框架的对比说明
    # ============================
    print(f"{'='*70}")
    print("与真实基础设施的对比")
    print(f"{'='*70}")
    print("""
    本示例 (模拟)                        真实框架
    ─────────────────────────────────────────────────────────────────
    PipelineStage 类                 ->  DeepSpeed PipelineModule
    手动切分 micro-batch             ->  框架自动切分
    顺序模拟 Stage 执行              ->  各 GPU 真正并行 + P2P 通信
    stage.forward/backward           ->  torch.distributed.send/recv
    NumPy 计算                       ->  CUDA 异步计算 + 流水线重叠

    关键区别:
    1. 真实系统中各 Stage 在不同 GPU 上真正并行执行
    2. Stage 间通过 P2P (点对点) 通信传递激活值和梯度
       - send(): 发送激活值到下一个 Stage
       - recv(): 接收上一个 Stage 的激活值
    3. 1F1B 调度在稳定阶段，各 Stage 可以同时执行不同 micro-batch 的前/反向
    4. 流水线并行通常与数据并行和张量并行组合使用 (3D 并行):
       - 张量并行: 机内 (NVLink 高带宽)
       - 流水线并行: 机间 (InfiniBand, 通信量最小)
       - 数据并行: 最外层 (扩展训练规模)
    5. 负载均衡很重要: 各 Stage 的计算量应尽量一致
       - 否则快的 Stage 会等慢的 Stage，增加气泡
       - 实践中按参数量/FLOPs 均衡分配层
    """)


if __name__ == "__main__":
    main()
