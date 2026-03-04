"""
模块8: LLM成本建模与监控 (Cost Modeling & Monitoring)

知识点总结:
==========

1. GPU定价体系:
   - 按需实例 (On-demand): 最贵但最灵活, A100 约 $2-3/小时, H100 约 $3-4/小时
   - 预留实例 (Reserved): 1-3年合约, 可节省30-60%
   - 竞价实例 (Spot/Preemptible): 最便宜但可能被抢占, 节省60-90%
   - 云厂商定价差异: AWS, GCP, Azure 各有不同定价策略

2. Token经济学:
   - 输入Token vs 输出Token: 输出通常比输入贵2-4倍 (因为自回归生成更耗算力)
   - GPT-4级别: 输入 ~$30/1M tokens, 输出 ~$60/1M tokens
   - GPT-3.5级别: 输入 ~$0.5/1M tokens, 输出 ~$1.5/1M tokens
   - 开源模型自部署: 成本取决于硬件和吞吐量优化

3. 训练成本估算 (Chinchilla Scaling Law):
   - FLOPs ≈ 6 * N * D (N=参数量, D=训练token数)
   - 训练时间 = FLOPs / (GPU数量 * 单GPU FLOPS * MFU)
   - MFU (Model FLOPs Utilization): 实际利用率通常30-60%
   - 训练成本 = 训练时间 * GPU数量 * 每GPU每小时价格

4. TCO (Total Cost of Ownership) 总拥有成本:
   - 硬件成本: GPU/CPU/内存/存储/网络设备
   - 电力成本: GPU功耗200-700W, PUE(数据中心能效比)通常1.1-1.4
   - 冷却成本: 液冷vs风冷, 占电力成本的30-40%
   - 人力成本: MLOps工程师, 基础设施团队
   - 网络带宽: 多节点训练需要高速互联 (InfiniBand/RoCE)

5. 成本优化策略:
   - 混合精度训练 (FP16/BF16): 减少50%显存, 提升2x吞吐
   - 模型并行 + 数据并行: 提高GPU利用率
   - KV Cache优化: 减少推理时重复计算
   - 批处理优化: 增大batch size提高吞吐量
   - 量化部署 (INT8/INT4): 减少推理成本2-4倍
"""

import psutil       # 系统资源监控
import os           # 操作系统接口
import time         # 时间相关
import threading    # 多线程 (后台采样)
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ============================================================
# 第一部分: 资源监控器 (ResourceMonitor)
# ============================================================

class ResourceMonitor:
    """
    实时资源监控器
    - 监控CPU利用率、内存使用(RSS/VMS)、进程运行时间
    - 支持上下文管理器: with ResourceMonitor() as mon:
    - 后台线程每0.5秒采样一次
    """

    def __init__(self, interval: float = 0.5):
        """
        初始化监控器

        参数:
            interval: 采样间隔(秒), 默认0.5秒
        """
        self.interval = interval          # 采样间隔
        self.samples: List[dict] = []     # 采样记录列表
        self._stop_event = threading.Event()  # 停止信号
        self._thread: Optional[threading.Thread] = None  # 后台采样线程
        self._process = psutil.Process(os.getpid())       # 当前进程句柄
        self.start_time: float = 0.0      # 监控开始时间
        self.end_time: float = 0.0        # 监控结束时间

    def _sample(self):
        """采集一次资源快照"""
        try:
            # 获取CPU利用率 (需要一个短暂的时间窗口)
            cpu_percent = self._process.cpu_percent(interval=None)

            # 获取内存信息
            mem_info = self._process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)   # 常驻内存 (MB)
            vms_mb = mem_info.vms / (1024 * 1024)    # 虚拟内存 (MB)

            # 获取系统级CPU利用率
            system_cpu = psutil.cpu_percent(interval=None)

            # 获取系统内存信息
            sys_mem = psutil.virtual_memory()

            elapsed = time.time() - self.start_time  # 已运行时间

            sample = {
                "timestamp": elapsed,                          # 相对时间(秒)
                "process_cpu_percent": cpu_percent,            # 进程CPU利用率(%)
                "system_cpu_percent": system_cpu,              # 系统CPU利用率(%)
                "rss_mb": round(rss_mb, 2),                   # 进程RSS内存(MB)
                "vms_mb": round(vms_mb, 2),                   # 进程VMS内存(MB)
                "system_mem_percent": sys_mem.percent,         # 系统内存利用率(%)
            }
            self.samples.append(sample)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # 进程已结束或无权访问, 静默处理
            pass

    def _sampling_loop(self):
        """后台采样循环 (在独立线程中运行)"""
        # 先做一次初始采样, 初始化CPU计数器
        self._process.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(self.interval)  # 等待指定间隔或收到停止信号

    def start(self):
        """启动监控"""
        self.start_time = time.time()
        self.samples.clear()
        self._stop_event.clear()
        # 创建并启动后台守护线程
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止监控"""
        self.end_time = time.time()
        self._stop_event.set()            # 发送停止信号
        if self._thread is not None:
            self._thread.join(timeout=2)  # 等待线程结束, 最多2秒

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
        return False  # 不吞掉异常

    @property
    def elapsed(self) -> float:
        """返回总运行时间(秒)"""
        return self.end_time - self.start_time

    def get_summary(self) -> dict:
        """
        获取监控摘要统计信息
        返回CPU和内存的平均值、峰值等
        """
        if not self.samples:
            return {"error": "没有采样数据"}

        cpu_values = [s["process_cpu_percent"] for s in self.samples]
        rss_values = [s["rss_mb"] for s in self.samples]
        vms_values = [s["vms_mb"] for s in self.samples]

        return {
            "elapsed_seconds": round(self.elapsed, 3),           # 总运行时间
            "num_samples": len(self.samples),                     # 采样次数
            "cpu_avg_percent": round(sum(cpu_values) / len(cpu_values), 2),  # CPU平均利用率
            "cpu_max_percent": round(max(cpu_values), 2),         # CPU峰值利用率
            "rss_avg_mb": round(sum(rss_values) / len(rss_values), 2),  # RSS平均内存
            "rss_max_mb": round(max(rss_values), 2),              # RSS峰值内存
            "vms_avg_mb": round(sum(vms_values) / len(vms_values), 2),  # VMS平均内存
            "vms_max_mb": round(max(vms_values), 2),              # VMS峰值内存
        }

    def print_report(self):
        """打印格式化的监控报告"""
        summary = self.get_summary()
        if "error" in summary:
            print(f"  [错误] {summary['error']}")
            return

        print(f"  运行时间:     {summary['elapsed_seconds']:.3f} 秒")
        print(f"  采样次数:     {summary['num_samples']}")
        print(f"  CPU利用率:    平均 {summary['cpu_avg_percent']:.1f}%  |  峰值 {summary['cpu_max_percent']:.1f}%")
        print(f"  RSS内存:      平均 {summary['rss_avg_mb']:.1f} MB  |  峰值 {summary['rss_max_mb']:.1f} MB")
        print(f"  VMS内存:      平均 {summary['vms_avg_mb']:.1f} MB  |  峰值 {summary['vms_max_mb']:.1f} MB")


# ============================================================
# 第二部分: 成本估算器 (CostEstimator)
# ============================================================

@dataclass
class GPUSpec:
    """GPU规格数据类"""
    name: str                    # GPU型号名称
    fp16_tflops: float           # FP16/BF16算力 (TFLOPS)
    memory_gb: float             # 显存大小 (GB)
    tdp_watts: float             # 热设计功耗 (W)
    cloud_price_per_hour: float  # 云服务按需价格 ($/小时)
    generation: str              # 架构代次


class CostEstimator:
    """
    LLM训练和推理成本估算器

    功能:
    1. 基于FLOPs和GPU规格估算训练成本
    2. 基于吞吐量估算推理成本 ($/1M tokens)
    3. 内置常见GPU参考价格
    """

    # 常见GPU参考规格 (价格为云服务商大致按需价格)
    GPU_CATALOG: Dict[str, GPUSpec] = {
        "A100-40GB": GPUSpec(
            name="A100-40GB",
            fp16_tflops=312,       # FP16 Tensor Core算力
            memory_gb=40,
            tdp_watts=400,
            cloud_price_per_hour=2.21,  # AWS p4d 大致单卡价格
            generation="Ampere",
        ),
        "A100-80GB": GPUSpec(
            name="A100-80GB",
            fp16_tflops=312,
            memory_gb=80,
            tdp_watts=400,
            cloud_price_per_hour=2.93,  # AWS p4de 大致单卡价格
            generation="Ampere",
        ),
        "H100-80GB": GPUSpec(
            name="H100-80GB",
            fp16_tflops=990,       # FP16 Tensor Core (with sparsity ~1979)
            memory_gb=80,
            tdp_watts=700,
            cloud_price_per_hour=3.88,  # AWS p5 大致单卡价格
            generation="Hopper",
        ),
        "L4-24GB": GPUSpec(
            name="L4-24GB",
            fp16_tflops=121,
            memory_gb=24,
            tdp_watts=72,
            cloud_price_per_hour=0.81,  # GCP 大致价格
            generation="Ada Lovelace",
        ),
        "T4-16GB": GPUSpec(
            name="T4-16GB",
            fp16_tflops=65,
            memory_gb=16,
            tdp_watts=70,
            cloud_price_per_hour=0.53,  # AWS g4dn 大致单卡价格
            generation="Turing",
        ),
        "V100-16GB": GPUSpec(
            name="V100-16GB",
            fp16_tflops=125,
            memory_gb=16,
            tdp_watts=300,
            cloud_price_per_hour=1.46,  # AWS p3 大致单卡价格
            generation="Volta",
        ),
    }

    def __init__(self):
        """初始化成本估算器"""
        pass

    @staticmethod
    def estimate_training_flops(num_params: int, num_tokens: int) -> float:
        """
        估算训练所需总FLOPs (基于Chinchilla缩放定律近似)

        公式: FLOPs ≈ 6 * N * D
        - N: 模型参数量
        - D: 训练token数
        - 6: 前向传播约2x, 反向传播约4x (梯度计算+参数更新)

        参数:
            num_params: 模型参数量 (例如 7e9 表示7B)
            num_tokens: 训练token数 (例如 2e12 表示2T tokens)

        返回:
            总FLOPs (浮点运算次数)
        """
        return 6 * num_params * num_tokens

    def estimate_training_cost(
        self,
        num_params: int,
        num_tokens: int,
        gpu_type: str = "A100-80GB",
        num_gpus: int = 1,
        mfu: float = 0.40,
    ) -> dict:
        """
        估算训练总成本

        参数:
            num_params: 模型参数量
            num_tokens: 训练token数
            gpu_type: GPU型号 (必须在GPU_CATALOG中)
            num_gpus: 使用的GPU数量
            mfu: Model FLOPs Utilization, GPU实际利用率 (默认40%)

        返回:
            包含详细成本分解的字典
        """
        if gpu_type not in self.GPU_CATALOG:
            raise ValueError(f"未知GPU型号: {gpu_type}, 可选: {list(self.GPU_CATALOG.keys())}")

        gpu = self.GPU_CATALOG[gpu_type]

        # 总FLOPs
        total_flops = self.estimate_training_flops(num_params, num_tokens)

        # 单GPU有效算力 (FLOPS, 注意单位: TFLOPS -> FLOPS)
        effective_flops_per_gpu = gpu.fp16_tflops * 1e12 * mfu

        # 总有效算力
        total_effective_flops = effective_flops_per_gpu * num_gpus

        # 训练时间 (秒)
        training_seconds = total_flops / total_effective_flops
        training_hours = training_seconds / 3600
        training_days = training_hours / 24

        # 训练成本 ($)
        total_cost = training_hours * num_gpus * gpu.cloud_price_per_hour

        # 电力成本估算 (辅助参考)
        total_power_kwh = (gpu.tdp_watts * num_gpus * training_hours) / 1000
        electricity_cost = total_power_kwh * 0.10  # 假设 $0.10/kWh

        return {
            "model_params": num_params,
            "training_tokens": num_tokens,
            "total_flops": total_flops,
            "gpu_type": gpu_type,
            "num_gpus": num_gpus,
            "mfu": mfu,
            "training_hours": round(training_hours, 2),
            "training_days": round(training_days, 2),
            "cloud_cost_usd": round(total_cost, 2),
            "electricity_kwh": round(total_power_kwh, 2),
            "electricity_cost_usd": round(electricity_cost, 2),
        }

    def estimate_inference_cost(
        self,
        gpu_type: str = "A100-80GB",
        tokens_per_second: float = 1000,
        utilization: float = 0.70,
    ) -> dict:
        """
        估算推理成本 ($/1M tokens)

        参数:
            gpu_type: GPU型号
            tokens_per_second: 单GPU每秒生成token数 (吞吐量)
            utilization: GPU利用率 (考虑请求间的空闲时间)

        返回:
            包含推理成本分解的字典
        """
        if gpu_type not in self.GPU_CATALOG:
            raise ValueError(f"未知GPU型号: {gpu_type}")

        gpu = self.GPU_CATALOG[gpu_type]

        # 有效吞吐量 (考虑利用率)
        effective_tps = tokens_per_second * utilization

        # 每小时处理的token数
        tokens_per_hour = effective_tps * 3600

        # 每百万token的成本
        cost_per_million_tokens = (gpu.cloud_price_per_hour / tokens_per_hour) * 1_000_000

        # 每小时电力成本
        power_cost_per_hour = (gpu.tdp_watts / 1000) * 0.10  # $0.10/kWh

        return {
            "gpu_type": gpu_type,
            "tokens_per_second": tokens_per_second,
            "utilization": utilization,
            "effective_tps": effective_tps,
            "tokens_per_hour": tokens_per_hour,
            "gpu_cost_per_hour": gpu.cloud_price_per_hour,
            "cost_per_1m_tokens": round(cost_per_million_tokens, 4),
            "power_cost_per_hour": round(power_cost_per_hour, 4),
        }

    def print_gpu_comparison(self):
        """打印GPU对比表"""
        print("=" * 90)
        print("GPU 规格与价格对比表")
        print("=" * 90)
        header = f"{'GPU型号':<16} {'算力(TFLOPS)':<14} {'显存(GB)':<10} {'功耗(W)':<10} {'价格($/h)':<12} {'架构':<14}"
        print(header)
        print("-" * 90)
        for gpu in self.GPU_CATALOG.values():
            row = (
                f"{gpu.name:<16} "
                f"{gpu.fp16_tflops:<14.0f} "
                f"{gpu.memory_gb:<10.0f} "
                f"{gpu.tdp_watts:<10.0f} "
                f"{gpu.cloud_price_per_hour:<12.2f} "
                f"{gpu.generation:<14}"
            )
            print(row)
        print("-" * 90)
        print("注: 价格为云服务商(AWS/GCP)按需实例大致单卡价格, 实际价格因地区和供需变化")
        print()


# ============================================================
# 第三部分: 辅助函数
# ============================================================

def format_number(n: float) -> str:
    """将大数字格式化为易读形式 (如 7B, 2.1T)"""
    if n >= 1e12:
        return f"{n/1e12:.1f}T"
    elif n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return f"{n:.0f}"


def format_flops(flops: float) -> str:
    """将FLOPs格式化为易读形式"""
    if flops >= 1e24:
        return f"{flops/1e24:.2f} YottaFLOPs"
    elif flops >= 1e21:
        return f"{flops/1e21:.2f} ZettaFLOPs"
    elif flops >= 1e18:
        return f"{flops/1e18:.2f} ExaFLOPs"
    elif flops >= 1e15:
        return f"{flops/1e15:.2f} PetaFLOPs"
    elif flops >= 1e12:
        return f"{flops/1e12:.2f} TeraFLOPs"
    else:
        return f"{flops:.2e} FLOPs"


# ============================================================
# 第四部分: 演示主程序
# ============================================================

def demo_resource_monitor():
    """演示1: 资源监控器 - 监控矩阵乘法工作负载"""
    print("\n" + "=" * 70)
    print("演示1: 资源监控器 (ResourceMonitor)")
    print("=" * 70)
    print("运行一组矩阵乘法作为示例工作负载, 同时监控系统资源...\n")

    # 使用上下文管理器启动监控
    with ResourceMonitor(interval=0.5) as mon:
        # 模拟工作负载: 大量矩阵乘法 (纯CPU)
        size = 512  # 矩阵维度
        iterations = 20

        print(f"  工作负载: {iterations} 次 {size}x{size} 矩阵乘法")

        for i in range(iterations):
            # 创建随机矩阵
            a = [[0.0] * size for _ in range(size)]
            b = [[0.0] * size for _ in range(size)]

            # 简单填充 (避免import numpy, 保持纯Python)
            for r in range(size):
                for c in range(size):
                    a[r][c] = (r * c + 1) % 1000 / 1000.0
                    b[r][c] = (r + c + 1) % 1000 / 1000.0

            # 矩阵乘法 (只算部分行以控制时间)
            result_rows = min(64, size)
            result = []
            for r in range(result_rows):
                row = []
                for c in range(size):
                    val = 0.0
                    for k in range(size):
                        val += a[r][k] * b[k][c]
                    row.append(val)
                result.append(row)

            if (i + 1) % 5 == 0:
                print(f"    已完成 {i+1}/{iterations} 次迭代")

    # 打印监控报告
    print("\n  --- 资源监控报告 ---")
    mon.print_report()

    # 打印采样时间线
    if mon.samples:
        print("\n  --- 采样时间线 (前5条) ---")
        print(f"  {'时间(s)':<10} {'CPU(%)':<10} {'RSS(MB)':<12} {'VMS(MB)':<12}")
        print(f"  {'-'*44}")
        for s in mon.samples[:5]:
            print(f"  {s['timestamp']:<10.2f} {s['process_cpu_percent']:<10.1f} {s['rss_mb']:<12.1f} {s['vms_mb']:<12.1f}")
        if len(mon.samples) > 5:
            print(f"  ... (共 {len(mon.samples)} 条采样记录)")


def demo_training_cost():
    """演示2: 训练成本估算 - 不同规模模型的训练成本"""
    print("\n" + "=" * 70)
    print("演示2: 训练成本估算")
    print("=" * 70)

    estimator = CostEstimator()

    # 定义模型规模和训练设置
    # 遵循Chinchilla最优: 训练token数 ≈ 20 * 参数量
    models = [
        {"name": "1B模型",  "params": 1e9,  "tokens": 20e9,  "gpus": 8,   "gpu": "A100-80GB"},
        {"name": "7B模型",  "params": 7e9,  "tokens": 2e12,  "gpus": 32,  "gpu": "A100-80GB"},
        {"name": "13B模型", "params": 13e9, "tokens": 2e12,  "gpus": 64,  "gpu": "A100-80GB"},
        {"name": "70B模型", "params": 70e9, "tokens": 2e12,  "gpus": 512, "gpu": "H100-80GB"},
    ]

    # 打印表头
    print(f"\n  {'模型':<10} {'参数量':<10} {'训练Token':<12} {'GPU':<14} {'GPU数':<8} "
          f"{'总FLOPs':<20} {'训练天数':<10} {'估算成本($)':<14}")
    print(f"  {'-'*104}")

    for m in models:
        result = estimator.estimate_training_cost(
            num_params=m["params"],
            num_tokens=m["tokens"],
            gpu_type=m["gpu"],
            num_gpus=m["gpus"],
            mfu=0.40,  # 假设40%的MFU
        )
        total_flops = estimator.estimate_training_flops(m["params"], m["tokens"])

        print(
            f"  {m['name']:<10} "
            f"{format_number(m['params']):<10} "
            f"{format_number(m['tokens']):<12} "
            f"{m['gpu']:<14} "
            f"{m['gpus']:<8} "
            f"{format_flops(total_flops):<20} "
            f"{result['training_days']:<10.1f} "
            f"${result['cloud_cost_usd']:>12,.0f}"
        )

    print()
    print("  注: MFU=40%, 使用云服务按需价格, 实际成本因优化策略和定价而异")
    print("  注: Chinchilla最优比例 ≈ 20x参数量的训练token数")

    # 同一模型不同GPU的成本对比
    print(f"\n  --- 7B模型在不同GPU上的训练成本对比 (32卡, 2T tokens, MFU=40%) ---")
    print(f"  {'GPU型号':<16} {'单卡算力(TFLOPS)':<18} {'训练天数':<12} {'估算成本($)':<14}")
    print(f"  {'-'*64}")

    for gpu_name in ["T4-16GB", "V100-16GB", "A100-80GB", "H100-80GB"]:
        result = estimator.estimate_training_cost(
            num_params=7e9,
            num_tokens=2e12,
            gpu_type=gpu_name,
            num_gpus=32,
            mfu=0.40,
        )
        gpu_spec = estimator.GPU_CATALOG[gpu_name]
        print(
            f"  {gpu_name:<16} "
            f"{gpu_spec.fp16_tflops:<18.0f} "
            f"{result['training_days']:<12.1f} "
            f"${result['cloud_cost_usd']:>12,.0f}"
        )


def demo_inference_cost():
    """演示3: 推理成本估算 - 不同吞吐量下的成本"""
    print("\n" + "=" * 70)
    print("演示3: 推理成本估算 ($/1M tokens)")
    print("=" * 70)

    estimator = CostEstimator()

    # 不同吞吐量下的推理成本
    throughputs = [100, 500, 1000, 2000, 5000]

    print(f"\n  --- A100-80GB 不同吞吐量下的推理成本 ---")
    print(f"  {'吞吐量(tokens/s)':<20} {'有效吞吐量':<14} {'$/1M tokens':<14} {'每小时处理tokens':<18}")
    print(f"  {'-'*68}")

    for tps in throughputs:
        result = estimator.estimate_inference_cost(
            gpu_type="A100-80GB",
            tokens_per_second=tps,
            utilization=0.70,
        )
        print(
            f"  {tps:<20} "
            f"{result['effective_tps']:<14.0f} "
            f"${result['cost_per_1m_tokens']:<13.4f} "
            f"{format_number(result['tokens_per_hour']):<18}"
        )

    # 不同GPU的推理成本对比 (固定吞吐量按比例)
    print(f"\n  --- 不同GPU的推理成本对比 (1000 tokens/s 基准, 利用率70%) ---")
    print(f"  {'GPU型号':<16} {'价格($/h)':<12} {'$/1M tokens':<14} {'性价比评分':<12}")
    print(f"  {'-'*56}")

    # 计算基准
    base_results = {}
    for gpu_name in ["T4-16GB", "L4-24GB", "V100-16GB", "A100-80GB", "H100-80GB"]:
        result = estimator.estimate_inference_cost(
            gpu_type=gpu_name,
            tokens_per_second=1000,
            utilization=0.70,
        )
        base_results[gpu_name] = result

    # 用A100的成本作为基准计算性价比评分
    a100_cost = base_results["A100-80GB"]["cost_per_1m_tokens"]
    for gpu_name, result in base_results.items():
        score = a100_cost / result["cost_per_1m_tokens"] * 100  # A100 = 100分
        print(
            f"  {gpu_name:<16} "
            f"${result['gpu_cost_per_hour']:<11.2f} "
            f"${result['cost_per_1m_tokens']:<13.4f} "
            f"{score:<12.0f}"
        )

    print("\n  注: 性价比评分以A100为100基准, 分数越高越好")
    print("  注: 实际吞吐量取决于模型大小、batch size、序列长度等因素")


def demo_real_cost_comparison():
    """演示4: 与真实基础设施成本对比"""
    print("\n" + "=" * 70)
    print("演示4: 真实LLM成本参考")
    print("=" * 70)

    print("""
  --- 知名模型训练成本估算 (公开报道/估算) ---
  +------------------+----------+-------------+------------------+------------------+
  | 模型             | 参数量   | 训练Token   | 估算GPU小时      | 估算成本($)      |
  +------------------+----------+-------------+------------------+------------------+
  | GPT-3            | 175B     | 300B        | ~3,640 V100天    | ~$4,600,000      |
  | LLaMA-2 7B       | 7B       | 2T          | 184,320 A100时   | ~$500,000        |
  | LLaMA-2 70B      | 70B      | 2T          | 1,720,320 A100时 | ~$5,000,000      |
  | Chinchilla (70B) | 70B      | 1.4T        | ~530,000 A100时  | ~$1,500,000      |
  | GPT-4 (传闻)     | ~1.8T MoE| ~13T        | -                | ~$100,000,000    |
  +------------------+----------+-------------+------------------+------------------+

  --- 主要云厂商API推理定价 (2024年参考) ---
  +---------------------+---------------------+---------------------+
  | 模型/服务           | 输入 ($/1M tokens)  | 输出 ($/1M tokens)  |
  +---------------------+---------------------+---------------------+
  | GPT-4o              | $2.50               | $10.00              |
  | GPT-4o-mini         | $0.15               | $0.60               |
  | Claude 3.5 Sonnet   | $3.00               | $15.00              |
  | Claude 3 Haiku      | $0.25               | $1.25               |
  | Llama 3.1 70B (云)  | $0.50 - $0.90       | $0.50 - $0.90       |
  | Llama 3.1 8B (云)   | $0.05 - $0.20       | $0.05 - $0.20       |
  +---------------------+---------------------+---------------------+

  --- 成本优化经验法则 ---
  1. 预留实例 vs 按需: 节省 30-60% (1-3年合约)
  2. 竞价实例 (Spot):  节省 60-90% (但可能被中断)
  3. 混合精度训练:     减少约50%显存, 吞吐量提升约2倍
  4. INT8量化推理:     成本降低约2倍, 精度损失极小
  5. INT4量化推理:     成本降低约4倍, 需要评估精度影响
  6. 持续批处理:       推理吞吐量提升2-5倍
  7. 投机解码:         推理速度提升1.5-3倍
  8. PagedAttention:    显存利用率提升, 吞吐量提升2-4倍
""")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("模块8: LLM成本建模与监控")
    print("纯CPU演示 - 无需GPU")
    print("=" * 70)

    # 打印GPU对比表
    estimator = CostEstimator()
    estimator.print_gpu_comparison()

    # 演示1: 资源监控
    demo_resource_monitor()

    # 演示2: 训练成本估算
    demo_training_cost()

    # 演示3: 推理成本估算
    demo_inference_cost()

    # 演示4: 真实成本对比
    demo_real_cost_comparison()

    print("=" * 70)
    print("演示完成! 本模块展示了LLM成本建模的核心概念:")
    print("  1. 资源监控: 使用psutil实时追踪CPU和内存使用")
    print("  2. 训练成本: 基于FLOPs和GPU规格估算训练费用")
    print("  3. 推理成本: 基于吞吐量估算每百万token的服务成本")
    print("  4. 真实参考: 对比业界公开的模型训练和API定价")
    print("=" * 70)
