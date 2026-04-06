"""
电力系统潮流计算 - 学习项目

本项目从零实现牛顿-拉夫逊法潮流计算，包含：
1. 3节点简单算例 - 适合入门理解
2. IEEE 14节点标准算例 - 工业界经典测试

运行方式:
    python main.py

依赖:
    pip install numpy matplotlib
"""

import sys
import os
import numpy as np

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from power_flow.newton_raphson import newton_raphson_power_flow, print_results
from power_flow.visualization import plot_voltage_profile, plot_convergence, plot_network
from cases.case3 import create_3bus_system
from cases.case_ieee14 import create_ieee14_system


def run_case3():
    """运行3节点简单算例"""
    print("\n" + "#" * 70)
    print("#  算例1：3节点简单系统")
    print("#" * 70)

    net = create_3bus_system()
    net.summary()

    # 打印导纳矩阵（小系统可以看看）
    print("\n导纳矩阵 Y_bus:")
    print(np.array2string(net.y_bus, precision=4, suppress_small=True))

    # 运行潮流计算
    result = newton_raphson_power_flow(net, tol=1e-8, verbose=True)
    print_results(net, result)

    # 可视化
    plot_voltage_profile(result, net, title="3节点系统 - 电压幅值分布")
    plot_convergence(result, title="3节点系统 - NR法收敛曲线")
    plot_network(net, result, title="3节点系统 - 网络拓扑")


def run_ieee14():
    """运行IEEE 14节点标准算例"""
    print("\n" + "#" * 70)
    print("#  算例2：IEEE 14节点标准测试系统")
    print("#" * 70)

    net = create_ieee14_system()
    net.summary()

    # 运行潮流计算
    result = newton_raphson_power_flow(net, tol=1e-8, verbose=True)
    print_results(net, result)

    # 与MATPOWER参考值对比
    print("\n" + "=" * 70)
    print("与MATPOWER参考值对比（电压幅值）")
    print("=" * 70)
    # MATPOWER case14 参考电压幅值
    matpower_v = {
        1: 1.0600, 2: 1.0450, 3: 1.0100, 4: 1.0177, 5: 1.0195,
        6: 1.0700, 7: 1.0615, 8: 1.0900, 9: 1.0559, 10: 1.0510,
        11: 1.0569, 12: 1.0552, 13: 1.0504, 14: 1.0355,
    }
    print(f"{'节点':>4} {'计算值':>10} {'参考值':>10} {'误差':>12}")
    print("-" * 40)
    for i in range(net.n_bus):
        bus = net.get_bus_by_idx(i)
        bid = bus.bus_id
        v_calc = result['v_mag'][i]
        v_ref = matpower_v.get(bid, 0)
        err = abs(v_calc - v_ref)
        status = "OK" if err < 0.005 else "!!"
        print(f"{bid:>4} {v_calc:>10.4f} {v_ref:>10.4f} {err:>10.6f}  {status}")

    # 可视化
    plot_voltage_profile(result, net, title="IEEE 14节点系统 - 电压幅值分布")
    plot_convergence(result, title="IEEE 14节点系统 - NR法收敛曲线")
    plot_network(net, result, title="IEEE 14节点系统 - 网络拓扑")


if __name__ == "__main__":
    print("=" * 70)
    print("  电力系统潮流计算 - 牛顿-拉夫逊法")
    print("  Power Flow Analysis - Newton-Raphson Method")
    print("=" * 70)

    # 运行两个算例
    run_case3()
    run_ieee14()

    print("\n计算完成!")
