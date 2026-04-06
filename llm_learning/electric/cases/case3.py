"""
3节点简单算例

这是最小的有意义潮流计算算例，包含三种节点类型各一个：
- 节点1: SLACK（平衡节点），V=1.05, θ=0°
- 节点2: PV（发电机节点），P=0.5, V=1.02
- 节点3: PQ（负荷节点），P=-1.0, Q=-0.5

网络拓扑：
    (1) ----线路1---- (2)
     |                 |
    线路3             线路2
     |                 |
    (3) ──────────────(3)
    （节点3通过两条线路分别与节点1、2相连）

适合用于：
- 验证算法正确性（可手算对比）
- 理解牛顿-拉夫逊法的迭代过程
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from power_flow.network import Bus, Branch, PowerNetwork, BusType


def create_3bus_system():
    """创建3节点测试系统

    系统参数（标幺值，基准功率100MVA）：
    - 线路1 (1-2): R=0.01, X=0.05, B=0.02
    - 线路2 (2-3): R=0.02, X=0.06, B=0.03
    - 线路3 (1-3): R=0.03, X=0.08, B=0.02
    """
    net = PowerNetwork()

    # 添加节点
    # 节点1: 平衡节点 - 电压1.05pu，相角0°
    net.add_bus(Bus(1, BusType.SLACK, v_mag=1.05, v_ang=0.0))

    # 节点2: PV节点 - 发电0.5pu有功，电压维持1.02pu
    net.add_bus(Bus(2, BusType.PV, p_gen=0.5, v_mag=1.02))

    # 节点3: PQ节点 - 负荷1.0+j0.5 pu
    net.add_bus(Bus(3, BusType.PQ, p_load=1.0, q_load=0.5))

    # 添加支路
    net.add_branch(Branch(1, 2, r=0.01, x=0.05, b=0.02))
    net.add_branch(Branch(2, 3, r=0.02, x=0.06, b=0.03))
    net.add_branch(Branch(1, 3, r=0.03, x=0.08, b=0.02))

    # 构建导纳矩阵
    net.build_y_bus()

    return net
