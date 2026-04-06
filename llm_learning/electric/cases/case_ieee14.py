"""
IEEE 14节点标准测试系统

IEEE 14-Bus Test Case 是电力系统分析中最经典的测试算例之一。
该系统模型来自1960年代美国中西部电网的简化版本。

系统组成：
- 14个节点
- 5台发电机（节点1, 2, 3, 6, 8）
- 20条支路（含3台变压器）
- 11个负荷节点

节点1为平衡节点，节点2/3/6/8为PV节点（带电压调节的发电机）。

数据来源：
https://labs.ece.uw.edu/pstca/pf14/pg_tca14bus.htm
参考值来自MATPOWER (case14.m)

所有数据为标幺值，基准功率 Sbase = 100 MVA。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from power_flow.network import Bus, Branch, PowerNetwork, BusType


def create_ieee14_system():
    """创建IEEE 14节点测试系统"""
    net = PowerNetwork()

    # ========== 节点数据 ==========
    # 节点1: 平衡节点（Slack bus）
    net.add_bus(Bus(1, BusType.SLACK, v_mag=1.060, v_ang=0.0))

    # 节点2: PV节点 - 发电机
    net.add_bus(Bus(2, BusType.PV, p_gen=0.40, v_mag=1.045,
                    p_load=0.217, q_load=0.127))

    # 节点3: PV节点 - 发电机
    net.add_bus(Bus(3, BusType.PV, p_gen=0.0, v_mag=1.010,
                    p_load=0.942, q_load=0.190))

    # 节点4: PQ节点 - 负荷
    net.add_bus(Bus(4, BusType.PQ, p_load=0.478, q_load=-0.039))

    # 节点5: PQ节点 - 负荷
    net.add_bus(Bus(5, BusType.PQ, p_load=0.076, q_load=0.016))

    # 节点6: PV节点 - 同步调相机（发无功，有功为0）
    net.add_bus(Bus(6, BusType.PV, p_gen=0.0, v_mag=1.070,
                    p_load=0.112, q_load=0.075))

    # 节点7: PQ节点 - 无负荷，汇流节点
    net.add_bus(Bus(7, BusType.PQ))

    # 节点8: PV节点 - 同步调相机
    net.add_bus(Bus(8, BusType.PV, p_gen=0.0, v_mag=1.090))

    # 节点9: PQ节点 - 负荷
    net.add_bus(Bus(9, BusType.PQ, p_load=0.295, q_load=0.166,
                    b_shunt=0.19))

    # 节点10: PQ节点 - 负荷
    net.add_bus(Bus(10, BusType.PQ, p_load=0.090, q_load=0.058))

    # 节点11: PQ节点 - 负荷
    net.add_bus(Bus(11, BusType.PQ, p_load=0.035, q_load=0.018))

    # 节点12: PQ节点 - 负荷
    net.add_bus(Bus(12, BusType.PQ, p_load=0.061, q_load=0.016))

    # 节点13: PQ节点 - 负荷
    net.add_bus(Bus(13, BusType.PQ, p_load=0.135, q_load=0.058))

    # 节点14: PQ节点 - 负荷
    net.add_bus(Bus(14, BusType.PQ, p_load=0.149, q_load=0.050))

    # ========== 支路数据 ==========
    # 输电线路（20条支路）
    # Branch(from, to, R, X, B, tap)
    # tap=1.0 表示普通线路，tap≠1.0 表示变压器

    # --- 普通输电线路 ---
    net.add_branch(Branch(1, 2, 0.01938, 0.05917, 0.0528))
    net.add_branch(Branch(1, 5, 0.05403, 0.22304, 0.0492))
    net.add_branch(Branch(2, 3, 0.04699, 0.19797, 0.0438))
    net.add_branch(Branch(2, 4, 0.05811, 0.17632, 0.0374))
    net.add_branch(Branch(2, 5, 0.05695, 0.17388, 0.0340))
    net.add_branch(Branch(3, 4, 0.06701, 0.17103, 0.0346))
    net.add_branch(Branch(4, 5, 0.01335, 0.04211, 0.0128))
    net.add_branch(Branch(6, 11, 0.09498, 0.19890, 0.0))
    net.add_branch(Branch(6, 12, 0.12291, 0.25581, 0.0))
    net.add_branch(Branch(6, 13, 0.06615, 0.13027, 0.0))
    net.add_branch(Branch(9, 10, 0.03181, 0.08450, 0.0))
    net.add_branch(Branch(9, 14, 0.12711, 0.27038, 0.0))
    net.add_branch(Branch(10, 11, 0.08205, 0.19207, 0.0))
    net.add_branch(Branch(12, 13, 0.22092, 0.19988, 0.0))
    net.add_branch(Branch(13, 14, 0.17093, 0.34802, 0.0))

    # --- 变压器支路 ---
    # 节点4-7: 变压器，变比0.978
    net.add_branch(Branch(4, 7, 0.0, 0.20912, 0.0, tap=0.978))
    # 节点4-9: 变压器，变比0.969
    net.add_branch(Branch(4, 9, 0.0, 0.55618, 0.0, tap=0.969))
    # 节点5-6: 变压器，变比0.932
    net.add_branch(Branch(5, 6, 0.0, 0.25202, 0.0, tap=0.932))
    # 节点7-8: 变压器（无损耗，连接同步调相机）
    net.add_branch(Branch(7, 8, 0.0, 0.17615, 0.0))
    # 节点7-9: 变压器
    net.add_branch(Branch(7, 9, 0.0, 0.11001, 0.0))

    # 构建导纳矩阵
    net.build_y_bus()

    return net
