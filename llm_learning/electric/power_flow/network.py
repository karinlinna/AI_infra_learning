"""
电力网络数据结构模块

本模块定义了潮流计算所需的基本数据结构：
- Bus（节点）：电力系统中的母线，分为PQ、PV、Slack三种类型
- Branch（支路）：连接两个节点的输电线路或变压器
- PowerNetwork（电力网络）：管理所有节点和支路，构建导纳矩阵

关键概念：
- 标幺值(per-unit)系统：所有参数都用标幺值表示，简化不同电压等级的计算
- 导纳矩阵 Y_bus：描述网络拓扑和参数的核心矩阵，是潮流计算的基础
"""

import numpy as np
from enum import Enum


class BusType(Enum):
    """节点类型

    电力系统中每个节点有4个变量：P（有功功率）、Q（无功功率）、V（电压幅值）、θ（电压相角）
    不同类型的节点指定不同的已知量和待求量：

    - PQ节点（负荷节点）：已知 P、Q，求 V、θ
      大部分负荷节点都是PQ节点
    - PV节点（发电机节点）：已知 P、V，求 Q、θ
      有调压能力的发电机节点，可以控制电压幅值
    - SLACK节点（平衡节点）：已知 V、θ，求 P、Q
      系统中只有一个，用于平衡系统的功率差额（补偿网络损耗）
    """
    PQ = 1
    PV = 2
    SLACK = 3


class Bus:
    """电力系统节点（母线）

    参数说明（均为标幺值）：
        bus_id: 节点编号（从1开始）
        bus_type: 节点类型
        p_load: 有功负荷（消耗的有功功率）
        q_load: 无功负荷（消耗的无功功率）
        p_gen: 有功发电（发电机输出的有功功率）
        q_gen: 无功发电（发电机输出的无功功率，PQ节点已知，PV节点待求）
        v_mag: 电压幅值（PV和SLACK节点为给定值，PQ节点为初始猜测值）
        v_ang: 电压相角（弧度），SLACK节点通常设为0
        q_min/q_max: PV节点的无功出力上下限
        b_shunt: 并联电纳（无功补偿装置，如电容器）
    """
    def __init__(self, bus_id, bus_type=BusType.PQ,
                 p_load=0.0, q_load=0.0,
                 p_gen=0.0, q_gen=0.0,
                 v_mag=1.0, v_ang=0.0,
                 q_min=-999, q_max=999,
                 b_shunt=0.0):
        self.bus_id = bus_id
        self.bus_type = bus_type
        self.p_load = p_load
        self.q_load = q_load
        self.p_gen = p_gen
        self.q_gen = q_gen
        self.v_mag = v_mag
        self.v_ang = v_ang
        self.q_min = q_min
        self.q_max = q_max
        self.b_shunt = b_shunt

    @property
    def p_net(self):
        """净注入有功功率 = 发电 - 负荷"""
        return self.p_gen - self.p_load

    @property
    def q_net(self):
        """净注入无功功率 = 发电 - 负荷"""
        return self.q_gen - self.q_load

    def __repr__(self):
        return f"Bus({self.bus_id}, {self.bus_type.name}, V={self.v_mag:.4f}∠{np.degrees(self.v_ang):.2f}°)"


class Branch:
    """电力系统支路（输电线路或变压器）

    支路的π型等值电路模型：
    ┌─────┐
    ───┤  Z  ├───
    │  └─────┘  │
    ┤            ├
    │ y_shunt/2 │ y_shunt/2
    ┤            ├
    │            │
    ─            ─

    参数说明（均为标幺值）：
        from_bus: 起始节点编号
        to_bus: 终止节点编号
        r: 电阻（有功损耗）
        x: 电抗（磁场储能，通常远大于电阻）
        b: 线路充电电纳（长线路的对地电容效应）
        tap: 变压器变比（非变压器支路为1.0）
             tap = from侧电压 / to侧电压
    """
    def __init__(self, from_bus, to_bus, r, x, b=0.0, tap=1.0):
        self.from_bus = from_bus
        self.to_bus = to_bus
        self.r = r
        self.x = x
        self.b = b
        self.tap = tap

    @property
    def z(self):
        """支路阻抗 Z = R + jX"""
        return complex(self.r, self.x)

    @property
    def y(self):
        """支路导纳 Y = 1/Z = G + jB
        导纳是阻抗的倒数，在导纳矩阵中更方便使用
        """
        return 1.0 / self.z

    def __repr__(self):
        return f"Branch({self.from_bus}->{self.to_bus}, R={self.r}, X={self.x})"


class PowerNetwork:
    """电力网络

    管理所有节点和支路数据，核心功能是构建节点导纳矩阵 Y_bus。

    导纳矩阵 Y_bus 的构建规则：
    - 对角元素 Y_ii = Σ(与节点i相连的所有支路导纳) + 节点i的并联导纳
    - 非对角元素 Y_ij = -(节点i和j之间的支路导纳)（注意负号！）
    - Y_bus 是对称矩阵（当没有移相器时）

    带变压器的支路导纳矩阵修正：
    变比为t的变压器，其π型等值电路的导纳矩阵为：
    | y/t²      -y/t  |
    | -y/t       y    |
    """
    def __init__(self):
        self.buses = {}       # {bus_id: Bus}
        self.branches = []    # [Branch, ...]
        self.y_bus = None     # 导纳矩阵（numpy复数矩阵）
        self.n_bus = 0        # 节点数

    def add_bus(self, bus):
        self.buses[bus.bus_id] = bus
        self.n_bus = len(self.buses)

    def add_branch(self, branch):
        self.branches.append(branch)

    def build_y_bus(self):
        """构建节点导纳矩阵 Y_bus

        这是潮流计算的基础步骤。

        Y_bus 是一个 n×n 的复数矩阵（n为节点数），描述了整个网络的电气连接关系。

        构建步骤：
        1. 初始化为零矩阵
        2. 遍历每条支路，根据π型等值电路填充矩阵元素
        3. 加入各节点的并联电纳
        """
        n = self.n_bus
        # bus_id 可能不是从0开始的连续整数，需要建立映射
        self._bus_ids = sorted(self.buses.keys())
        self._id_to_idx = {bid: idx for idx, bid in enumerate(self._bus_ids)}

        self.y_bus = np.zeros((n, n), dtype=complex)

        # 遍历每条支路，填充导纳矩阵
        for br in self.branches:
            i = self._id_to_idx[br.from_bus]
            j = self._id_to_idx[br.to_bus]
            y = br.y          # 支路导纳
            t = br.tap        # 变压器变比
            b_half = 1j * br.b / 2  # 半线路充电导纳

            # 考虑变压器变比的π型等值电路
            # from侧（i侧）对角元素增加
            self.y_bus[i, i] += y / (t * t) + b_half
            # to侧（j侧）对角元素增加
            self.y_bus[j, j] += y + b_half
            # 互导纳（非对角元素，注意负号）
            self.y_bus[i, j] -= y / t
            self.y_bus[j, i] -= y / t

        # 加入各节点的并联电纳（无功补偿等）
        for bus_id, bus in self.buses.items():
            idx = self._id_to_idx[bus_id]
            self.y_bus[idx, idx] += 1j * bus.b_shunt

        return self.y_bus

    def get_bus_by_idx(self, idx):
        """通过矩阵索引获取节点对象"""
        return self.buses[self._bus_ids[idx]]

    def get_idx_by_id(self, bus_id):
        """通过节点编号获取矩阵索引"""
        return self._id_to_idx[bus_id]

    def get_pq_indices(self):
        """获取所有PQ节点的矩阵索引"""
        return [self._id_to_idx[bid] for bid, bus in self.buses.items()
                if bus.bus_type == BusType.PQ]

    def get_pv_indices(self):
        """获取所有PV节点的矩阵索引"""
        return [self._id_to_idx[bid] for bid, bus in self.buses.items()
                if bus.bus_type == BusType.PV]

    def get_slack_index(self):
        """获取平衡节点的矩阵索引"""
        for bid, bus in self.buses.items():
            if bus.bus_type == BusType.SLACK:
                return self._id_to_idx[bid]
        raise ValueError("网络中没有平衡节点(SLACK)")

    def summary(self):
        """打印网络摘要信息"""
        pq_count = sum(1 for b in self.buses.values() if b.bus_type == BusType.PQ)
        pv_count = sum(1 for b in self.buses.values() if b.bus_type == BusType.PV)
        slack_count = sum(1 for b in self.buses.values() if b.bus_type == BusType.SLACK)
        print(f"电力网络摘要:")
        print(f"  节点数: {self.n_bus} (PQ: {pq_count}, PV: {pv_count}, Slack: {slack_count})")
        print(f"  支路数: {len(self.branches)}")
