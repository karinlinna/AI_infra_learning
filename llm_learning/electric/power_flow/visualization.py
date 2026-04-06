"""
潮流计算结果可视化模块

提供三种可视化：
1. 电压幅值分布图 - 直观展示各节点电压水平
2. 收敛曲线 - 展示牛顿法的二次收敛特性
3. 网络拓扑图 - 展示电网结构和功率流向
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from power_flow.network import BusType

# 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_voltage_profile(result, network, title="电压幅值分布", save_path=None):
    """绘制各节点电压幅值柱状图

    在实际电力系统中，电压幅值应维持在 0.95~1.05 pu 范围内。
    超出此范围可能导致设备损坏或无法正常工作。
    """
    n = network.n_bus
    bus_ids = [network.get_bus_by_idx(i).bus_id for i in range(n)]
    v_mag = result['v_mag']

    fig, ax = plt.subplots(figsize=(10, 5))

    # 根据电压水平着色：绿色=正常，黄色=偏低/偏高，红色=严重越限
    colors = []
    for v in v_mag:
        if 0.95 <= v <= 1.05:
            colors.append('#2ecc71')   # 绿色 - 正常
        elif 0.90 <= v <= 1.10:
            colors.append('#f39c12')   # 黄色 - 注意
        else:
            colors.append('#e74c3c')   # 红色 - 越限

    bars = ax.bar(range(n), v_mag, color=colors, edgecolor='black', linewidth=0.5)

    # 添加电压合格范围参考线
    ax.axhline(y=1.05, color='orange', linestyle='--', linewidth=1, label='上限 1.05 pu')
    ax.axhline(y=1.00, color='gray', linestyle=':', linewidth=0.8, label='额定 1.00 pu')
    ax.axhline(y=0.95, color='orange', linestyle='--', linewidth=1, label='下限 0.95 pu')

    # 在柱上标注数值
    for i, (bar, v) in enumerate(zip(bars, v_mag)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('节点编号')
    ax.set_ylabel('电压幅值 (pu)')
    ax.set_title(title)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(bid) for bid in bus_ids])
    ax.set_ylim(min(v_mag) - 0.05, max(v_mag) + 0.05)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    plt.show()


def plot_convergence(result, title="收敛曲线", save_path=None):
    """绘制牛顿-拉夫逊法的收敛曲线

    牛顿法的特点是二次收敛（quadratic convergence）：
    每次迭代后，误差大约变为上一次的平方。
    在对数坐标下，收敛曲线的斜率应逐渐增大。
    """
    history = result['mismatch_history']
    iterations = range(1, len(history) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(iterations, history, 'bo-', linewidth=2, markersize=8, label='最大功率不平衡量')

    # 标注每个点的数值
    for i, (x, y) in enumerate(zip(iterations, history)):
        ax.annotate(f'{y:.2e}', (x, y), textcoords="offset points",
                    xytext=(10, 5), fontsize=8, color='blue')

    ax.set_xlabel('迭代次数')
    ax.set_ylabel('最大功率不平衡量 (pu)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 标注收敛特性
    if len(history) >= 3:
        ax.text(0.02, 0.02,
                f'收敛次数: {len(history)}\n'
                f'最终精度: {history[-1]:.2e}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    plt.show()


def plot_network(network, result, title="电力网络拓扑", save_path=None):
    """绘制电力网络拓扑图

    节点颜色表示电压水平，支路粗细表示功率流大小，
    箭头表示有功功率流向。
    """
    n = network.n_bus

    # 为节点分配位置（简单的圆形布局）
    if n <= 5:
        # 小型系统用圆形布局
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {network.get_bus_by_idx(i).bus_id:
               (2 * np.cos(angles[i]), 2 * np.sin(angles[i]))
               for i in range(n)}
    else:
        # IEEE 14节点系统的预设位置（接近实际地理布局）
        pos = _ieee14_positions()

    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制支路（线路）
    if result.get('branch_flows'):
        max_flow = max(abs(bf['p_from']) for bf in result['branch_flows']) or 1
        for bf in result['branch_flows']:
            from_id, to_id = bf['from'], bf['to']
            if from_id in pos and to_id in pos:
                x1, y1 = pos[from_id]
                x2, y2 = pos[to_id]
                # 线宽与功率流大小成正比
                lw = 1 + 3 * abs(bf['p_from']) / max_flow
                ax.plot([x1, x2], [y1, y2], 'gray', linewidth=lw, alpha=0.6, zorder=1)

                # 在支路中点标注功率
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mx, my, f"{abs(bf['p_from']):.3f}",
                        fontsize=7, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))

    # 绘制节点
    v_mag = result['v_mag']
    for i in range(n):
        bus = network.get_bus_by_idx(i)
        if bus.bus_id not in pos:
            continue
        x, y = pos[bus.bus_id]

        # 节点颜色
        v = v_mag[i]
        if bus.bus_type == BusType.SLACK:
            color = '#e74c3c'     # 红色 - 平衡节点
            marker_size = 500
        elif bus.bus_type == BusType.PV:
            color = '#3498db'     # 蓝色 - PV节点
            marker_size = 400
        else:
            color = '#2ecc71'     # 绿色 - PQ节点
            marker_size = 300

        ax.scatter(x, y, s=marker_size, c=color, edgecolors='black',
                   linewidths=1.5, zorder=3)

        # 节点标签
        ax.annotate(f'Bus {bus.bus_id}\n{v:.4f} pu\n{np.degrees(result["v_ang"][i]):.2f}°',
                    (x, y), textcoords="offset points", xytext=(15, 10),
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))


    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=12, label='Slack 节点'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=12, label='PV 节点 (发电机)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
               markersize=12, label='PQ 节点 (负荷)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    plt.show()


def _ieee14_positions():
    """IEEE 14节点系统的节点位置（手动布局，接近标准图）"""
    return {
        1:  (0, 4),
        2:  (2, 4),
        3:  (4, 4),
        4:  (3, 2.5),
        5:  (1, 2.5),
        6:  (0, 0),
        7:  (3, 1),
        8:  (5, 1),
        9:  (3.5, 0),
        10: (2, -0.5),
        11: (1, -0.5),
        12: (-1, -0.5),
        13: (-0.5, -1.5),
        14: (2.5, -1.5),
    }
