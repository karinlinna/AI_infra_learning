"""
牛顿-拉夫逊法潮流计算模块

牛顿-拉夫逊法(Newton-Raphson Method)是工程中最常用的潮流计算方法。
它的核心思想是：将非线性功率方程在当前工作点进行线性化（泰勒展开取一阶），
然后迭代求解，直到功率不平衡量足够小。

算法流程：
1. 设定初始电压（平启动：所有PQ节点V=1.0, θ=0）
2. 计算每个节点的注入功率 P_calc, Q_calc
3. 计算功率不平衡量 ΔP = P_spec - P_calc, ΔQ = Q_spec - Q_calc
4. 如果最大不平衡量 < 容差，则收敛，结束
5. 构建雅可比矩阵 J（功率对电压的偏导数矩阵）
6. 求解线性方程 J · [Δθ, ΔV/V]ᵀ = [ΔP, ΔQ]ᵀ
7. 更新电压：θ += Δθ, V *= (1 + ΔV/V)
8. 回到步骤2

牛顿法的优点是具有二次收敛特性（每次迭代误差平方级下降），
通常4-6次迭代即可收敛到很高精度。
"""

import numpy as np
from .network import PowerNetwork, BusType


def newton_raphson_power_flow(network, tol=1e-6, max_iter=30, verbose=True):
    """牛顿-拉夫逊法潮流计算

    参数:
        network: PowerNetwork对象，需要已经调用过build_y_bus()
        tol: 收敛容差（功率不平衡量的最大允许值，标幺值）
        max_iter: 最大迭代次数
        verbose: 是否打印迭代过程

    返回:
        result: dict，包含计算结果
            - converged: bool, 是否收敛
            - iterations: int, 迭代次数
            - v_mag: ndarray, 各节点电压幅值
            - v_ang: ndarray, 各节点电压相角(弧度)
            - p_calc: ndarray, 各节点计算有功功率
            - q_calc: ndarray, 各节点计算无功功率
            - branch_flows: list, 各支路功率流
            - losses: complex, 网络总损耗
            - mismatch_history: list, 每次迭代的最大不平衡量（用于绘制收敛曲线）
    """
    if network.y_bus is None:
        network.build_y_bus()

    n = network.n_bus
    Y = network.y_bus

    # ========== 步骤1：初始化电压 ==========
    # 从各节点当前值初始化（PV/SLACK节点的V_mag已设好，PQ节点默认1.0）
    v_mag = np.array([network.get_bus_by_idx(i).v_mag for i in range(n)])
    v_ang = np.array([network.get_bus_by_idx(i).v_ang for i in range(n)])

    # 获取各类节点的索引
    pq_idx = network.get_pq_indices()
    pv_idx = network.get_pv_indices()
    slack_idx = network.get_slack_index()

    # 有功方程涉及的节点：PQ + PV（不含SLACK）
    pvpq_idx = sorted(pv_idx + pq_idx)

    # 已知的注入功率（指定值）
    p_spec = np.array([network.get_bus_by_idx(i).p_net for i in range(n)])
    q_spec = np.array([network.get_bus_by_idx(i).q_net for i in range(n)])

    mismatch_history = []

    if verbose:
        print("=" * 60)
        print("牛顿-拉夫逊潮流计算")
        print("=" * 60)
        print(f"节点数: {n}, PQ: {len(pq_idx)}, PV: {len(pv_idx)}")
        print(f"收敛容差: {tol}")
        print("-" * 60)

    for iteration in range(max_iter):
        # ========== 步骤2：计算注入功率 ==========
        # P_i + jQ_i = V_i · Σ(Y_ij · V_j)*  (共轭)
        # 展开为实部和虚部：
        # P_i = Σ |V_i||V_j|(G_ij·cos(θ_i-θ_j) + B_ij·sin(θ_i-θ_j))
        # Q_i = Σ |V_i||V_j|(G_ij·sin(θ_i-θ_j) - B_ij·cos(θ_i-θ_j))
        p_calc, q_calc = _calc_power_injection(v_mag, v_ang, Y, n)

        # ========== 步骤3：计算功率不平衡量 ==========
        dp = p_spec - p_calc  # 有功不平衡量
        dq = q_spec - q_calc  # 无功不平衡量

        # 构建不平衡量向量（只取需要的节点）
        # ΔP: PV和PQ节点的有功不平衡
        # ΔQ: 只有PQ节点的无功不平衡（PV节点的Q是待求量）
        mismatch = np.concatenate([dp[pvpq_idx], dq[pq_idx]])
        max_mismatch = np.max(np.abs(mismatch))
        mismatch_history.append(max_mismatch)

        if verbose:
            print(f"迭代 {iteration + 1}: 最大不平衡量 = {max_mismatch:.2e}")

        # ========== 步骤4：判断收敛 ==========
        if max_mismatch < tol:
            if verbose:
                print("-" * 60)
                print(f"收敛! 共迭代 {iteration + 1} 次")
                print("=" * 60)

            # 重新计算最终的功率
            p_calc, q_calc = _calc_power_injection(v_mag, v_ang, Y, n)

            # 更新各节点的计算结果
            for i in range(n):
                bus = network.get_bus_by_idx(i)
                bus.v_mag = v_mag[i]
                bus.v_ang = v_ang[i]

            # 计算支路功率流
            branch_flows = _calc_branch_flows(network, v_mag, v_ang)
            total_loss = sum(f['loss'] for f in branch_flows)

            return {
                'converged': True,
                'iterations': iteration + 1,
                'v_mag': v_mag.copy(),
                'v_ang': v_ang.copy(),
                'p_calc': p_calc,
                'q_calc': q_calc,
                'branch_flows': branch_flows,
                'losses': total_loss,
                'mismatch_history': mismatch_history,
            }

        # ========== 步骤5：构建雅可比矩阵 ==========
        J = _build_jacobian(v_mag, v_ang, Y, n, pvpq_idx, pq_idx)

        # ========== 步骤6：求解修正方程 ==========
        # J · Δx = mismatch
        # Δx = [Δθ_pvpq, ΔV/V_pq]
        dx = np.linalg.solve(J, mismatch)

        # ========== 步骤7：更新电压 ==========
        n_pvpq = len(pvpq_idx)
        # 更新相角（PV和PQ节点）
        v_ang[pvpq_idx] += dx[:n_pvpq]
        # 更新电压幅值（只有PQ节点，PV节点电压幅值固定）
        v_mag[pq_idx] *= (1 + dx[n_pvpq:])

    # 超过最大迭代次数仍未收敛
    if verbose:
        print(f"警告: 达到最大迭代次数 {max_iter}，未收敛!")
    return {
        'converged': False,
        'iterations': max_iter,
        'v_mag': v_mag.copy(),
        'v_ang': v_ang.copy(),
        'p_calc': p_calc,
        'q_calc': q_calc,
        'branch_flows': [],
        'losses': 0,
        'mismatch_history': mismatch_history,
    }


def _calc_power_injection(v_mag, v_ang, Y, n):
    """计算各节点的注入功率

    功率注入公式：
    S_i = V_i · (Σ Y_ij · V_j)*     （*表示共轭）

    展开为极坐标形式：
    P_i = Σ_j |V_i|·|V_j|·(G_ij·cos(θ_i - θ_j) + B_ij·sin(θ_i - θ_j))
    Q_i = Σ_j |V_i|·|V_j|·(G_ij·sin(θ_i - θ_j) - B_ij·cos(θ_i - θ_j))

    其中 G_ij + jB_ij = Y_ij（导纳矩阵元素的实部和虚部）
    """
    p_calc = np.zeros(n)
    q_calc = np.zeros(n)

    G = Y.real  # 电导矩阵
    B = Y.imag  # 电纳矩阵

    for i in range(n):
        for j in range(n):
            angle_diff = v_ang[i] - v_ang[j]
            p_calc[i] += v_mag[i] * v_mag[j] * (
                G[i, j] * np.cos(angle_diff) + B[i, j] * np.sin(angle_diff)
            )
            q_calc[i] += v_mag[i] * v_mag[j] * (
                G[i, j] * np.sin(angle_diff) - B[i, j] * np.cos(angle_diff)
            )

    return p_calc, q_calc


def _build_jacobian(v_mag, v_ang, Y, n, pvpq_idx, pq_idx):
    """构建雅可比矩阵

    雅可比矩阵是潮流方程对状态变量的偏导数矩阵，结构为 2×2 分块：

    J = | J1 (∂P/∂θ)   J2 (∂P/∂V) |
        | J3 (∂Q/∂θ)   J4 (∂Q/∂V) |

    - J1: 有功功率对相角的偏导（维度: n_pvpq × n_pvpq）
    - J2: 有功功率对电压幅值的偏导（维度: n_pvpq × n_pq）
    - J3: 无功功率对相角的偏导（维度: n_pq × n_pvpq）
    - J4: 无功功率对电压幅值的偏导（维度: n_pq × n_pq）

    注意：为了数值稳定性，实际使用的是 ΔV/V 而不是 ΔV，
    所以J2和J4需要乘以V_j。
    """
    G = Y.real
    B = Y.imag

    n_pvpq = len(pvpq_idx)
    n_pq = len(pq_idx)
    j_size = n_pvpq + n_pq

    J = np.zeros((j_size, j_size))

    # 计算完整的 n×n 雅可比子矩阵，然后提取需要的行列
    # J1: ∂P_i/∂θ_j
    J1_full = np.zeros((n, n))
    # J2: V_j · ∂P_i/∂V_j
    J2_full = np.zeros((n, n))
    # J3: ∂Q_i/∂θ_j
    J3_full = np.zeros((n, n))
    # J4: V_j · ∂Q_i/∂V_j
    J4_full = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            angle_diff = v_ang[i] - v_ang[j]
            if i == j:
                # 对角元素公式（i == j 时的偏导数）
                # 需要先计算求和项
                sum_p = 0.0
                sum_q = 0.0
                for k in range(n):
                    if k != i:
                        aik = v_ang[i] - v_ang[k]
                        sum_p += v_mag[k] * (G[i, k] * np.sin(aik) - B[i, k] * np.cos(aik))
                        sum_q += v_mag[k] * (G[i, k] * np.cos(aik) + B[i, k] * np.sin(aik))

                # ∂P_i/∂θ_i = V_i · Σ_{k≠i} V_k · (-G_ik·sin(θ_ik) + B_ik·cos(θ_ik))
                #            = -V_i · sum_p  (注意负号来自求导)
                J1_full[i, i] = -v_mag[i] * sum_p

                # V_i · ∂P_i/∂V_i = V_i · (2·V_i·G_ii + sum_q)
                J2_full[i, i] = v_mag[i] * (2 * v_mag[i] * G[i, i] + sum_q)

                # ∂Q_i/∂θ_i = V_i · sum_q
                J3_full[i, i] = v_mag[i] * sum_q

                # V_i · ∂Q_i/∂V_i = V_i · (-2·V_i·B_ii + sum_p_for_q)
                # sum_p_for_q 实际上就是之前的 sum_p 带符号变换
                sum_for_q4 = 0.0
                for k in range(n):
                    if k != i:
                        aik = v_ang[i] - v_ang[k]
                        sum_for_q4 += v_mag[k] * (G[i, k] * np.sin(aik) - B[i, k] * np.cos(aik))
                J4_full[i, i] = v_mag[i] * (-2 * v_mag[i] * B[i, i] + sum_for_q4)
            else:
                # 非对角元素公式（i ≠ j 时的偏导数）
                # ∂P_i/∂θ_j = V_i·V_j·(G_ij·sin(θ_ij) - B_ij·cos(θ_ij))
                J1_full[i, j] = v_mag[i] * v_mag[j] * (
                    G[i, j] * np.sin(angle_diff) - B[i, j] * np.cos(angle_diff)
                )
                # V_j·∂P_i/∂V_j = V_i·V_j·(G_ij·cos(θ_ij) + B_ij·sin(θ_ij))
                J2_full[i, j] = v_mag[i] * v_mag[j] * (
                    G[i, j] * np.cos(angle_diff) + B[i, j] * np.sin(angle_diff)
                )
                # ∂Q_i/∂θ_j = V_i·V_j·(-G_ij·cos(θ_ij) - B_ij·sin(θ_ij))
                J3_full[i, j] = v_mag[i] * v_mag[j] * (
                    -G[i, j] * np.cos(angle_diff) - B[i, j] * np.sin(angle_diff)
                )
                # V_j·∂Q_i/∂V_j = V_i·V_j·(G_ij·sin(θ_ij) - B_ij·cos(θ_ij))
                J4_full[i, j] = v_mag[i] * v_mag[j] * (
                    G[i, j] * np.sin(angle_diff) - B[i, j] * np.cos(angle_diff)
                )

    # 从完整矩阵中提取需要的行列，组装最终的雅可比矩阵
    # J = | J1[pvpq, pvpq]  J2[pvpq, pq] |
    #     | J3[pq, pvpq]    J4[pq, pq]   |
    J[:n_pvpq, :n_pvpq] = J1_full[np.ix_(pvpq_idx, pvpq_idx)]
    J[:n_pvpq, n_pvpq:] = J2_full[np.ix_(pvpq_idx, pq_idx)]
    J[n_pvpq:, :n_pvpq] = J3_full[np.ix_(pq_idx, pvpq_idx)]
    J[n_pvpq:, n_pvpq:] = J4_full[np.ix_(pq_idx, pq_idx)]

    return J


def _calc_branch_flows(network, v_mag, v_ang):
    """计算各支路的功率流

    支路功率流的计算：
    从节点i看向节点j的功率：
    S_ij = V_i · (V_i - V_j)* · y* + V_i · V_i* · (jb/2)*

    返回每条支路的：
    - from侧功率 S_ij
    - to侧功率 S_ji
    - 支路损耗 = S_ij + S_ji（实部为有功损耗，虚部为无功损耗）
    """
    branch_flows = []

    for br in network.branches:
        i = network.get_idx_by_id(br.from_bus)
        j = network.get_idx_by_id(br.to_bus)

        # 节点电压（复数形式）
        Vi = v_mag[i] * np.exp(1j * v_ang[i])
        Vj = v_mag[j] * np.exp(1j * v_ang[j])

        y = br.y
        t = br.tap
        b_half = 1j * br.b / 2

        # from -> to 方向的电流
        I_ij = (Vi / t - Vj) * y / t + Vi * b_half / (t * t)
        # to -> from 方向的电流
        I_ji = (Vj - Vi / t) * y + Vj * b_half

        # 功率 = 电压 × 电流共轭
        S_ij = Vi / t * np.conj(I_ij)
        S_ji = Vj * np.conj(I_ji)

        # 支路损耗
        loss = S_ij + S_ji

        branch_flows.append({
            'from': br.from_bus,
            'to': br.to_bus,
            'p_from': S_ij.real,
            'q_from': S_ij.imag,
            'p_to': S_ji.real,
            'q_to': S_ji.imag,
            'loss': loss,
        })

    return branch_flows


def print_results(network, result):
    """格式化打印潮流计算结果"""
    if not result['converged']:
        print("潮流计算未收敛!")
        return

    print("\n" + "=" * 70)
    print("潮流计算结果")
    print("=" * 70)

    # 节点结果
    print(f"\n{'节点':>4} {'类型':>6} {'电压(pu)':>10} {'相角(°)':>10} "
          f"{'P_gen':>10} {'Q_gen':>10} {'P_load':>10} {'Q_load':>10}")
    print("-" * 70)

    n = network.n_bus
    for i in range(n):
        bus = network.get_bus_by_idx(i)
        p = result['p_calc'][i]
        q = result['q_calc'][i]
        print(f"{bus.bus_id:>4} {bus.bus_type.name:>6} "
              f"{result['v_mag'][i]:>10.4f} {np.degrees(result['v_ang'][i]):>10.4f} "
              f"{bus.p_gen:>10.4f} {bus.q_gen:>10.4f} "
              f"{bus.p_load:>10.4f} {bus.q_load:>10.4f}")

    # 支路功率流
    if result['branch_flows']:
        print(f"\n{'支路':>10} {'P_from':>10} {'Q_from':>10} "
              f"{'P_to':>10} {'Q_to':>10} {'P_loss':>10} {'Q_loss':>10}")
        print("-" * 70)

        for bf in result['branch_flows']:
            print(f"{bf['from']:>4}->{bf['to']:<4} "
                  f"{bf['p_from']:>10.4f} {bf['q_from']:>10.4f} "
                  f"{bf['p_to']:>10.4f} {bf['q_to']:>10.4f} "
                  f"{bf['loss'].real:>10.4f} {bf['loss'].imag:>10.4f}")

    # 网络损耗
    total_loss = result['losses']
    print(f"\n总有功损耗: {total_loss.real:.6f} pu")
    print(f"总无功损耗: {total_loss.imag:.6f} pu")
