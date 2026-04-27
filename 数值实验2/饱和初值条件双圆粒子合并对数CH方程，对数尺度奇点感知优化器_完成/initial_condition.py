# initial_condition.py
import numpy as np
import torch
from config import *
# ==================== 全局参数 (与论文 Example 6.2.3 一致) ====================
# 计算域: Ω = (-1, 1)^2
HALF_L = DOMAIN_SIZE / 2.0

# ========== 双粒子合并问题的全局参数 ==========
# 粒子中心 (论文: m_{1/2} = ±(1/4 + 2γ, 0))
GAMMA_IC = 0.04 # 这里使用与 PDE 相同的 EPS/gamma
CENTER_OFFSET = 0.25 + 2 * GAMMA_IC
m1 = (-CENTER_OFFSET, 0.0)  # 左侧粒子中心
m2 = ( CENTER_OFFSET, 0.0)  # 右侧粒子中心
RADIUS = 0.3                # 粒子半径  

# 对数势 binodal 点 (u_bi^θ)
THETA = 0.2
THETA_C = 1.0
u_bi = np.tanh(np.sqrt((THETA_C - THETA) / THETA))

# ========== 新的初始条件函数：双粒子合并 ==========
def compute_initial_condition(x, y):
    """
    计算 Cahn-Hilliard 方程的双粒子合并初始条件。
    粒子内部为 -1 相，外部为 +1 相。
    """
    # 统一输入为 torch.Tensor
    if isinstance(x, np.ndarray):
        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)
    else:
        X, Y = x, y

    # 计算到两个粒子中心的距离
    dist1 = torch.sqrt((X - m1[0])**2 + (Y - m1[1])**2)
    dist2 = torch.sqrt((X - m2[0])**2 + (Y - m2[1])**2)

    # 有符号距离函数：在任一粒子内部即为正值
    d1 = RADIUS - dist1
    d2 = RADIUS - dist2
    d = torch.maximum(d1, d2) # 取最大值，表示离最近边界的距离

    # 关键：内部(正值)对应-1相，外部(负值)对应+1相
    # 使用论文中的 gamma (即 EPS) 作为界面厚度
    ic = -torch.tanh(d / (np.sqrt(2) * GAMMA_IC))

    return ic


# # initial_condition.py
# import numpy as np
# import torch

# # ==================== 全局参数 (与论文 Example 6.2.3 一致) ====================
# # 计算域: Ω = (-1, 1)^2
# DOMAIN_SIZE = 2.0  # 域的宽度/高度
# HALF_L = DOMAIN_SIZE / 2.0

# # ========== 温和初值参数（远离对数势奇点）==========
# U_MAX = 0.5  # 最大绝对值，确保 |u| <= U_MAX < 1（推荐 0.4 ~ 0.6）
# # 可添加更多频率控制参数（如波数）

# # ========== 新的初始条件函数：温和光滑扰动（远离奇点）==========
# def compute_initial_condition(x, y):
#     """
#     计算 Cahn-Hilliard 方程的温和初始条件，适用于对数非线性项。
#     初始值严格满足 |u| <= U_MAX < 1，远离奇点 u=±1。
    
#     使用确定性光滑函数组合（正弦波），保证可复现性和光滑性。
#     """
#     # 统一输入为 torch.Tensor
#     if isinstance(x, np.ndarray):
#         X = torch.tensor(x, dtype=torch.float32)
#         Y = torch.tensor(y, dtype=torch.float32)
#     else:
#         X, Y = x, y

#     # 将坐标从 [-1, 1] 映射到 [0, 2π] 便于使用三角函数
#     # 注意：原域是 (-1, 1)^2，所以 x ∈ [-1, 1]
#     k1 = 2 * np.pi / DOMAIN_SIZE  # 基频
#     k2 = 4 * np.pi / DOMAIN_SIZE  # 高频

#     # 构造温和扰动（可自由调整组合）
#     perturbation = (
#         0.6 * torch.sin(k1 * (X + 1.0)) * torch.cos(k1 * (Y + 1.0)) +
#         0.4 * torch.sin(k2 * (X + 1.0))
#     )

#     # 归一化到 [-1, 1] 范围（理论最大值为 0.6+0.4=1.0，但实际略小）
#     # 为安全起见，显式缩放并限制幅度
#     u0 = U_MAX * perturbation / 1.0  # 分母为理论最大振幅（此处为1.0）

#     # 可选：进一步裁剪确保绝对安全（通常不需要）
#     # u0 = torch.clamp(u0, -U_MAX, U_MAX)

#     return u0