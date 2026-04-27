# initial_condition.py
import numpy as np
import torch
from config import *
# ==================== 全局参数 (对应论文 Example 6.2.1) ====================
# 计算域: Ω = (-1, 1)^2
HALF_L = DOMAIN_SIZE / 2.0

# ========== 气泡闭合 (Closing of a Void) 全局参数 ==========
# 论文参数: r1 = 0.2 (内半径), r2 = 0.55 (外半径)
# 注意：这里定义的是界面的中心位置
R_INNER = 0.20   # 内部空洞的半径 (即将闭合的气泡边界)
R_OUTER = 0.55   # 外部粒子的半径

# 对数势 binodal 点 (u_bi^θ)
# 根据论文公式 (2) 和 Section 2.1
THETA = 0.2
THETA_C = 1.0
# u_bi = tanh(sqrt((Theta_c - Theta) / Theta))
u_bi_val = np.tanh(np.sqrt((THETA_C - THETA) / THETA))

# 界面厚度参数 (与 PDE 中的 gamma/eps 一致)
GAMMA_IC = 0.04 

# ========== 初始条件函数：气泡闭合 ==========
def compute_initial_condition(x, y):
    """
    计算 Cahn-Hilliard 方程的“气泡闭合” (Closing of a Void) 初始条件。
    
    几何结构:
    - 一个环形区域 (Annulus)，内半径 R_INNER，外半径 R_OUTER。
    - 环内 (r < R_INNER) 为 +1 相 (背景相)。
    - 环上 (R_INNER < r < R_OUTER) 为 -1 相 (粒子相)。
    - 环外 (r > R_OUTER) 为 +1 相 (背景相)。
    
    演化预期:
    - 内部空洞 (r < R_INNER 的 +1 区域) 会逐渐缩小直至消失。
    - 最终形成一个实心的 -1 相圆形粒子。
    """
    # 统一输入为 torch.Tensor
    if isinstance(x, np.ndarray):
        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)
    else:
        X, Y = x, y

    # 1. 计算到原点的距离 r = sqrt(x^2 + y^2)
    r = torch.sqrt(X**2 + Y**2)

    # 2. 构造有符号距离函数 (Signed Distance Function)
    # 我们需要一个函数 d(x)，使得:
    # - 在 -1 相区域 (环带内)，d > 0
    # - 在 +1 相区域 (环带外和空洞内)，d < 0
    
    # 距离内边界的距离 (内部为正): d1 = R_INNER - r
    # 如果 r < R_INNER, d1 > 0 (空洞内部) -> 但我们希望空洞是 +1 相，所以这里逻辑要反转
    # 让我们重新定义目标：
    # 目标：环带区域 (R_INNER < r < R_OUTER) 是 -1 相。
    # 其他区域是 +1 相。
    
    # 使用论文公式 (15) 的逻辑: d(x) = max(-d1, d2) 
    # 其中 d1 = r - r1 (距离内圈的距离，圈内为负), d2 = r2 - r (距离外圈的距离，圈外为负)
    # 修正理解论文公式 (15): 
    # d_j(x) = |x| - r_j. 
    # d(x) = max(-d1, d2) = max(-(r - r1), r2 - r) = max(r1 - r, r2 - r)
    # 让我们测试一下这个 d(x):
    # Case A: r < r1 (空洞内). r1-r > 0, r2-r > 0. max 是 r2-r (很大的正数). -> tanh(正) = +1. (错误，我们要空洞是+1，但环带是-1)
    # 等等，论文公式 (15) 定义 u~0 = -tanh(d / ...).
    # 如果 d > 0, u ~ -1. 如果 d < 0, u ~ +1.
    # 我们希望:
    # r < r1 (+1 相) => 需要 d < 0
    # r1 < r < r2 (-1 相) => 需要 d > 0
    # r > r2 (+1 相) => 需要 d < 0
    
    # 重新构造 d(x):
    # 距离内边界的“内部”距离: dist_inner = r - R_INNER (r>R_INNER 为正)
    # 距离外边界的“内部”距离: dist_outer = R_OUTER - r (r<R_OUTER 为正)
    # 只有当两者都为正时 (即在环带内)，我们才希望 d > 0.
    # 取最小值可以實現交集逻辑: d_temp = min(r - R_INNER, R_OUTER - r)
    # 如果 r < R_INNER: r - R_INNER < 0 -> min < 0. (OK, +1 相)
    # 如果 R_INNER < r < R_OUTER: 两者都 > 0 -> min > 0. (OK, -1 相)
    # 如果 r > R_OUTER: R_OUTER - r < 0 -> min < 0. (OK, +1 相)
    
    d_ring = torch.minimum(r - R_INNER, R_OUTER - r)
    
    # 3. 应用双曲正切剖面
    # 论文公式: u = -tanh(d / (sqrt(2)*gamma))
    # 注意：这里加了一个负号，是因为通常 tanh(x>0)=1, 但我们希望环带(d>0)是 -1 相。
    ic = -torch.tanh(d_ring / (np.sqrt(2) * GAMMA_IC))

    # 4. (可选) 严格截断到 binodal 点，防止初始值过于接近 ±1 导致对数势爆炸
    # 虽然 tanh 不会达到 ±1，但在对数势模拟中，限制在 [-u_bi, u_bi] 往往更稳定
    # ic = torch.clamp(ic, -u_bi_val, u_bi_val)
    
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