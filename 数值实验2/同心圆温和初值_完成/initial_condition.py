
# initial_condition.py
import numpy as np
import torch

# ==================== 全局参数 ====================
DOMAIN_SIZE = 2.0  # 域的宽度/高度 (-1 到 1)
U_MAX = 1.0        # 最大绝对值，确保远离对数势奇点 ±1

def compute_initial_condition(x, y):
    """
    计算基于径向距离的同心圆环状初始条件。
    形态：以原点为中心的同心圆波纹。
    适用场景：测试曲线界面的演化和曲率效应。
    
    Args:
        x, y: 坐标网格 (np.ndarray 或 torch.Tensor)
        
    Returns:
        torch.Tensor: 初始相场变量 u0
    """
    # 统一输入为 torch.Tensor
    if isinstance(x, np.ndarray):
        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)
    else:
        X, Y = x, y

    # 1. 计算径向距离 r (从中心 0,0 开始)
    # 域是 [-1, 1]，所以 r 的范围大约是 [0, sqrt(2)]
    R = torch.sqrt(X**2 + Y**2)
    
    # 2. 构造径向波动
    # 频率参数：控制圆环的密度。
    # 这里使用 3*pi 使得在半径 ~1.4 的范围内大约有 2-3 个完整的波周期
    k_r = 3.0 * np.pi 
    
    # 组合正弦波：sin(k*r)
    # 添加一个相位偏移或常数项可以改变中心的相 (这里是纯波动，均值为0)
    perturbation = torch.sin(k_r * R)
    
    # 为了增加一点不对称性，防止完美的径向对称导致数值上的特殊简并，
    # 可以叠加一个微弱的角度依赖项 (可选，此处保持纯净径向以便观察)
    # perturbation += 0.1 * torch.cos(4 * torch.atan2(Y, X))

    # 3. 缩放至安全范围 [-U_MAX, U_MAX]
    # sin 的值域本身就是 [-1, 1]，直接乘系数即可
    u0 = U_MAX * perturbation

    # 确保严格在范围内 (防御性编程)
    u0 = torch.clamp(u0, -U_MAX, U_MAX)

    return u0
# initial_condition.py
# import numpy as np
# import torch

# # ==================== 全局参数 ====================
# DOMAIN_SIZE = 2.0  # 域的宽度/高度 (-1 到 1)
# U_MAX = 1.0        # 稍微大一点的振幅，测试非线性更强的区域，但仍 < 1

# def compute_initial_condition(x, y):
#     """
#     计算基于对角线方向的正弦/余弦组合初始条件。
#     形态：对角线方向的条纹或交叉网格图案。
#     适用场景：测试多方向界面竞争和各向同性演化。
    
#     Args:
#         x, y: 坐标网格 (np.ndarray 或 torch.Tensor)
        
#     Returns:
#         torch.Tensor: 初始相场变量 u0
#     """
#     # 统一输入为 torch.Tensor
#     if isinstance(x, np.ndarray):
#         X = torch.tensor(x, dtype=torch.float32)
#         Y = torch.tensor(y, dtype=torch.float32)
#     else:
#         X, Y = x, y

#     # 1. 定义波数
#     # 域宽为 2，若要产生约 3-4 个条纹，频率需适当
#     k = 2.5 * np.pi 

#     # 2. 构造对角线波动
#     # 方向 1: 沿 y = -x 方向变化 (依赖 x + y)
#     wave_1 = torch.sin(k * (X + Y))
    
#     # 方向 2: 沿 y = x 方向变化 (依赖 x - y)
#     wave_2 = torch.cos(k * (X - Y))
    
#     # 3. 线性组合
#     # 权重可以调整以改变图案的主导方向
#     # 这里采用等权重叠加，形成类似 "X" 形的交叉纹理
#     perturbation = 0.5 * wave_1 + 0.5 * wave_2
    
#     # 理论最大振幅约为 0.5*1 + 0.5*1 = 1.0 (当 sin 和 cos 同时取 1 时虽不可能同时发生，但接近)
#     # 实际最大值略小于 1，为了安全，我们重新归一化或直接缩放
    
#     # 估算最大可能值 (粗略): 当 sin=1, cos=1 时和为 1。
#     # 实际上 sin(A)+cos(B) 的最大值是 sqrt(2) 如果 A=B，但这里参数不同。
#     # 保守起见，除以 1.0 并乘以 U_MAX 即可，或者显式 clamp
    
#     u0 = U_MAX * perturbation
    
#     # 强制裁剪以确保绝对安全，防止对数势报错
#     u0 = torch.clamp(u0, -U_MAX, U_MAX)

#     return u0