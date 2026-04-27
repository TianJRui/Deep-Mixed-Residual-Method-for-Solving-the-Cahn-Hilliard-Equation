
# initial_condition.py
import numpy as np
import torch

# # ==================== 全局参数 ====================
# DOMAIN_SIZE = 2.0  # 域的宽度/高度 (-1 到 1)
# U_MAX = 0.75        # 最大绝对值，确保远离对数势奇点 ±1

# def compute_initial_condition(x, y):
#     """
#     计算基于径向距离的同心圆环状初始条件。
#     形态：以原点为中心的同心圆波纹。
#     适用场景：测试曲线界面的演化和曲率效应。
    
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

#     # 1. 计算径向距离 r (从中心 0,0 开始)
#     # 域是 [-1, 1]，所以 r 的范围大约是 [0, sqrt(2)]
#     R = torch.sqrt(X**2 + Y**2)
    
#     # 2. 构造径向波动
#     # 频率参数：控制圆环的密度。
#     # 这里使用 3*pi 使得在半径 ~1.4 的范围内大约有 2-3 个完整的波周期
#     k_r = 3.0 * np.pi 
    
#     # 组合正弦波：sin(k*r)
#     # 添加一个相位偏移或常数项可以改变中心的相 (这里是纯波动，均值为0)
#     perturbation = torch.sin(k_r * R)
    
#     # 为了增加一点不对称性，防止完美的径向对称导致数值上的特殊简并，
#     # 可以叠加一个微弱的角度依赖项 (可选，此处保持纯净径向以便观察)
#     # perturbation += 0.1 * torch.cos(4 * torch.atan2(Y, X))

#     # 3. 缩放至安全范围 [-U_MAX, U_MAX]
#     # sin 的值域本身就是 [-1, 1]，直接乘系数即可
#     u0 = U_MAX * perturbation

#     # 确保严格在范围内 (防御性编程)
#     u0 = torch.clamp(u0, -U_MAX, U_MAX)

#     return u0
# initial_condition.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

# ==================== 全局参数 ====================
DOMAIN_SIZE = 2.0  # 域的宽度/高度 (-1 到 1)
U_MAX = 1.0        # 稍微大一点的振幅，测试非线性更强的区域，但仍 < 1

def compute_initial_condition(x, y):
    """
    计算基于对角线方向的正弦/余弦组合初始条件。
    形态：对角线方向的条纹或交叉网格图案。
    适用场景：测试多方向界面竞争和各向同性演化。
    
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

    # 1. 定义波数
    # 域宽为 2，若要产生约 3-4 个条纹，频率需适当
    k = 1.0 * np.pi 

    # 2. 构造对角线波动
    # 方向 1: 沿 y = -x 方向变化 (依赖 x + y)
    wave_1 = torch.sin(k * (X + Y))
    
    # 方向 2: 沿 y = x 方向变化 (依赖 x - y)
    wave_2 = torch.cos(k * (X - Y))
    
    # 3. 线性组合
    # 权重可以调整以改变图案的主导方向
    # 这里采用等权重叠加，形成类似 "X" 形的交叉纹理
    perturbation = 1.0 * wave_1 + 1.0 * wave_2
    
    u0 = U_MAX * perturbation
    
    # 强制裁剪以确保绝对安全，防止对数势报错
    u0 = torch.clamp(u0, -U_MAX, U_MAX)

    return u0

# def plot_initial_condition(resolution=100, save_path=None):
#     """
#     绘制初始条件在 [-1, 1]×[-1, 1] 域上的分布
    
#     Args:
#         resolution: 网格分辨率 (resolution x resolution)
#         save_path: 图片保存路径 (None 表示不保存，直接显示)
#     """
#     # 1. 创建坐标网格
#     x = np.linspace(-1, 1, resolution)
#     y = np.linspace(-1, 1, resolution)
#     X, Y = np.meshgrid(x, y)
    
#     # 2. 计算初始条件
#     u0 = compute_initial_condition(X, Y)
#     # 转换为numpy数组用于绘图
#     u0_np = u0.numpy()
    
#     # 3. 创建绘图
#     fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
#     # 绘制热力图
#     im = ax.imshow(u0_np, 
#                    extent=[-1, 1, -1, 1],  # 坐标范围
#                    origin='lower',         # 原点在左下角
#                    cmap=cm.coolwarm,       # 冷暖色配色
#                    vmin=-U_MAX,            # 颜色范围最小值
#                    vmax=U_MAX)             # 颜色范围最大值
    
#     # 添加等高线，增强图案辨识度
#     contour = ax.contour(X, Y, u0_np, 
#                          levels=15,        # 等高线数量
#                          colors='k',       # 黑色等高线
#                          linewidths=0.5)   # 线宽
    
#     # 4. 设置图表样式
#     ax.set_xlabel('x', fontsize=12)
#     ax.set_ylabel('y', fontsize=12)
#     ax.set_title('Initial Condition: Diagonal Wave Pattern', fontsize=14, pad=15)
#     ax.set_aspect('equal')  # 等比例显示
    
#     # 添加颜色条
#     cbar = fig.colorbar(im, ax=ax, shrink=0.85)
#     cbar.set_label('u(x, y)', fontsize=12)
    
#     # 优化布局
#     plt.tight_layout()
    
#     # 保存或显示图片
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"图像已保存至: {save_path}")
#     else:
#         plt.show()
    
#     plt.close()
    
#     return fig, ax

# # ==================== 测试绘图 ====================
# if __name__ == "__main__":
#     import os
#     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#     # 绘制初始条件，分辨率100x100
#     plot_initial_condition(resolution=200)
    
#     # 如果需要保存图片，可以使用：
# #     # plot_initial_condition(resolution=200, save_path='initial_condition.png')