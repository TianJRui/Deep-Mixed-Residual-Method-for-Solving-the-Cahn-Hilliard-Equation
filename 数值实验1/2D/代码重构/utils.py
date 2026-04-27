
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import roots_legendre
from config import DEVICE, DOMAIN_SIZE, T_FINAL

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_2d_solution_snapshots(model, times, nx=100, ny=100, 
                               x_range=(0, 1), y_range=(0, 1),
                               filename_prefix="solution_2d_snapshot"):
    """
    绘制并保存 2D 解在指定时间点的快照。
    使用 pcolormesh 显示 u(x, y, t)
    """
    model.eval()
    x_min, x_max = x_range
    y_min, y_max = y_range

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    fig, axes = plt.subplots(1, len(times), figsize=(5 * len(times), 4), constrained_layout=True)

    if len(times) == 1:
        axes = [axes]

    for idx, t_val in enumerate(times):
        T_flat = np.full_like(X_flat, t_val)
        input_tensor = torch.tensor(np.stack([X_flat, Y_flat, T_flat], axis=1), 
                                    dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            # 从input_tensor中拆分出x, y, t分量
            x = input_tensor[:, 0:1]  # 取第一列作为x
            y = input_tensor[:, 1:2]  # 取第二列作为y
            t = input_tensor[:, 2:3]  # 取第三列作为t
            u_pred = model(x, y, t).cpu().numpy().flatten()  # 按正确参数格式传入
            U = u_pred.reshape(nx, ny)

        im = axes[idx].pcolormesh(X, Y, U, shading='auto', cmap='viridis')
        axes[idx].set_title(f't = {t_val:.2f}')
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('y')
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.savefig(f"{filename_prefix}.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {filename_prefix}.png")

def gauss_legendre_quadrature_2d(f, n=10):
    x_legendre, w_legendre = roots_legendre(n)
    x_scaled = (x_legendre + 1) / 2
    w_scaled = w_legendre / 2
    x_mesh, y_mesh = np.meshgrid(x_scaled, x_scaled, indexing='ij')
    w_mesh = np.outer(w_scaled, w_scaled)
    return np.sum(w_mesh * f(x_mesh, y_mesh))
def compute_initial_mean_2d(gauss_n=20):
    """
    使用 Gauss-Legendre 积分计算初始条件 u(x,y,0) 在 [0,1]^2 上的积分。
    """
    from scipy.special import roots_legendre
    x_legendre, w_legendre = roots_legendre(gauss_n)
    # 映射到 [0, 1]
    x_scaled = (x_legendre + 1) / 2.0
    w_scaled = w_legendre / 2.0  # 因为 dx = (b-a)/2 * dξ, a=0,b=1 → factor=1/2

    # 构建 2D 权重矩阵
    w_mesh = np.outer(w_scaled, w_scaled)  # shape (gauss_n, gauss_n)

    # 计算 u0 在 Gauss 点上的值
    X, Y = np.meshgrid(x_scaled, x_scaled, indexing='ij')
    u0_vals = 0.05 * (np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y))

    # 数值积分 ∫∫ u0 dx dy ≈ ΣΣ w_i w_j u0(x_i, y_j)
    integral = np.sum(w_mesh * u0_vals)
    return integral
def mass_history_2d(net, t_vals, u0_mean, gauss_n=20):
    """
    计算网络 net 在多个时间点 t_vals 上的质量 ∫∫ u(x,y,t) dx dy，
    并返回其与初始质量 u0_mean 的偏差。
    """
    from scipy.special import roots_legendre
    x_legendre, w_legendre = roots_legendre(gauss_n)
    x_scaled = (x_legendre + 1) / 2.0
    w_scaled = w_legendre / 2.0
    w_mesh = np.outer(w_scaled, w_scaled)  # (gauss_n, gauss_n)

    # 准备所有 Gauss 点坐标（展平用于批量推理）
    X, Y = np.meshgrid(x_scaled, x_scaled, indexing='ij')
    x_flat = X.flatten()[:, None]  # (N, 1)
    y_flat = Y.flatten()[:, None]  # (N, 1)
    x_tensor = torch.tensor(x_flat, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y_flat, dtype=torch.float32).to(DEVICE)

    masses = []
    for t in t_vals:
        t_tensor = torch.full_like(x_tensor, t, dtype=torch.float32)
        with torch.no_grad():
            u_pred = net(x_tensor, y_tensor, t_tensor).cpu().detach().numpy().flatten()
        # 重塑回 (gauss_n, gauss_n)
        u_2d = u_pred.reshape(gauss_n, gauss_n)
        # 计算当前质量
        current_mass = np.sum(w_mesh * u_2d)
        # 偏差 = 当前质量 - 初始质量（理论应为 0）
        masses.append(current_mass - u0_mean)
    return np.array(masses)