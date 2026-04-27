import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from config import DEVICE, DOMAIN_SIZE
from initial_condition import compute_initial_condition
from typing import Optional, Union, List
import torch.nn as nn
# --- 新增导入 ---
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
                               x_range=(-DOMAIN_SIZE/2, DOMAIN_SIZE/2), 
                               y_range=(-DOMAIN_SIZE/2, DOMAIN_SIZE/2), 
                               filename_prefix="solution_2d_snapshot",
                               use_same_colorbar=False):
    """
    绘制并保存 2D 解在指定时间点的快照。
    
    Parameters:
    -----------
    model : torch.nn.Module
        训练好的 PINN 模型。
    times : list
        要绘制的时间点列表。
    nx, ny : int
        网格分辨率。
    x_range, y_range : tuple
        绘图区域的 x 和 y 范围。
    filename_prefix : str
        保存文件的前缀名。
    use_same_colorbar : bool
        如果为 True，所有子图共享同一个颜色条范围 (vmin/vmax)，且只显示一个颜色条。
        如果为 False，每个子图根据自己的数据范围独立显示颜色条。
    """
    model.eval()
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # 生成网格
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    # 预计算所有时间点的预测值，以便确定全局范围 (如果需要)
    u_predictions = []
    t_values_clean = []
    
    # 准备输入张量 (一次性构建所有需要的输入以加速，或者循环处理)
    # 这里为了内存安全，还是循环获取预测值
    for t_val in times:
        if isinstance(t_val, torch.Tensor):
            t_val_float = t_val.item()
        else:
            t_val_float = float(t_val)
        
        t_values_clean.append(t_val_float)
        
        T_flat = np.full_like(X_flat, t_val_float)
        input_tensor = np.stack([X_flat, Y_flat, T_flat], axis=1)
        input_tensor_torch = torch.tensor(input_tensor, dtype=torch.float32).to(DEVICE) # 确保使用正确的 DEVICE
        
        with torch.no_grad():
            # 拆分输入以适配模型 forward(x, y, t) 的签名
            # 假设模型接受 (batch, 1) 形状的张量作为单独参数
            x_in = input_tensor_torch[:, 0:1]
            y_in = input_tensor_torch[:, 1:2]
            t_in = input_tensor_torch[:, 2:3]
            u_pred = model(x_in, y_in, t_in)
            u_pred = torch.tanh(u_pred).cpu().numpy().flatten()
            u_predictions.append(u_pred.reshape(nx, ny))

    # 创建画布
    fig, axes = plt.subplots(1, len(times), figsize=(5 * len(times), 4), constrained_layout=True)
    if len(times) == 1:
        axes = [axes]

    # 如果需要统一颜色条，先计算全局 min/max
    vmin_global, vmax_global = None, None
    if use_same_colorbar:
        all_data = np.concatenate([u.flatten() for u in u_predictions])
        vmin_global = float(np.min(all_data))
        vmax_global = float(np.max(all_data))
        # 可选：稍微扩展一点范围让颜色更好看
        # span = vmax_global - vmin_global
        # vmin_global -= 0.05 * span
        # vmax_global += 0.05 * span

    # 开始绘图
    im = None # 用于最后创建 colorbar
    for idx, t_val in enumerate(t_values_clean):
        U = u_predictions[idx]
        
        if use_same_colorbar:
            im = axes[idx].pcolormesh(X, Y, U, shading='auto', cmap='viridis', 
                                      vmin=vmin_global, vmax=vmax_global)
        else:
            im = axes[idx].pcolormesh(X, Y, U, shading='auto', cmap='viridis')
        
        axes[idx].set_title(f't = {t_val:.2f}')
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('y')
        axes[idx].set_aspect('equal') # 保持纵横比一致

        # 只有在不统一颜色条，或者是统一颜色条的最后一个图时，才添加 colorbar
        if not use_same_colorbar:
            fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        elif use_same_colorbar and idx == len(times) - 1:
            # 统一颜色条时，只在最后一个轴上画，并调整位置
            cbar = fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            cbar.set_label('u(x,y,t)')

    plt.savefig(f"{filename_prefix}.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {filename_prefix}.png (Colorbar mode: {'Unified' if use_same_colorbar else 'Independent'})")
# --- 辅助函数：将 [0,1] 坐标映射到 [-1,1] ---
def map_to_original_domain(x, y):
    """将 (0,1)^2 上的点映射到 (-1,1)^2"""
    return 2*x - 1, 2*y - 1

def gauss_legendre_quadrature_2d(f, n=10):
    x_legendre, w_legendre = roots_legendre(n)
    x_scaled = (x_legendre + 1) / 2
    w_scaled = w_legendre / 2
    x_mesh, y_mesh = np.meshgrid(x_scaled, x_scaled, indexing='ij')
    w_mesh = np.outer(w_scaled, w_scaled)
    return np.sum(w_mesh * f(x_mesh, y_mesh))

def compute_total_mass(net, t_val, gauss_n=20):
    """
    计算特定时间 t 的总质量: M_total = M_bulk + M_boundary
    """
    # 1. 获取高斯点和权重 (映射到 [0, 1])
    x_leg, w_leg = roots_legendre(gauss_n)
    x_gp = (x_leg + 1) / 2.0
    w_gp = w_leg / 2.0
    
    # 2. 计算体相质量 (使用 2D 高斯积分)
    X, Y = np.meshgrid(x_gp, x_gp, indexing='ij')
    w_mesh = np.outer(w_gp, w_gp)
    
    x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=DEVICE)
    y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32, device=DEVICE)
    t_tensor = torch.full_like(x_flat[:, 0:1], t_val, dtype=torch.float32)
    
    with torch.no_grad():
        u_bulk = torch.tanh(net(x_flat, y_flat, t_tensor)).cpu().numpy().flatten()
        u_bulk_2d = u_bulk.reshape(gauss_n, gauss_n)
        mass_bulk = np.sum(w_mesh * u_bulk_2d)

    # 3. 计算边界质量 (使用 1D 高斯积分，四条边)
    # 假设边界范围为 [-1, 1] 或者 [0, 1]，这里假设映射后的域是 [0,1] 对应原始域 [-1, 1]
    # 如果是 [-1, 1]，权重需要 * 1 (不除以2)
    
    # 辅助函数计算单边积分
    def integrate_edge(coord_fixed, var_axis, val):
        # coord_fixed: 'x' or 'y'
        # var_axis: 变化的坐标张量 (gauss_n, 1)
        # val: 固定的坐标值
        
        if coord_fixed == 'x':
            x_tens = torch.full_like(var_axis, val)
            y_tens = var_axis
        else:
            x_tens = var_axis
            y_tens = torch.full_like(var_axis, val)
            
        t_tens = torch.full_like(var_axis, t_val)
        with torch.no_grad():
            u_edge = torch.tanh(net(x_tens, y_tens, t_tens)).cpu().numpy().flatten()
        # 积分: sum(w * u) * scale_factor (假设域是 [0,1]，scale=1; 如果域是 [-1,1]，scale=2)
        # 这里假设物理域是 [-1, 1]，所以长度是 2
        return np.sum(w_gp * u_edge) * 2.0

    # 四条边积分 (假设物理域是 [-1, 1])
    # 左 x=-1, 右 x=1, 下 y=-1, 上 y=1
    # 需要将 gauss 点映射回 [-1, 1]
    x_leg_tensor = torch.tensor(x_leg[:, None], dtype=torch.float32, device=DEVICE)
    
    mass_left   = integrate_edge('x', x_leg_tensor, -1.0)
    mass_right  = integrate_edge('x', x_leg_tensor,  1.0)
    mass_bottom = integrate_edge('y', x_leg_tensor, -1.0)
    mass_top    = integrate_edge('y', x_leg_tensor,  1.0)
    
    mass_boundary = mass_left + mass_right + mass_bottom + mass_top
    
    return mass_bulk + mass_boundary

import torch
import numpy as np
from typing import Optional, Union, List
import torch.nn as nn

def compute_adaptive_eps_reg(
    net_u: nn.Module, 
    net_mu: Optional[nn.Module] = None, 
    epoch: int = 0, 
    x_sample: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None, 
    y_sample: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None, 
    t_sample: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None, 
) -> float:
    """
    根据当前神经网络输出的绝对值最大值，自适应计算正则化参数 eps_reg。
    
    原理：
    对数势导数 f'(u) 包含 ln(1+u+eps) 和 ln(1-u+eps)。
    当 u 接近 1 或 -1 时，需要较大的 eps 来避免数值奇异性。
    核心逻辑：
    1. 首要计算 |u|的最大值 - 1（反映u超出[-1,1]物理范围的程度）
    2. 叠加指数衰减项作为基础兜底
    3. 结合mu的异常值辅助调整（可选）
    
    Args:
        net_u (nn.Module): 预测 u 的网络。
        net_mu (nn.Module, optional): 预测 mu 的网络 (用于 DRM 方法)。
        epoch (int): 当前训练轮次，用于指数衰减策略。
        x_sample (torch.Tensor | List[torch.Tensor], optional): x 坐标采样点。
        y_sample (torch.Tensor | List[torch.Tensor], optional): y 坐标采样点。
        t_sample (torch.Tensor | List[torch.Tensor], optional): t 坐标采样点。
        
    Returns:
        float: 计算得到的自适应 eps_reg 值。
    """
    # 延迟导入以避免循环依赖，同时获取配置常量
    from config import EPS_REG_INITIAL, EPS_REG_DECAY_RATE, EPS_REG_MIN, DEVICE

    # 1. 计算基础指数衰减项（兜底项）
    decay_eps = max(EPS_REG_MIN, EPS_REG_INITIAL * np.exp(-EPS_REG_DECAY_RATE * epoch))
    
    # 2. 若未提供采样点，直接返回指数衰减项
    if x_sample is None or y_sample is None or t_sample is None:
        return decay_eps
    
    # 3. 处理采样点输入（列表转Tensor + 设备/类型对齐）
    if isinstance(x_sample, list):
        x_sample = torch.cat([t.float() for t in x_sample], dim=0).to(DEVICE)
        y_sample = torch.cat([t.float() for t in y_sample], dim=0).to(DEVICE)
        t_sample = torch.cat([t.float() for t in t_sample], dim=0).to(DEVICE)
    elif isinstance(x_sample, torch.Tensor) and isinstance(y_sample, torch.Tensor) and isinstance(t_sample, torch.Tensor):
        x_sample = x_sample.to(DEVICE, dtype=torch.float)
        y_sample = y_sample.to(DEVICE, dtype=torch.float)
        t_sample = t_sample.to(DEVICE, dtype=torch.float)
    else:
        raise ValueError("x/y/t_sample 必须是 torch.Tensor 或 List[torch.Tensor]")

    # 4. 无梯度计算网络输出的极值
    with torch.no_grad():
        
        u = net_u(x_sample, y_sample, t_sample)
        u_pred = torch.tanh(u)
        max_abs_u = torch.max(torch.abs(u_pred)).item()  # |u|的最大值
        
        # 核心动态项：|u|最大值 - 1（反映u超出[-1,1]的程度）
        dynamic_u = max(0.0, max_abs_u - 1.0)
        
        # 可选：mu异常值辅助调整（仅当mu过大时增加少量正则化）
        dynamic_mu = 0.0
        if net_mu is not None:
            mu_pred = net_mu(x_sample, y_sample, t_sample)
            max_abs_mu = torch.max(torch.abs(mu_pred)).item()
            # mu正常量级假设为10，超过则线性增加辅助项（可根据实际调整）
            dynamic_mu = max(0.0, (max_abs_mu - 10.0) / 100.0)

    # 5. 综合计算eps_reg：基础衰减项 + u动态项 + mu辅助项
    eps_reg = decay_eps + dynamic_u + dynamic_mu
    
    # 6. 限制范围：不低于最小值（防止正则化过弱）
    eps_reg = max(EPS_REG_MIN, eps_reg)
    
    return eps_reg

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from config import DEVICE, DOMAIN_SIZE
from losses import compute_ch_pde_residual

def plot_residual_landscape_epoch(
    model, 
    epoch, 
    t_val=0.0, 
    nx=100, 
    ny=100, 
    results_dir="results_2d_heat",
    log_scale=True
):
    """
    绘制并保存指定 epoch 和时间点下的 PDE 残差景观图像 (2D Contourf)。
    
    参数:
        model (nn.Module): 训练好的神经网络模型 (net_u)。
        epoch (int): 当前的训练轮次，用于文件名标记。
        t_val (float): 想要可视化的时间点 (默认 0.0)。
        nx, ny (int): 网格分辨率。
        results_dir (str): 保存图像的目录。
        log_scale (bool): 是否使用对数刻度显示残差绝对值 (推荐 True，因为残差通常跨越多个数量级)。
    """
    model.eval()
    
    # 1. 生成网格点
    half_domain = DOMAIN_SIZE / 2.0
    x = np.linspace(-half_domain, half_domain, nx)
    y = np.linspace(-half_domain, half_domain, ny)
    X, Y = np.meshgrid(x, y)
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    T_flat = np.full_like(X_flat, t_val)
    
    # 转换为 Tensor 并移动到设备
    x_tensor = torch.tensor(X_flat[:, None], dtype=torch.float32, device=DEVICE)
    y_tensor = torch.tensor(Y_flat[:, None], dtype=torch.float32, device=DEVICE)
    t_tensor = torch.tensor(T_flat[:, None], dtype=torch.float32, device=DEVICE)
    
    # 2. 计算残差
    # 需要设置 requires_grad=True 以便 compute_ch_pde_residual 计算高阶导数
    x_tensor.requires_grad_(True)
    y_tensor.requires_grad_(True)
    t_tensor.requires_grad_(True)
    
    # 调用 losses.py 中定义的残差计算函数
    # 注意：这里使用了默认的 gamma, theta 等参数，如需修改需在此处传入
    residual = compute_ch_pde_residual(model, x_tensor, y_tensor, t_tensor)
        
    # 获取残差的绝对值并移回 CPU
    res_abs = torch.abs(residual).detach().cpu().numpy().flatten()

    # 重塑为 2D 网格
    Z = res_abs.reshape(nx, ny)
    
    # 3. 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 处理可能出现的 0 值导致 log 报错的情况
    if log_scale:
        # 使用掩码避免 log(0)
        Z_plot = np.ma.masked_less_equal(Z, 0)
        # 如果最小值仍很小，可以设定一个下限以便可视化，或者直接画 log10
        im = ax.contourf(X, Y, Z_plot, levels=50, cmap='plasma', vmax=np.max(Z))
        ax.set_title(f'Residual Landscape | Epoch {epoch} | t = {t_val:.2f} (Log Scale)')
    else:
        im = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
        ax.set_title(f'Residual Landscape | Epoch {epoch} | t = {t_val:.2f}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('|Residual|' if not log_scale else 'log(|Residual|)')
    
    # 4. 保存图像
    os.makedirs(results_dir, exist_ok=True)
    scale_str = "log" if log_scale else "linear"
    filename = os.path.join(results_dir, f"residual_landscape_epoch_{epoch}_t{t_val:.2f}_{scale_str}.png")
    
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return filename