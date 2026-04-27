# losses.py
"""
CH方程的损失函数（多项式非线性项版本）。
"""

import torch
import numpy as np
from scipy.special import roots_legendre
from config import gauss_n
def _generate_mass_conservation_inputs(t_f, gauss_n=20, device='cpu'):
    """
    生成质量守恒损失所需的输入：
    1. 固定的二维高斯积分点 (gx, gy) 和权重 (gw)。
    2. 均匀时间间隔对应的当前时刻 t 和初始时刻 t=0 的张量。
    
    Args:
        t_f (torch.Tensor): 当前PDE批次的時間输入，形状 (N, 1)。用于提取当前时刻 t。
        gauss_n (int): 高斯积分点的数量（每维）。
        device: 计算设备。
        
    Returns:
        gx, gy (torch.Tensor): 形状 (M, 1) 的二维高斯点坐标 (M = gauss_n^2)。
        gw (torch.Tensor): 形状 (M,) 的高斯权重。
        t_current (torch.Tensor): 形状 (M, 1)，广播后的当前时刻。
        t_initial (torch.Tensor): 形状 (M, 1)，全零的初始时刻。
    """
    # 1. 生成一维高斯 - 勒让德节点和权重 (区间 [-1, 1])
    x_leg, w_leg = roots_legendre(gauss_n)
    
    # 转换为 Tensor
    x_leg = torch.tensor(x_leg, dtype=torch.float32, device=device)
    w_leg = torch.tensor(w_leg, dtype=torch.float32, device=device)
    
    # 2. 构建二维网格 (Meshgrid)
    # gx, gy 形状为 (gauss_n, gauss_n)
    gx, gy = torch.meshgrid(x_leg, x_leg, indexing='ij')
    gx = gx.flatten().unsqueeze(1) # 形状 (M, 1)
    gy = gy.flatten().unsqueeze(1) # 形状 (M, 1)
    
    # 3. 计算二维权重 (外积)
    # gw 形状为 (gauss_n, gauss_n) -> 展平为 (M,)
    gw = torch.outer(w_leg, w_leg).flatten()
    
    # 4. 构造时间输入
    # 当前时刻：取批次中第一个时间步作为代表 (假设批次内时间相近或取平均值)
    # 如果 t_f 是单个值或需要特定处理，可在此调整
    t_val = t_f[0:1] if t_f.dim() > 0 else t_f 
    t_current = t_val.expand(gx.shape[0], 1) # 广播到 M 个点
    
    # 初始时刻：全零
    t_initial = torch.zeros_like(t_current)
    
    return gx, gy, gw, t_current, t_initial

def compute_boundary_loss_heat(u_b, x_b, y_b, create_graph=True):
    """
    计算齐次 Neumann 边界条件损失 (∂u/∂n = 0)。
    在单位正方形 [-1,1]^2 上，法向导数在各边上简化为坐标方向偏导。
    """
    u_b_x = torch.autograd.grad(u_b, x_b, grad_outputs=torch.ones_like(u_b), create_graph=create_graph)[0]
    u_b_y = torch.autograd.grad(u_b, y_b, grad_outputs=torch.ones_like(u_b), create_graph=create_graph)[0]
    
    eps = 1e-6
    mask_right = (x_b >= 1 - eps)
    mask_left = (x_b <= -1 + eps)
    mask_top = (y_b >= 1 - eps)
    mask_bottom = (y_b <= -1 + eps)
    
    loss = 0.0
    count = 0
    if mask_bottom.any():
        loss += torch.mean(u_b_y[mask_bottom] ** 2)
        count += 1
    if mask_top.any():
        loss += torch.mean(u_b_y[mask_top] ** 2)
        count += 1
    if mask_left.any():
        loss += torch.mean(u_b_x[mask_left] ** 2)
        count += 1
    if mask_right.any():
        loss += torch.mean(u_b_x[mask_right] ** 2)
        count += 1
    result = loss / max(count, 1)
    if not isinstance(result, torch.Tensor):
        result = torch.tensor(result)
    return result

def logarithmic_potential_derivative(u, theta=0.8, theta_c=1.0, epsilon=1e-6):
    """
    计算对数双井势的导数 f'(u)。
    
    势函数形式:
    F(u) = (theta/2) * [(1+u)ln(1+u) + (1-u)ln(1-u)] + (theta_c/2) * (1 - u^2)
    
    导数形式:
    f'(u) = (theta/2) * ln((1+u)/(1-u)) - theta_c * u
    
    Args:
        u (torch.Tensor): 网络输出，相场变量。理论上应在 (-1, 1) 之间。
        theta (float): 无量纲温度参数。theta < theta_c 时发生相分离。
                       默认 0.8 (假设 theta_c=1)。
        theta_c (float): 临界温度参数，通常设为 1.0。
        epsilon (float): 数值稳定性参数，用于裁剪 u 的范围，防止 ln(0)。
        
    Returns:
        torch.Tensor: f'(u)
    """
    # 1. 数值稳定性处理：将 u 裁剪到 (-1 + epsilon, 1 - epsilon)
    # 防止出现 ln(0) 或除以零的情况
    u_clamped = torch.clamp(u, -1.0 + epsilon, 1.0 - epsilon)
    
    # 2. 计算对数项: ln((1+u)/(1-u))
    # 使用 torch.log 和基本的算术运算
    log_term = torch.log((1.0 + u_clamped) / (1.0 - u_clamped))
    
    # 3. 组合最终结果: (theta/2) * log_term - theta_c * u
    # 注意：这里减去的是 theta_c * u (原始未裁剪的 u 或裁剪后的 u 均可，通常影响微小)
    return (theta / 2.0) * log_term - theta_c * u

def compute_ch_pde_residual(
    net_u, x, y, t,
    gamma=0.0016,       # γ = ε_CH²
    mobility_type="degenerate"
):
    """计算含多项式非线性项的 CH 方程的 PDE 残差: u_t - ∇·(M(u)∇μ) = 0"""
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    phi = net_u(x, y, t)
    u = torch.tanh(phi)
    # 时间导数
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # 空间梯度
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # Laplacian
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    lap_u = u_xx + u_yy

    f_prime = logarithmic_potential_derivative(u)
    mu = -gamma * lap_u + f_prime

    # ∇μ
    mu_x = torch.autograd.grad(mu, x, grad_outputs=torch.ones_like(mu), create_graph=True)[0]
    mu_y = torch.autograd.grad(mu, y, grad_outputs=torch.ones_like(mu), create_graph=True)[0]

    # 迁移率 M(u)
    if mobility_type == "degenerate":
        # 退化迁移率 M(u) = 1 - u^2，保证在纯相区通量为0
        M = 1.0 - u**2
        # 可选：防止 M 变为负数（虽然理论上 u 应在 [-1,1] 之间，但数值误差可能导致越界）
        # M = torch.clamp(M, min=0.0) 
    else:
        M = torch.ones_like(u)

    # 通量 J = M ∇μ
    flux_x = M * mu_x
    flux_y = M * mu_y

    # 散度 ∇·J
    div_flux_x = torch.autograd.grad(flux_x, x, grad_outputs=torch.ones_like(flux_x), create_graph=True)[0]
    div_flux_y = torch.autograd.grad(flux_y, y, grad_outputs=torch.ones_like(flux_y), create_graph=True)[0]
    div_flux = div_flux_x + div_flux_y

    residual = u_t - div_flux
    return residual


def compute_standard_loss_CH(
    net_u, x_f, y_f, t_f,
    x_b, y_b, t_b,
    x_i, y_i, t_i, ic,
    lambda_bc=1.0, lambda_ic=1.0,lambda_mass=1e-6,
    gamma=0.016,
    mobility_type="degenerate",
):
    """
    含多项式非线性项的 Cahn-Hilliard 方程损失函数。
    移除了 theta, theta_c, eps_reg 等对数势相关参数。
    """
    # --- PDE Loss ---
    pde_residual = compute_ch_pde_residual(
        net_u, x_f, y_f, t_f,
        gamma=gamma,
        mobility_type=mobility_type
    )
    loss_pde = torch.mean(pde_residual ** 2)

    # --- Neumann BC: ∂u/∂n = 0 ---
    x_b.requires_grad_(True)
    y_b.requires_grad_(True)
    
    u_raw_b = net_u(x_b, y_b, t_b)
    
    u_b = torch.tanh(u_raw_b)
    
    u_b_x = torch.autograd.grad(u_b, x_b, grad_outputs=torch.ones_like(u_b), create_graph=True)[0]
    u_b_y = torch.autograd.grad(u_b, y_b, grad_outputs=torch.ones_like(u_b), create_graph=True)[0]

    eps = 1e-8  # 边界检测容差
    mask_right = (x_b >= 1 - eps)
    mask_left = (x_b <= -1 + eps)
    mask_top = (y_b >= 1 - eps)
    mask_bottom = (y_b <= -1 + eps)

    loss_bc = 0.0
    count = 0
    if mask_bottom.any():
        loss_bc += torch.mean(u_b_y[mask_bottom] ** 2)
        count += 1
    if mask_top.any():
        loss_bc += torch.mean(u_b_y[mask_top] ** 2)
        count += 1
    if mask_left.any():
        loss_bc += torch.mean(u_b_x[mask_left] ** 2)
        count += 1
    if mask_right.any():
        loss_bc += torch.mean(u_b_x[mask_right] ** 2)
        count += 1

    loss_bc = (loss_bc / max(count, 1)) * lambda_bc

    # --- Initial Condition ---
    u_raw_i = net_u(x_i, y_i, t_i)
    u_i = torch.tanh(u_raw_i)
    loss_ic = torch.mean((u_i - ic) ** 2) * lambda_ic

    # --- Mass Conservation Loss (高斯积分版) ---
    # 1. 生成固定的高斯积分点和时间输入
    device = t_f.device
    gx, gy, gw, t_current, t_initial = _generate_mass_conservation_inputs(
        t_f, gauss_n=gauss_n, device=device
    )
    
    # 2. 计算当前时刻 t 和初始时刻 t=0 的网络输出
    # 注意：网络输出通常需要 tanh 激活，视具体模型定义而定，此处假设 net_u 输出 raw，需 tanh
    u_raw_curr = net_u(gx, gy, t_current)
    u_curr = torch.tanh(u_raw_curr)
    
    u_raw_init = net_u(gx, gy, t_initial)
    u_init = torch.tanh(u_raw_init)
    
    # 3. 高斯积分计算总质量
    # Integral ≈ sum(w_i * u_i)
    # gw 形状 (M,), u 形状 (M, 1) -> 需 squeeze
    mass_current = torch.sum(gw * u_curr.squeeze())
    mass_initial = torch.sum(gw * u_init.squeeze())
    
    # 4. 计算损失
    loss_mass = (mass_current - mass_initial) ** 2 * lambda_mass
    
    
    total_loss = loss_pde + loss_bc + loss_ic + loss_mass
    return total_loss, loss_pde, loss_bc, loss_ic


def compute_drm_loss_CH(
    net_u, net_mu,
    x_f, y_f, t_f,
    x_b, y_b, t_b,
    x_i, y_i, t_i, ic,
    lambda_bc=1.0,
    lambda_ic=1.0,
    lambda_eq1=1.0,   # 权重 for equation (A): mu + gamma*lap_u - f'(u) = 0
    lambda_eq2=1.0,   # 权重 for equation (B): u_t - div(M(u) grad mu) = 0
    lambda_mass=1e-2,
    gamma=0.016,
    mobility_type="degenerate"
):
    """
    双网络深度混合残量法 (DRM) 损失函数 for Cahn-Hilliard 方程 。
    """
    # ========================
    # 1. 在内部点计算 PDE 残差
    # ========================
    x_f.requires_grad_(True)
    y_f.requires_grad_(True)
    t_f.requires_grad_(True)

    u_raw_f = net_u(x_f, y_f, t_f)
    u = torch.tanh(u_raw_f)
    mu_pred = net_mu(x_f, y_f, t_f)

    # ---- 计算 Equation (A): mu + gamma * Δu - f'(u) = 0 ----
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_f, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    lap_u = u_xx + u_yy

    # 【修改点】使用多项式势导数
    f_prime = logarithmic_potential_derivative(u)
    
    # 注意：原公式是 mu = -gamma*lap_u + f'(u) => mu + gamma*lap_u - f'(u) = 0
    residual_eq1 = mu_pred + gamma * lap_u - f_prime
    loss_eq1 = torch.mean(residual_eq1 ** 2) * lambda_eq1

    # ---- 计算 Equation (B): u_t - ∇·(M(u) ∇mu) = 0 ----
    u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    mu_x = torch.autograd.grad(mu_pred, x_f, grad_outputs=torch.ones_like(mu_pred), create_graph=True)[0]
    mu_y = torch.autograd.grad(mu_pred, y_f, grad_outputs=torch.ones_like(mu_pred), create_graph=True)[0]

    if mobility_type == "degenerate":
        M = 1.0 - u**2
    else:
        M = torch.ones_like(u)

    flux_x = M * mu_x
    flux_y = M * mu_y

    div_flux_x = torch.autograd.grad(flux_x, x_f, grad_outputs=torch.ones_like(flux_x), create_graph=True)[0]
    div_flux_y = torch.autograd.grad(flux_y, y_f, grad_outputs=torch.ones_like(flux_y), create_graph=True)[0]
    div_flux = div_flux_x + div_flux_y

    residual_eq2 = u_t - div_flux
    loss_eq2 = torch.mean(residual_eq2 ** 2) * lambda_eq2

    # ========================
    # 2. 边界条件: ∂u/∂n = 0, ∂mu/∂n = 0 （齐次 Neumann）
    # ========================
    x_b.requires_grad_(True)
    y_b.requires_grad_(True)
    u_raw_b = net_u(x_b, y_b, t_b)
    u_b = torch.tanh(u_raw_b)
    mu_b = net_mu(x_b, y_b, t_b)

    u_b_x = torch.autograd.grad(u_b, x_b, grad_outputs=torch.ones_like(u_b), create_graph=True)[0]
    u_b_y = torch.autograd.grad(u_b, y_b, grad_outputs=torch.ones_like(u_b), create_graph=True)[0]
    mu_b_x = torch.autograd.grad(mu_b, x_b, grad_outputs=torch.ones_like(mu_b), create_graph=True)[0]
    mu_b_y = torch.autograd.grad(mu_b, y_b, grad_outputs=torch.ones_like(mu_b), create_graph=True)[0]

    eps = 1e-8
    mask_right = (x_b >= 1 - eps)
    mask_left = (x_b <= -1 + eps)
    mask_top = (y_b >= 1 - eps)
    mask_bottom = (y_b <= -1 + eps)

    loss_bc_u = 0.0
    loss_bc_mu = 0.0
    count = 0

    for mask in [mask_bottom, mask_top, mask_left, mask_right]:
        if mask.any():
            # ∂u/∂n = 0
            if mask is mask_bottom or mask is mask_top:
                loss_bc_u += torch.mean(u_b_y[mask] ** 2)
            else:  # left or right
                loss_bc_u += torch.mean(u_b_x[mask] ** 2)
            
            # ∂mu/∂n = 0
            if mask is mask_bottom or mask is mask_top:
                loss_bc_mu += torch.mean(mu_b_y[mask] ** 2)
            else:
                loss_bc_mu += torch.mean(mu_b_x[mask] ** 2)
            
            count += 1

    loss_bc = (loss_bc_u + loss_bc_mu) / max(count, 1) * lambda_bc

    # ========================
    # 3. 初始条件: u(x, y, 0) = ic(x, y)
    # ========================
    u_raw_i = net_u(x_i, y_i, t_i)
    u_i = torch.tanh(u_raw_i)
    loss_ic = torch.mean((u_i - ic) ** 2) * lambda_ic

    
    # --- Mass Conservation Loss (高斯积分版) ---
    # 1. 生成固定的高斯积分点和时间输入
    device = t_f.device
    gx, gy, gw, t_current, t_initial = _generate_mass_conservation_inputs(
        t_f, gauss_n=gauss_n, device=device
    )
    
    # 2. 计算当前时刻 t 和初始时刻 t=0 的网络输出
    # 注意：网络输出通常需要 tanh 激活，视具体模型定义而定，此处假设 net_u 输出 raw，需 tanh
    u_raw_curr = net_u(gx, gy, t_current)
    u_curr = torch.tanh(u_raw_curr)
    
    u_raw_init = net_u(gx, gy, t_initial)
    u_init = torch.tanh(u_raw_init)
    
    # 3. 高斯积分计算总质量
    # Integral ≈ sum(w_i * u_i)
    # gw 形状 (M,), u 形状 (M, 1) -> 需 squeeze
    mass_current = torch.sum(gw * u_curr.squeeze())
    mass_initial = torch.sum(gw * u_init.squeeze())
    
    # 4. 计算损失
    loss_mass = (mass_current - mass_initial) ** 2 * lambda_mass
    
    # ========================
    # 4. 总损失
    # ========================
    total_loss = loss_eq1 + loss_eq2 + loss_bc + loss_ic + loss_mass

    return total_loss, loss_eq1, loss_eq2, loss_bc, loss_ic