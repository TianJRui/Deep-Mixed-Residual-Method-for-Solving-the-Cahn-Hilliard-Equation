# losses.py
"""
CH方程的损失函数（对数非线性项动态边界条件版本）。
"""

import torch

def g(u,a=1.0, b=0.0):
    return a*u+b

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
    # 2. 计算对数项: ln((1+u)/(1-u))
    # 使用 torch.log 和基本的算术运算
    log_term = torch.log((1.0 + u + epsilon) / (1.0 - u - epsilon))
    
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

def compute_mass_residual(net_u, x_f, y_f, t_f, x_b, y_b, t_b, gamma=0.016):
    """
    计算总质量守恒残差: d/dt (Integral_Omega u dV + Integral_Gamma u ds)
    目标是最小化这个值的平方，使其趋近于 0。
    """
    # 确保坐标需要梯度，以便计算时间导数
    x_f = x_f.detach().requires_grad_(True)
    y_f = y_f.detach().requires_grad_(True)
    t_f = t_f.detach().requires_grad_(True)
    
    x_b = x_b.detach().requires_grad_(True)
    y_b = y_b.detach().requires_grad_(True)
    t_b = t_b.detach().requires_grad_(True)

    # 1. 计算体相质量的时间导数: d/dt (Integral_Omega u dV) -> Integral_Omega u_t dV
    # 在 PINN 中，我们通过采样点 u_t 的平均值来近似积分平均值
    u_raw_f = net_u(x_f, y_f, t_f)
    u_f = torch.tanh(u_raw_f)
    
    # 计算体相的时间偏导数 u_t
    u_t = torch.autograd.grad(u_f, t_f, grad_outputs=torch.ones_like(u_f), create_graph=True)[0]
    
    # 近似体相质量变化率 (均值代表积分密度的平均)
    # 注意：如果域大小不是1，这里理论上需要乘以域体积/点数，但在损失最小化中常数系数影响较小
    mass_rate_bulk = torch.mean(u_t**2) 

    # 2. 计算边界质量的时间导数: d/dt (Integral_Gamma u ds) -> Integral_Gamma u_t ds
    u_raw_b = net_u(x_b, y_b, t_b)
    u_b = torch.tanh(u_raw_b)
    
    # 计算边界的时间偏导数 u_t
    u_t_b = torch.autograd.grad(u_b, t_b, grad_outputs=torch.ones_like(u_b), create_graph=True)[0]
    
    # 近似边界质量变化率
    mass_rate_boundary = torch.mean(u_t_b**2)

    # 3. 总质量变化率 = 体相变化 + 边界变化
    # 物理定律要求 d(Total Mass)/dt = 0
    total_mass_rate = mass_rate_bulk + mass_rate_boundary
    
    return total_mass_rate
def compute_standard_loss_CH(
    net_u, x_f, y_f, t_f,
    x_b, y_b, t_b,
    x_i, y_i, t_i, ic,
    lambda_bc=1.0, lambda_ic=1.0,
    gamma=0.016,
    eta=1.0,      # 边界扩散系数
    sigma=1.0     # 边界相互作用系数
):
    """
    计算带有动态边界条件的 Cahn-Hilliard 损失函数。
    修正了法向导数和 Laplace-Beltrami 算子的计算。
    """
    # --- 1. PDE Loss (体相) ---
    # 保持原有逻辑，确保 compute_ch_pde_residual 内部使用的是 f(u) 而非 f'(u)
    pde_residual = compute_ch_pde_residual(
        net_u, x_f, y_f, t_f,
        gamma=gamma,
        mobility_type="degenerate"
    )
    loss_pde = torch.mean(pde_residual ** 2)

    # --- 2. Boundary Loss (边界) ---
    x_b.requires_grad_(True)
    y_b.requires_grad_(True)
    t_b.requires_grad_(True)

    # 网络输出与相场变量
    u_raw_b = net_u(x_b, y_b, t_b)
    u_b = torch.tanh(u_raw_b)

    eps = 1e-8  # 边界掩码容差
    loss_bc = 0.0
    count = 0
    # ========================
    # 2. 边界条件残差 (修正版：先求导，后切片)
    # ========================
    x_b.requires_grad_(True)
    y_b.requires_grad_(True)
    t_b.requires_grad_(True)

    # 获取边界上的预测值
    u_raw_b = net_u(x_b, y_b, t_b)
    u_b = torch.tanh(u_raw_b)

    eps = 1e-8
    loss_bc = 0.0
    count = 0

    # -------------------------------------------------------
    # 0. 预计算全局导数 (针对整个 Batch)
    # -------------------------------------------------------
    # 这一步非常关键：在切片之前，基于完整的张量计算所有需要的导数
    # 这样可以保证计算图是连通的

    # --- u 的导数 ---
    # 一阶
    u_b_x = torch.autograd.grad(u_b, x_b, grad_outputs=torch.ones_like(u_b), create_graph=True)[0]
    u_b_y = torch.autograd.grad(u_b, y_b, grad_outputs=torch.ones_like(u_b), create_graph=True)[0]
    u_b_t = torch.autograd.grad(u_b, t_b, grad_outputs=torch.ones_like(u_b), create_graph=True)[0]

    # 二阶
    u_b_xx = torch.autograd.grad(u_b_x, x_b, grad_outputs=torch.ones_like(u_b_x), create_graph=True)[0]
    u_b_yy = torch.autograd.grad(u_b_y, y_b, grad_outputs=torch.ones_like(u_b_y), create_graph=True)[0]

    # --- mu 的计算与导数 ---
    # 1. 先计算全场的 Laplacian (u_xx + u_yy)
    delta_u_bulk_full = u_b_xx + u_b_yy

    # 2. 计算全场的化学势 mu
    # 注意：这里使用的是完整的 u_b 和 delta_u_bulk_full
    mu_b_full = -gamma * delta_u_bulk_full + logarithmic_potential_derivative(u_b)

    # 3. 计算 mu 对坐标的梯度 (全场)
    mu_b_x_full = torch.autograd.grad(mu_b_full, x_b, grad_outputs=torch.ones_like(mu_b_full), create_graph=True)[0]
    mu_b_y_full = torch.autograd.grad(mu_b_full, y_b, grad_outputs=torch.ones_like(mu_b_full), create_graph=True)[0]

    # -------------------------------------------------------
    # 1. 下边界 (y = -1)
    # -------------------------------------------------------
    mask_bottom = (y_b <= -1 + eps)
    if mask_bottom.any():
        # --- 提取切片数据 ---
        u_b_mask = u_b[mask_bottom]
        u_b_t_mask = u_b_t[mask_bottom]
        u_b_y_mask = u_b_y[mask_bottom]
        u_b_xx_mask = u_b_xx[mask_bottom]
        u_b_yy_mask = u_b_yy[mask_bottom]
        
        # 从预计算的全场张量中提取 mu 和 mu 的导数
        mu_b = mu_b_full[mask_bottom]
        mu_b_y_val = mu_b_y_full[mask_bottom]

        # --- 物理量计算 ---
        # 表面拉普拉斯 (切向为 x)
        laplace_beltrami_u = u_b_xx_mask

        # 法向导数 (n 指向 -y，所以 dn = -dy)
        mu_b_n = -mu_b_y_val
        u_v = -u_b_y_mask

        # --- 残差方程 ---
        res_bc_1 = u_b_t_mask - (eta * laplace_beltrami_u - mu_b_n)
        res_bc_2 = mu_b - (-sigma * laplace_beltrami_u + u_v + g(u_b_mask))

        loss_bc += torch.mean(res_bc_1**2 + res_bc_2**2)
        count += 1

    # -------------------------------------------------------
    # 2. 上边界 (y = 1)
    # -------------------------------------------------------
    mask_top = (y_b >= 1 - eps)
    if mask_top.any():
        # --- 提取切片数据 ---
        u_b_mask = u_b[mask_top]
        u_b_t_mask = u_b_t[mask_top]
        u_b_y_mask = u_b_y[mask_top]
        u_b_xx_mask = u_b_xx[mask_top]
        u_b_yy_mask = u_b_yy[mask_top]
        
        # 从预计算的全场张量中提取 mu 和 mu 的导数
        mu_b = mu_b_full[mask_top]
        mu_b_y_val = mu_b_y_full[mask_top]

        # --- 物理量计算 ---
        # 表面拉普拉斯 (切向为 x)
        laplace_beltrami_u = u_b_xx_mask

        # 法向导数 (n 指向 +y)
        mu_b_n = mu_b_y_val
        u_v = u_b_y_mask

        # --- 残差方程 ---
        res_bc_1 = u_b_t_mask - (eta * laplace_beltrami_u - mu_b_n)
        res_bc_2 = mu_b - (-sigma * laplace_beltrami_u + u_v + g(u_b_mask))

        loss_bc += torch.mean(res_bc_1**2 + res_bc_2**2)
        count += 1

    # -------------------------------------------------------
    # 3. 左边界 (x = -1)
    # -------------------------------------------------------
    mask_left = (x_b <= -1 + eps)
    if mask_left.any():
        # --- 提取切片数据 ---
        u_b_mask = u_b[mask_left]
        u_b_t_mask = u_b_t[mask_left]
        u_b_x_mask = u_b_x[mask_left]
        u_b_xx_mask = u_b_xx[mask_left]
        u_b_yy_mask = u_b_yy[mask_left]
        
        # 从预计算的全场张量中提取 mu 和 mu 的导数
        mu_b = mu_b_full[mask_left]
        mu_b_x_val = mu_b_x_full[mask_left]

        # --- 物理量计算 ---
        # 表面拉普拉斯 (切向为 y)
        laplace_beltrami_u = u_b_yy_mask

        # 法向导数 (n 指向 -x)
        mu_b_n = -mu_b_x_val
        u_v = -u_b_x_mask

        # --- 残差方程 ---
        res_bc_1 = u_b_t_mask - (eta * laplace_beltrami_u - mu_b_n)
        res_bc_2 = mu_b - (-sigma * laplace_beltrami_u + u_v + g(u_b_mask))

        loss_bc += torch.mean(res_bc_1**2 + res_bc_2**2)
        count += 1

    # -------------------------------------------------------
    # 4. 右边界 (x = 1)
    # -------------------------------------------------------
    mask_right = (x_b >= 1 - eps)
    if mask_right.any():
        # --- 提取切片数据 ---
        u_b_mask = u_b[mask_right]
        u_b_t_mask = u_b_t[mask_right]
        u_b_x_mask = u_b_x[mask_right]
        u_b_xx_mask = u_b_xx[mask_right]
        u_b_yy_mask = u_b_yy[mask_right]
        
        # 从预计算的全场张量中提取 mu 和 mu 的导数
        mu_b = mu_b_full[mask_right]
        mu_b_x_val = mu_b_x_full[mask_right]

        # --- 物理量计算 ---
        # 表面拉普拉斯 (切向为 y)
        laplace_beltrami_u = u_b_yy_mask

        # 法向导数 (n 指向 +x)
        mu_b_n = mu_b_x_val
        u_v = u_b_x_mask

        # --- 残差方程 ---
        res_bc_1 = u_b_t_mask - (eta * laplace_beltrami_u - mu_b_n)
        res_bc_2 = mu_b - (-sigma * laplace_beltrami_u + u_v + g(u_b_mask))

        loss_bc += torch.mean(res_bc_1**2 + res_bc_2**2)
        count += 1

    loss_bc = (loss_bc / max(count, 1)) * lambda_bc
    # --- 3. Initial Condition ---
    u_raw_i = net_u(x_i, y_i, t_i)
    u_i = torch.tanh(u_raw_i)
    loss_ic = torch.mean((u_i - ic) ** 2) * lambda_ic
    # ----4. mass Loss----
    loss_mass = compute_mass_residual(net_u, x_f, y_f, t_f, x_b, y_b, t_b, gamma=0.016)
    total_loss = loss_pde + loss_bc + loss_ic + loss_mass
    return total_loss, loss_pde, loss_bc, loss_ic


def compute_drm_loss_CH(
    net_u, net_mu,
    x_f, y_f, t_f,
    x_b, y_b, t_b,
    x_i, y_i, t_i, ic,
    lambda_bc=1.0,
    lambda_ic=1.0,
    lambda_eq1=1.0,
    lambda_eq2=1.0,
    gamma=0.016,
    eta=1.0,
    sigma=1.0,
    mobility_type="degenerate"
):
    """
    双网络深度混合残量法 (DRM) 损失函数 for Cahn-Hilliard 方程。
    修正了边界条件的计算逻辑和梯度计算错误。
    """
    # ========================
    # 1. 在内部点计算 PDE 残差
    # ========================
    x_f.requires_grad_(True)
    y_f.requires_grad_(True)
    t_f.requires_grad_(True)

    # --- 网络预测 ---
    u_raw_f = net_u(x_f, y_f, t_f)
    u = torch.tanh(u_raw_f)
    mu_pred = net_mu(x_f, y_f, t_f)

    # ---- Equation (A): mu + gamma * Δu - f'(u) = 0 ----
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_f, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    lap_u = u_xx + u_yy

    f_prime = logarithmic_potential_derivative(u)
    residual_eq1 = mu_pred + gamma * lap_u - f_prime
    loss_eq1 = torch.mean(residual_eq1 ** 2) * lambda_eq1

    # ---- Equation (B): u_t - ∇·(M(u) ∇mu) = 0 ----
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
    # 2. 边界条件残差 (修正版：先求导，后切片)
    # ========================
    x_b.requires_grad_(True)
    y_b.requires_grad_(True)
    t_b.requires_grad_(True)

    # 获取边界上的预测值
    u_raw_b = net_u(x_b, y_b, t_b)
    u_b = torch.tanh(u_raw_b)
    mu_b = net_mu(x_b, y_b, t_b)

    eps = 1e-8
    loss_bc = 0.0
    count = 0

    # -------------------------------------------------------
    # 0. 预计算全局导数 (针对整个 Batch)
    # -------------------------------------------------------
    # 这样做可以避免在每个边界判断中重复计算梯度，且保证了计算图的完整性

    # --- u 的导数 ---
    # 一阶
    u_b_x = torch.autograd.grad(u_b, x_b, grad_outputs=torch.ones_like(u_b), create_graph=True)[0]
    u_b_y = torch.autograd.grad(u_b, y_b, grad_outputs=torch.ones_like(u_b), create_graph=True)[0]
    u_b_t = torch.autograd.grad(u_b, t_b, grad_outputs=torch.ones_like(u_b), create_graph=True)[0]

    # 二阶
    u_b_xx = torch.autograd.grad(u_b_x, x_b, grad_outputs=torch.ones_like(u_b_x), create_graph=True)[0]
    u_b_yy = torch.autograd.grad(u_b_y, y_b, grad_outputs=torch.ones_like(u_b_y), create_graph=True)[0]

    # --- mu 的导数 ---
    # 我们只需要 mu 对 x 和 y 的一阶导数
    mu_b_x = torch.autograd.grad(mu_b, x_b, grad_outputs=torch.ones_like(mu_b), create_graph=True)[0]
    mu_b_y = torch.autograd.grad(mu_b, y_b, grad_outputs=torch.ones_like(mu_b), create_graph=True)[0]

    # -------------------------------------------------------
    # 1. 上边界 (y = 1)
    # -------------------------------------------------------
    mask_top = (y_b >= 1 - eps)
    if mask_top.any():
        # 提取边界点的值
        u_b_mask = u_b[mask_top]
        u_b_t_mask = u_b_t[mask_top]
        u_b_y_mask = u_b_y[mask_top]
        u_b_xx_mask = u_b_xx[mask_top]
        
        mu_b_mask = mu_b[mask_top]
        mu_b_y_mask = mu_b_y[mask_top] # 提取 mu_y 在边界上的值

        # 表面拉普拉斯 (切向为 x)
        laplace_beltrami_u = u_b_xx_mask

        # 法向导数 (n 指向 +y, dn = dy)
        mu_b_n = mu_b_y_mask
        u_v = u_b_y_mask

        res_bc_1 = u_b_t_mask - (eta * laplace_beltrami_u - mu_b_n)
        res_bc_2 = mu_b_mask - (-sigma * laplace_beltrami_u + u_v + g(u_b_mask))

        loss_bc += torch.mean(res_bc_1**2 + res_bc_2**2)
        count += 1

    # -------------------------------------------------------
    # 2. 下边界 (y = -1)
    # -------------------------------------------------------
    mask_bottom = (y_b <= -1 + eps)
    if mask_bottom.any():
        # 提取边界点的值
        u_b_mask = u_b[mask_bottom]
        u_b_t_mask = u_b_t[mask_bottom]
        u_b_y_mask = u_b_y[mask_bottom]
        u_b_xx_mask = u_b_xx[mask_bottom]
        
        mu_b_mask = mu_b[mask_bottom]
        mu_b_y_mask = mu_b_y[mask_bottom]

        # 表面拉普拉斯 (切向为 x)
        laplace_beltrami_u = u_b_xx_mask

        # 法向导数 (n 指向 -y, dn = -dy)
        mu_b_n = -mu_b_y_mask
        u_v = -u_b_y_mask

        res_bc_1 = u_b_t_mask - (eta * laplace_beltrami_u - mu_b_n)
        res_bc_2 = mu_b_mask - (-sigma * laplace_beltrami_u + u_v + g(u_b_mask))

        loss_bc += torch.mean(res_bc_1**2 + res_bc_2**2)
        count += 1

    # -------------------------------------------------------
    # 3. 左边界 (x = -1)
    # -------------------------------------------------------
    mask_left = (x_b <= -1 + eps)
    if mask_left.any():
        # 提取边界点的值
        u_b_mask = u_b[mask_left]
        u_b_t_mask = u_b_t[mask_left]
        u_b_x_mask = u_b_x[mask_left]
        u_b_yy_mask = u_b_yy[mask_left]
        
        mu_b_mask = mu_b[mask_left]
        mu_b_x_mask = mu_b_x[mask_left]

        # 表面拉普拉斯 (切向为 y)
        laplace_beltrami_u = u_b_yy_mask

        # 法向导数 (n 指向 -x, dn = -dx)
        mu_b_n = -mu_b_x_mask
        u_v = -u_b_x_mask

        res_bc_1 = u_b_t_mask - (eta * laplace_beltrami_u - mu_b_n)
        res_bc_2 = mu_b_mask - (-sigma * laplace_beltrami_u + u_v + g(u_b_mask))

        loss_bc += torch.mean(res_bc_1**2 + res_bc_2**2)
        count += 1

    # -------------------------------------------------------
    # 4. 右边界 (x = 1)
    # -------------------------------------------------------
    mask_right = (x_b >= 1 - eps)
    if mask_right.any():
        # 提取边界点的值
        u_b_mask = u_b[mask_right]
        u_b_t_mask = u_b_t[mask_right]
        u_b_x_mask = u_b_x[mask_right]
        u_b_yy_mask = u_b_yy[mask_right]
        
        mu_b_mask = mu_b[mask_right]
        mu_b_x_mask = mu_b_x[mask_right]

        # 表面拉普拉斯 (切向为 y)
        laplace_beltrami_u = u_b_yy_mask

        # 法向导数 (n 指向 +x, dn = dx)
        mu_b_n = mu_b_x_mask
        u_v = u_b_x_mask

        res_bc_1 = u_b_t_mask - (eta * laplace_beltrami_u - mu_b_n)
        res_bc_2 = mu_b_mask - (-sigma * laplace_beltrami_u + u_v + g(u_b_mask))

        loss_bc += torch.mean(res_bc_1**2 + res_bc_2**2)
        count += 1

    loss_bc = (loss_bc / max(count, 1)) * lambda_bc
    # ========================
    # 3. 初始条件
    # ========================
    u_raw_i = net_u(x_i, y_i, t_i)
    u_i = torch.tanh(u_raw_i)
    loss_ic = torch.mean((u_i - ic) ** 2) * lambda_ic

    
    # ========================
    # 5. 质量损失
    # ========================
    
    loss_mass = compute_mass_residual(net_u, x_f, y_f, t_f, x_b, y_b, t_b, gamma=0.016)
    # ========================
    # 4. 总损失
    # ========================
    total_loss = loss_eq1 + loss_eq2 + loss_bc + loss_ic+ loss_mass

    return total_loss, loss_eq1, loss_eq2, loss_bc, loss_ic
