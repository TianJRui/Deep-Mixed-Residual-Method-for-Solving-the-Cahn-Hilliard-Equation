# data.py
import torch
import numpy as np
from config import DEVICE, T_FINAL, T_REVERSE
from initial_condition import compute_initial_condition as compute_ic_original
from losses import compute_ch_pde_residual  # 导入计算残差的函数

def tensors(arr):
    return torch.tensor(arr, dtype=torch.float32).to(DEVICE)

def generate_points_in_domain(N, device=DEVICE):
    """在 [-1, 1]^2 域上生成 N 个随机点"""
    x = torch.rand(N, 1, device=device) * 2 - 1
    y = torch.rand(N, 1, device=device) * 2 - 1
    return x, y

def generate_data(N_f=20000, N_b=800, N_i=800, device=DEVICE, mode='forward'):
    """
    基于时间域平移策略生成数据。
    
    策略说明:
    1. Reverse Mode (反向寻源):
       - 时间域: t /in [-T_REVERSE, 0]
       - IC (t = -T_REVERSE): 原始尖锐初值 (对应原问题的 t=0)
       - 目标: 学习从尖锐态演化到 t=0 的平滑态。
       
    2. Forward Mode (全量求解):
       - 时间域: t /in [-T_REVERSE, T_FINAL]
       - IC (t = -T_REVERSE): 原始尖锐初值
       - 优势: 覆盖了反向阶段，利用反向训练好的权重或中间平滑特性辅助收敛。
    """
    
    # 确定时间边界
    if mode == 'reverse':
        t_start = -T_REVERSE
        t_end = 0.0
    elif mode == 'forward':
        t_start = -T_REVERSE
        t_end = T_FINAL
    else:
        raise ValueError("Mode must be 'forward' or 'reverse'")
    
    t_span = t_end - t_start

    # --- 1. 内部残差点 (PDE Loss) ---
    # 时间均匀分布在 [t_start, t_end]
    x_f, y_f = generate_points_in_domain(N_f, device)
    t_f = torch.rand(N_f, 1, device=device) * t_span + t_start

    # --- 2. 边界点 (BC Loss) ---
    n_each = N_b // 4
    # 生成空间坐标
    x_b = torch.rand(n_each, 1, device=device) * 2 - 1
    y_b = -torch.ones(n_each, 1, device=device)
    x_t = torch.rand(n_each, 1, device=device) * 2 - 1
    y_t = torch.ones(n_each, 1, device=device)
    x_l = -torch.ones(n_each, 1, device=device)
    y_l = torch.rand(n_each, 1, device=device) * 2 - 1
    x_r = torch.ones(n_each, 1, device=device)
    y_r = torch.rand(n_each, 1, device=device) * 2 - 1

    x_b_all = torch.cat([x_b, x_t, x_l, x_r], dim=0)
    y_b_all = torch.cat([y_b, y_t, y_l, y_r], dim=0)
    # 边界点时间在 [t_start, t_end]
    t_b = torch.rand(N_b, 1, device=device) * t_span + t_start

    # --- 3. 初始条件点 (IC Loss) ---
    # 关键改动：初始条件始终施加在 t = t_start (即 -T_REVERSE)
    x_i, y_i = generate_points_in_domain(N_i, device)
    t_i = torch.full((N_i, 1), t_start, device=device)
    
    # 计算原始尖锐初值 (无论模式如何，起点都是这个尖锐状态)
    ic_values = compute_ic_original(x_i, y_i)

    # --- 4. 划分训练/验证集 ---
    split_f = int(0.8 * N_f)
    split_b = int(0.8 * N_b)
    split_i = int(0.8 * N_i)

    data = {
        'mode': mode,
        't_start': t_start,
        't_end': t_end,
        't_span': t_span,
        # Interior
        'x_f_train': x_f[:split_f], 'y_f_train': y_f[:split_f], 't_f_train': t_f[:split_f],
        'x_f_val': x_f[split_f:], 'y_f_val': y_f[split_f:], 't_f_val': t_f[split_f:],
        # Boundary
        'x_b_train': x_b_all[:split_b], 'y_b_train': y_b_all[:split_b], 't_b_train': t_b[:split_b],
        'x_b_val': x_b_all[split_b:], 'y_b_val': y_b_all[split_b:], 't_b_val': t_b[split_b:],
        # Initial Condition (at t_start)
        'x_i_train': x_i[:split_i], 'y_i_train': y_i[:split_i], 't_i_train': t_i[:split_i],
        'x_i_val': x_i[split_i:], 'y_i_val': y_i[split_i:], 't_i_val': t_i[split_i:],
        'ic_train': ic_values[:split_i],
        'ic_val': ic_values[split_i:]
    }

    return data

def resample_interior_points_adaptive(net_u, N_f, device=DEVICE, t_start=None, t_span=None,reg=1e-4, n_candidates_factor=10)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    基于残差的自适应内部点重采样 (Residual-based Adaptive Sampling)
    
    策略:
    1. 生成 N_f * factor 个候选点 (均匀分布)。
    2. 计算这些点的 PDE 残差绝对值 |R|。
    3. 根据 |R| 的大小作为权重，从中采样 N_f 个点。
       残差越大的点，被选中的概率越高。
    """
    if t_start is None or t_span is None:
        raise ValueError("t_start and t_span must be provided")
    
    if net_u is None:
        # 如果还没有模型（初始阶段），退化为均匀采样
        [x,y] = generate_points_in_domain(N_f, device)
        return x,y, torch.rand(N_f, 1, device=device) * t_span + t_start

    net_u.eval()
    
    # 1. 生成候选点池 (比需要的点多几倍)
    N_candidates = N_f * n_candidates_factor
    x_cand, y_cand = generate_points_in_domain(N_candidates, device)
    t_cand = torch.rand(N_candidates, 1, device=device) * t_span + t_start
    
    # 确保需要梯度以计算残差
    x_cand.requires_grad_(True)
    y_cand.requires_grad_(True)
    t_cand.requires_grad_(True)
    
    # 2. 计算候选点的 PDE 残差
    # 注意：这里需要传入正确的 eps_reg，如果无法获取，可使用默认值或从 config 读取
    # 为了简化，这里假设使用默认正则化参数，或者你可以从 main.py 传递进来
    try:
        residual = compute_ch_pde_residual(net_u, x_cand, y_cand, t_cand)
    except Exception:
        # 如果计算失败（例如梯度问题），回退到均匀采样
        x_cand.requires_grad_(False)
        t_cand.requires_grad_(False)
        idx = torch.randperm(N_candidates)[:N_f]
        return x_cand[idx], y_cand[idx], t_cand[idx]

    # 3. 计算采样权重
    # 使用残差的绝对值作为权重基础
    residual_abs = torch.abs(residual).detach()
    
    # 避免权重为0，添加一个小量，并进行归一化
    # 策略 A: 直接使用残差大小作为概率权重
    weights = residual_abs + 1e-8 
    
    # 归一化概率分布
    probs = weights / torch.sum(weights)
    
    # 4. 根据概率分布进行多项式采样 (Multinomial Sampling)
    # torch.multinomial 返回的是索引
    indices = torch.multinomial(probs.squeeze(), N_f, replacement=True)
    
    # 5. 选取对应的点
    x_new = x_cand[indices].detach()
    y_new = y_cand[indices].detach()
    t_new = t_cand[indices].detach()
    
    # 重置梯度状态
    x_new.requires_grad_(False)
    y_new.requires_grad_(False)
    t_new.requires_grad_(False)
    
    return x_new, y_new, t_new
def resample_interior_points(N_f, device=DEVICE, t_start=None, t_span=None):
    if t_start is None or t_span is None:
        raise ValueError("t_start and t_span must be provided")
    x_f, y_f = generate_points_in_domain(N_f, device)
    t_f = torch.rand(N_f, 1, device=device) * t_span + t_start
    return x_f, y_f, t_f

from scipy.stats.qmc import LatinHypercube

def resample_boundary_points(N_b, device, t_start=None, t_span=None):
    """
    使用拉丁超立方采样 (LHS) 生成边界点。
    """
    if t_start is None or t_span is None:
        raise ValueError("t_start and t_span must be provided")

    # 确保 N_b 能被 4 整除，或者处理余数
    # 这里为了保持逻辑简单，假设 N_b 是 4 的倍数，或者取整
    n_each = N_b // 4
    
    # --- 1. 空间采样 (使用 LHS) ---
    # 每条边是一个 1D 的采样问题 (位置坐标在 [-1, 1] 之间)
    sampler = LatinHypercube(d=1)
    
    # Bottom (y = -1): 采样 x
    sample_x = sampler.random(n=n_each)
    x_b = torch.tensor(2 * sample_x - 1, dtype=torch.float32, device=device) # 映射到 [-1, 1]
    y_b = -torch.ones(n_each, 1, dtype=torch.float32, device=device)
    
    # Top (y = 1): 采样 x
    sample_x = sampler.random(n=n_each)
    x_t = torch.tensor(2 * sample_x - 1, dtype=torch.float32, device=device)
    y_t = torch.ones(n_each, 1, dtype=torch.float32, device=device)
    
    # Left (x = -1): 采样 y
    sample_y = sampler.random(n=n_each)
    y_l = torch.tensor(2 * sample_y - 1, dtype=torch.float32, device=device)
    x_l = -torch.ones(n_each, 1, dtype=torch.float32, device=device)
    
    # Right (x = 1): 采样 y
    sample_y = sampler.random(n=n_each)
    y_r = torch.tensor(2 * sample_y - 1, dtype=torch.float32, device=device)
    x_r = torch.ones(n_each, 1, dtype=torch.float32, device=device)
    
    # --- 2. 时间采样 (使用 LHS) ---
    # 时间也是 1D 采样
    sampler_t = LatinHypercube(d=1)
    sample_t = sampler_t.random(n=N_b)
    t_b = torch.tensor(sample_t * t_span + t_start, dtype=torch.float32, device=device)
    
    # --- 3. 拼接 ---
    x_b_all = torch.cat([x_b, x_t, x_l, x_r], dim=0)
    y_b_all = torch.cat([y_b, y_t, y_l, y_r], dim=0)
    
    # 确保拼接后的空间点数量与时间点数量匹配
    # 如果 N_b 不能被 4 整除，这里可能会有尺寸不匹配的问题，
    # 但原代码逻辑也是这样，所以保持一致。
    return x_b_all, y_b_all, t_b
def resample_initial_points(N_i, device=DEVICE):
    x_i, y_i = generate_points_in_domain(N_i, device)
    t_i = torch.zeros(N_i, 1, device=device)
    ic = compute_ic_original(x_i, y_i)
    return x_i, y_i, t_i, ic