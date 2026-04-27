# data.py
import torch
import numpy as np
from config import DEVICE, DOMAIN_SIZE

def tensors(arr):
    return torch.tensor(arr, dtype=torch.float32).to(DEVICE)

def generate_data(N_f=20000, N_b=800, N_i=800, device=DEVICE):
    L = 1.0
    x_f = np.random.uniform(0, L, N_f).reshape(-1, 1)
    y_f = np.random.uniform(0, L, N_f).reshape(-1, 1)
    t_f = np.random.uniform(0, 10, N_f).reshape(-1, 1)

    x_b_bottom = np.random.uniform(0, L, N_b//4).reshape(-1, 1)
    y_b_bottom = np.zeros((N_b//4, 1))
    x_b_top = np.random.uniform(0, L, N_b//4).reshape(-1, 1)
    y_b_top = np.ones((N_b//4, 1))
    x_b_left = np.zeros((N_b//4, 1))
    y_b_left = np.random.uniform(0, L, N_b//4).reshape(-1, 1)
    x_b_right = np.ones((N_b//4, 1))
    y_b_right = np.random.uniform(0, L, N_b//4).reshape(-1, 1)

    x_b = np.vstack([x_b_bottom, x_b_top, x_b_left, x_b_right])
    y_b = np.vstack([y_b_bottom, y_b_top, y_b_left, y_b_right])
    t_b = np.random.uniform(0, 10, N_b).reshape(-1, 1)

    x_i = np.random.uniform(0, L, N_i).reshape(-1, 1)
    y_i = np.random.uniform(0, L, N_i).reshape(-1, 1)
    t_i = np.zeros((N_i, 1))
    ic = 0.05 * (np.cos(4 * np.pi * x_i) + np.cos(4 * np.pi * y_i))

    tensors = lambda arr: torch.tensor(arr, dtype=torch.float32).to(device)
    x_f, y_f, t_f = map(tensors, [x_f, y_f, t_f])
    x_b, y_b, t_b = map(tensors, [x_b, y_b, t_b])
    x_i, y_i, t_i, ic = map(tensors, [x_i, y_i, t_i, ic])

    split_f = int(0.8 * N_f)
    split_b = int(0.8 * N_b)
    split_i = int(0.8 * N_i)

    data = {
        'x_f_train': x_f[:split_f], 'y_f_train': y_f[:split_f], 't_f_train': t_f[:split_f],
        'x_f_val': x_f[split_f:], 'y_f_val': y_f[split_f:], 't_f_val': t_f[split_f:],
        'x_b_train': x_b[:split_b], 'y_b_train': y_b[:split_b], 't_b_train': t_b[:split_b],
        'x_b_val': x_b[split_b:], 'y_b_val': y_b[split_b:], 't_b_val': t_b[split_b:],
        'x_i_train': x_i[:split_i], 'y_i_train': y_i[:split_i], 't_i_train': t_i[:split_i],
        'x_i_val': x_i[split_i:], 'y_i_val': y_i[split_i:], 't_i_val': t_i[split_i:],
        'ic_train': ic[:split_i], 'ic_val': ic[split_i:]
    }
    return data


# --- 动态采样函数 ---
def resample_interior_points(N_f, device):
    L = DOMAIN_SIZE
    x_f = torch.rand(N_f, 1, device=device) * L
    y_f = torch.rand(N_f, 1, device=device) * L
    t_f = torch.rand(N_f, 1, device=device) * 10
    return x_f, y_f, t_f


def resample_boundary_points(N_b, device):
    L = 1.0
    n_each = N_b // 4
    # Bottom
    x_b = torch.rand(n_each, 1, device=device) * L
    y_b = torch.zeros(n_each, 1, device=device)
    # Top
    x_t = torch.rand(n_each, 1, device=device) * L
    y_t = torch.ones(n_each, 1, device=device)
    # Left
    x_l = torch.zeros(n_each, 1, device=device)
    y_l = torch.rand(n_each, 1, device=device) * L
    # Right
    x_r = torch.ones(n_each, 1, device=device)
    y_r = torch.rand(n_each, 1, device=device) * L

    x_b_all = torch.cat([x_b, x_t, x_l, x_r], dim=0)
    y_b_all = torch.cat([y_b, y_t, y_l, y_r], dim=0)
    t_b_all = torch.rand(N_b, 1, device=device)*10
    return x_b_all, y_b_all, t_b_all


def resample_initial_points(N_i, device):
    L = DOMAIN_SIZE
    x_i = torch.rand(N_i, 1, device=device) * L
    y_i = torch.rand(N_i, 1, device=device) * L
    t_i = torch.zeros(N_i, 1, device=device)
    ic = 0.05 * (torch.cos(4 * np.pi * x_i) + torch.cos(4 * np.pi * y_i))
    return x_i, y_i, t_i, ic
def generate_fixed_boundary_points(N_b_per_edge=100, device=DEVICE):
    L = 1.0
    s = torch.linspace(0, L, N_b_per_edge, device=device)
    
    # Bottom: y=0
    x_b = s.clone(); y_b = torch.zeros_like(s)
    # Top: y=1
    x_t = s.clone(); y_t = torch.ones_like(s)
    # Left: x=0
    x_l = torch.zeros_like(s); y_l = s.clone()
    # Right: x=1
    x_r = torch.ones_like(s); y_r = s.clone()

    x_b_all = torch.cat([x_b, x_t, x_l, x_r], dim=0).unsqueeze(1)
    y_b_all = torch.cat([y_b, y_t, y_l, y_r], dim=0).unsqueeze(1)
    t_b_all = torch.rand(x_b_all.shape[0], 1, device=device)  

    return x_b_all, y_b_all, t_b_all
