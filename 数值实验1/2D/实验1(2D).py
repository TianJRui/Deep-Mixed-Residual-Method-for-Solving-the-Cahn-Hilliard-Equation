"""
实验1（2D）：适配正则非线性项与Neumann边界的深度混合残量模型 vs 标准PINN
根据毕业论文第三章内容进行修改和增强
注意：本实验为2D Cahn-Hilliard方程，使用Neumann边界条件。
【修复版】：正确Neumann BC + 动态采样 + 初始条件强化
"""

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
import random  # 新增：导入random模块

# 固定随机种子，保证实验可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况下
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.cuda.init() 
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")
# ----------------------------
# 1. 定义 PINN 网络 (2D)
# ----------------------------
# ----------------------------
# 1. 定义 带 Fourier 特征的 PINN 网络 (2D)
# ----------------------------
class PINN(nn.Module):
    def __init__(self, hidden_dim=512, num_hidden_layers=5, activation='tanh', sigma=10.0, fourier_dim=64):
        """
        使用随机傅里叶特征映射处理空间坐标 (x, y)，提升高频拟合能力。
        - sigma: 控制频率尺度（越大，越能捕捉高频）
        - fourier_dim: 傅里叶映射后的维度（总输入维度 = 2 * fourier_dim + 1）
        """
        super(PINN, self).__init__()
        self.fourier_dim = fourier_dim
        self.sigma = sigma

        # 随机傅里叶矩阵 B ∈ R^{2 × fourier_dim}
        self.register_buffer('B', torch.randn(2, fourier_dim) * sigma)

        input_dim = 2 * fourier_dim + 1  # [sin(B·xy), cos(B·xy), t]
        output_dim = 1

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        if activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'relu':
            layers.append(nn.ReLU())
        else:
            raise ValueError("Unsupported activation")

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # 初始化权重（Xavier）
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        # 拼接空间坐标
        xy = torch.cat([x, y], dim=1)  # shape: [N, 2]

        # 傅里叶特征：proj = xy @ B → [N, fourier_dim]
        proj = torch.matmul(xy, self.B)  # 使用显式矩阵乘法函数 #type:ignore
        # 构造 sin/cos 特征
        fourier_feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)  # [N, 2 * fourier_dim]

        # 拼接时间 t
        input_feat = torch.cat([fourier_feat, t], dim=1)  # [N, 2*fourier_dim + 1]

        return self.net(input_feat)
# ----------------------------
# 3. 生成训练/验证数据 (2D Cahn-Hilliard with Neumann BC)
# ----------------------------
def generate_data(N_f=20000, N_b=800, N_i=800):
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

# ----------------------------
# 4. 正确的边界损失计算（按边分离）
# ----------------------------
def compute_boundary_loss(u_b, x_b, y_b, create_graph=True):
    u_b_x = torch.autograd.grad(u_b, x_b, grad_outputs=torch.ones_like(u_b), create_graph=create_graph)[0]
    u_b_y = torch.autograd.grad(u_b, y_b, grad_outputs=torch.ones_like(u_b), create_graph=create_graph)[0]

    eps = 1e-6
    mask_bottom = (y_b <= eps)
    mask_top = (y_b >= 1 - eps)
    mask_left = (x_b <= eps)
    mask_right = (x_b >= 1 - eps)

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

    return loss / max(count, 1)


# ----------------------------
# 5. 损失函数定义
# ----------------------------
def compute_standard_loss_ch(net_u, x_f, y_f, t_f, x_b, y_b, t_b, x_i, y_i, t_i, ic, lambda_bc, lambda_ic, eps):
    x_f.requires_grad_(True)
    y_f.requires_grad_(True)
    t_f.requires_grad_(True)
    u = net_u(x_f, y_f, t_f)

    u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_f, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    lap_u = u_xx + u_yy

    lap_u_x = torch.autograd.grad(lap_u, x_f, grad_outputs=torch.ones_like(lap_u), create_graph=True)[0]
    lap_u_y = torch.autograd.grad(lap_u, y_f, grad_outputs=torch.ones_like(lap_u), create_graph=True)[0]
    lap_u_xx = torch.autograd.grad(lap_u_x, x_f, grad_outputs=torch.ones_like(lap_u_x), create_graph=True)[0]
    lap_u_yy = torch.autograd.grad(lap_u_y, y_f, grad_outputs=torch.ones_like(lap_u_y), create_graph=True)[0]
    bi_lap_u = lap_u_xx + lap_u_yy

    nonlinear = u**3 - u
    nonlinear_x = torch.autograd.grad(nonlinear, x_f, grad_outputs=torch.ones_like(nonlinear), create_graph=True)[0]
    nonlinear_y = torch.autograd.grad(nonlinear, y_f, grad_outputs=torch.ones_like(nonlinear), create_graph=True)[0]
    nonlinear_xx = torch.autograd.grad(nonlinear_x, x_f, grad_outputs=torch.ones_like(nonlinear_x), create_graph=True)[0]
    nonlinear_yy = torch.autograd.grad(nonlinear_y, y_f, grad_outputs=torch.ones_like(nonlinear_y), create_graph=True)[0]
    lap_nonlinear = nonlinear_xx + nonlinear_yy

    pde_residual = u_t + eps**2 * bi_lap_u - lap_nonlinear
    loss_pde = torch.mean(pde_residual ** 2)

    # --- Correct Neumann BC ---
    x_b.requires_grad_(True)
    y_b.requires_grad_(True)
    t_b.requires_grad_(True)
    u_b = net_u(x_b, y_b, t_b)
    loss_bc = compute_boundary_loss(u_b, x_b, y_b, create_graph=True)
    loss_bc *= lambda_bc

    # Initial condition
    u_i = net_u(x_i, y_i, t_i)
    ic = 0.05 * (torch.cos(4 * torch.pi * x_i) + torch.cos(4 * torch.pi * y_i))

    loss_ic = torch.mean((u_i - ic) ** 2) * lambda_ic

    total_loss = loss_ic + loss_pde + loss_bc 
    return total_loss, loss_pde, loss_bc, loss_ic


def create_mixed_loss_ch(x_f, y_f, t_f, x_b, y_b, t_b, x_i, y_i, t_i, ic, lambda_bc, lambda_ic, eps):
    def loss_fn(net_u, net_mu):
        x_f.requires_grad_(True); y_f.requires_grad_(True); t_f.requires_grad_(True)
        u = net_u(x_f, y_f, t_f)
        mu = net_mu(x_f, y_f, t_f)

        u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        mu_x = torch.autograd.grad(mu, x_f, grad_outputs=torch.ones_like(mu), create_graph=True)[0]
        mu_y = torch.autograd.grad(mu, y_f, grad_outputs=torch.ones_like(mu), create_graph=True)[0]
        mu_xx = torch.autograd.grad(mu_x, x_f, grad_outputs=torch.ones_like(mu_x), create_graph=True)[0]
        mu_yy = torch.autograd.grad(mu_y, y_f, grad_outputs=torch.ones_like(mu_y), create_graph=True)[0]
        lap_mu = mu_xx + mu_yy
        loss_pde_1 = torch.mean((u_t - lap_mu) ** 2)

        u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y_f, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        lap_u = u_xx + u_yy
        f_prime_u = u**3 - u
        mu_pred = -eps**2 * lap_u + f_prime_u
        loss_pde_2 = torch.mean((mu - mu_pred) ** 2)


        # Initial condition
        u_i = net_u(x_i, y_i, t_i)
        ic = 0.05 * (torch.cos(4 * torch.pi * x_i) + torch.cos(4 * torch.pi * y_i))
        loss_ic = torch.mean((u_i - ic) ** 2) * lambda_ic

        # Boundary for u
        x_b.requires_grad_(True); y_b.requires_grad_(True); t_b.requires_grad_(True)
        u_b = net_u(x_b, y_b, t_b)

        loss_bc_u = compute_boundary_loss(u_b, x_b, y_b, create_graph=True)
        loss_bc = (loss_bc_u) * lambda_bc

        total_loss = loss_ic + loss_pde_1 + loss_pde_2 + loss_bc

        return total_loss, loss_pde_1, loss_bc, loss_ic
    return loss_fn


# ----------------------------
# 6. 动态采样函数（用于更新训练点）
# ----------------------------
def resample_interior_points(N_f, device):
    L = 1.0
    x_f = torch.rand(N_f, 1, device=device) * L
    y_f = torch.rand(N_f, 1, device=device)
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
    L = 1.0
    x_i = torch.rand(N_i, 1, device=device) * L
    y_i = torch.rand(N_i, 1, device=device) * L
    t_i = torch.zeros(N_i, 1, device=device)
    ic = 0.05 * (torch.cos(4 * np.pi * x_i) + torch.cos(4 * np.pi * y_i))
    return x_i, y_i, t_i, ic


# ----------------------------
# 7. 参数设置
# ----------------------------
hidden_dim = 512
num_hidden_layers = 5
learning_rate = 1e-4
num_epochs_full = 50000
eps = 0.01
resample_freq = 100  # 每100轮重采样一次

# Hyperparameter search: 强化 IC 权重
lambda_candidates = [1.0, 10.0, 100.0]
lambda_combinations = [(bc, ic) for bc in lambda_candidates for ic in [ 1.0, 10.0, 100.0]]  # λ_ic 至少 10
num_epochs_search = 5000
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
                                    dtype=torch.float32).to(device)

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

# # ----------------------------
# # 8. 超参数搜索（使用混合模型）
# # ----------------------------
def search_best_lambda(data, lambda_combinations, num_epochs_search, device):
    keys = ['x_f', 'y_f', 't_f', 'x_b', 'y_b', 't_b', 'x_i', 'y_i', 't_i', 'ic']
    train_data = {k+'_train': data[k+'_train'] for k in keys}
    val_data = {k+'_val': data[k+'_val'] for k in keys}

    print(f"🔍 Starting hyperparameter search: {len(lambda_combinations)} combinations")
    lambda_performance = {}
    for idx, (lambda_bc, lambda_ic) in enumerate(lambda_combinations, 1):
        print(f"\n--- Progress: {idx}/{len(lambda_combinations)} | Current: (λ_bc={lambda_bc}, λ_ic={lambda_ic}) ---")
        
        net_u = PINN(hidden_dim=512, num_hidden_layers=num_hidden_layers, activation='tanh').to(device)
        net_mu = PINN(hidden_dim=512, num_hidden_layers=num_hidden_layers, activation='tanh').to(device)
        optimizer = torch.optim.Adam(list(net_u.parameters()) + list(net_mu.parameters()), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

        loss_fn = create_mixed_loss_ch(
            train_data['x_f_train'], train_data['y_f_train'], train_data['t_f_train'],
            train_data['x_b_train'], train_data['y_b_train'], train_data['t_b_train'],
            train_data['x_i_train'], train_data['y_i_train'], train_data['t_i_train'],
            train_data['ic_train'], lambda_bc, lambda_ic, eps
        )
        val_loss_fn = create_mixed_loss_ch(
            val_data['x_f_val'], val_data['y_f_val'], val_data['t_f_val'],
            val_data['x_b_val'], val_data['y_b_val'], val_data['t_b_val'],
            val_data['x_i_val'], val_data['y_i_val'], val_data['t_i_val'],
            val_data['ic_val'], lambda_bc, lambda_ic, eps
        )
        
        best_val_loss = float('inf')
        for epoch in range(num_epochs_search):
            net_u.train(); net_mu.train()
            optimizer.zero_grad()
            train_loss, _, _, _ = loss_fn(net_u, net_mu)
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            
            net_u.eval(); net_mu.eval()
            val_loss, _, _, _ = val_loss_fn(net_u, net_mu)
            val_loss = val_loss.item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        
        lambda_performance[(lambda_bc, lambda_ic)] = best_val_loss
        print(f"  Done: Final Val Loss={best_val_loss:.3e}")
    
    best_lambda = min(lambda_performance.items(), key=lambda x: x[1])[0]
    print(f"\n🎉 Best (λ_bc, λ_ic): {best_lambda}")
    return best_lambda, lambda_performance


# # ----------------------------
# # 9. 执行超参搜索
# # ----------------------------
data = generate_data(N_f=1500, N_b=1500, N_i=1500)
best_lambda, lambda_perf = search_best_lambda(data, lambda_combinations, num_epochs_search, device)

# Heatmap visualization (unchanged)
import seaborn as sns
perf_matrix = np.zeros((len(lambda_candidates), len([1.0, 10.0, 100.0])))
ic_vals = [1.0, 10.0, 100.0]
for i, bc in enumerate(lambda_candidates):
    for j, ic in enumerate(ic_vals):
        perf_matrix[i, j] = lambda_perf.get((bc, ic), np.nan)

plt.figure(figsize=(8, 6))
sns.heatmap(
    perf_matrix,
    xticklabels=[str(ic) for ic in ic_vals],
    yticklabels=[str(bc) for bc in lambda_candidates],
    annot=True,
    fmt='.1e',
    cmap='YlOrRd_r',
    cbar_kws={'label': 'Validation Loss'}
)
plt.xlabel('λ_ic')
plt.ylabel('λ_bc')
plt.title('Hyperparameter Search Result (2D Cahn-Hilliard)')
plt.tight_layout()
plt.savefig('lambda_search_heatmap_2D_CH.png', dpi=150, bbox_inches='tight')

# ----------------------------
# 10. 主训练：Mixed Residual Model（带动态采样）
# ----------------------------
lambda_bc, lambda_ic = best_lambda
net_u_mixed = PINN(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, activation='tanh').to(device)
net_mu_mixed = PINN(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, activation='tanh').to(device)
optimizer_mixed = torch.optim.Adam(
    list(net_u_mixed.parameters()) + list(net_mu_mixed.parameters()),
    lr=learning_rate ,
    weight_decay=1e-4       # 可选 L2 正则
)
scheduler_mixed = torch.optim.lr_scheduler.StepLR(optimizer_mixed, step_size=5000, gamma=0.9)

# 初始数据
x_f_train, y_f_train, t_f_train = data['x_f_train'], data['y_f_train'], data['t_f_train']
x_b_train, y_b_train, t_b_train = data['x_b_train'], data['y_b_train'], data['t_b_train']
x_i_train, y_i_train, t_i_train = data['x_i_train'], data['y_i_train'], data['t_i_train']
ic_train = data['ic_train']

x_f_val, y_f_val, t_f_val = data['x_f_val'], data['y_f_val'], data['t_f_val']
x_b_val, y_b_val, t_b_val = data['x_b_val'], data['y_b_val'], data['t_b_val']
x_i_val, y_i_val, t_i_val = data['x_i_val'], data['y_i_val'], data['t_i_val']
ic_val = data['ic_val']

val_loss_fn = create_mixed_loss_ch(
    x_f_val, y_f_val, t_f_val,
    x_b_val, y_b_val, t_b_val,
    x_i_val, y_i_val, t_i_val,
    ic_val, lambda_bc, lambda_ic, eps
)

loss_history_mixed = []
start_time_mixed = time.time()
# ----------------------------
# 10. 主训练：Mixed Residual Model —— 三阶段训练策略
# ----------------------------
lambda_bc, lambda_ic = best_lambda

net_u_mixed = PINN(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, activation='tanh').to(device)
net_mu_mixed = PINN(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, activation='tanh').to(device)

optimizer_mixed = torch.optim.Adam(
    list(net_u_mixed.parameters()) + list(net_mu_mixed.parameters()),
    lr=learning_rate,
    weight_decay=1e-4
)
scheduler_mixed = torch.optim.lr_scheduler.StepLR(optimizer_mixed, step_size=5000, gamma=0.9)

# 初始数据
x_f_train, y_f_train, t_f_train = data['x_f_train'], data['y_f_train'], data['t_f_train']
x_b_train, y_b_train, t_b_train = data['x_b_train'], data['y_b_train'], data['t_b_train']
x_i_train, y_i_train, t_i_train = data['x_i_train'], data['y_i_train'], data['t_i_train']
ic_train = data['ic_train']

x_f_val, y_f_val, t_f_val = data['x_f_val'], data['y_f_val'], data['t_f_val']
x_b_val, y_b_val, t_b_val = data['x_b_val'], data['y_b_val'], data['t_b_val']
x_i_val, y_i_val, t_i_val = data['x_i_val'], data['y_i_val'], data['t_i_val']
ic_val = data['ic_val']

# 验证损失函数（完整）
val_loss_fn = create_mixed_loss_ch(
    x_f_val, y_f_val, t_f_val,
    x_b_val, y_b_val, t_b_val,
    x_i_val, y_i_val, t_i_val,
    ic_val, lambda_bc, lambda_ic, eps
)

loss_history_mixed = []
start_time_mixed = time.time()

optimizer_mixed1 = torch.optim.Adam(
    list(net_u_mixed.parameters()),
    lr=1e-4,
    weight_decay=1e-4       # 可选 L2 正则
)
scheduler_mixed1 = torch.optim.lr_scheduler.StepLR(optimizer_mixed1, step_size=5000, gamma=0.9)

# ======================
# 阶段1：仅训练初始条件
# ======================
print("🔹 Stage 1: Training Initial Condition Only")
num_epochs_stage1 = 10000
for epoch in range(num_epochs_stage1):
    if epoch % resample_freq == 0:
        N_f_train = x_f_train.shape[0]
        N_i_train = x_i_train.shape[0]
        x_f_train, y_f_train, t_f_train = resample_interior_points(N_f_train, device)
        x_i_train, y_i_train, t_i_train, ic_train = resample_initial_points(N_i_train, device)

    # 只计算 IC loss（其他损失设为0）
    u_i = net_u_mixed(x_i_train, y_i_train, t_i_train)
    ic_true = 0.05 * (torch.cos(4 * torch.pi * x_i_train) + torch.cos(4 * torch.pi * y_i_train))
    loss_ic = torch.mean((u_i - ic_true) ** 2) * lambda_ic
    loss = loss_ic  # 忽略 PDE 和 BC
    torch.nn.utils.clip_grad_norm_(net_u_mixed.parameters(), max_norm=1.0)    
    loss.backward()
    optimizer_mixed1.step()
    scheduler_mixed1.step()

    # 验证（仍用完整损失）
    val_loss, _, _, _ = val_loss_fn(net_u_mixed, net_mu_mixed)
    loss_history_mixed.append(val_loss.item())

    if epoch % 100 == 0:
        print(f'[Mixed Stage1] Epoch {epoch}, Train Loss (IC): {loss.item():.3e}, Val Loss: {val_loss.item():.3e}')

plot_2d_solution_snapshots(
    model=net_u_mixed,
    times=[0.0],
    nx=100, ny=100,
    x_range=(0, 1),
    y_range=(0, 1),
    filename_prefix="solution_2d_snapshots_t0"
)
# ======================
# 阶段2：IC + PDE
# ======================
print("🔹 Stage 2: Training IC + PDE")
num_epochs_stage2 = 40000
for epoch in range(num_epochs_stage2):
    if epoch % resample_freq == 0:
        N_f_train = x_f_train.shape[0]
        N_i_train = x_i_train.shape[0]
        x_f_train, y_f_train, t_f_train = resample_interior_points(N_f_train, device)
        x_i_train, y_i_train, t_i_train, ic_train = resample_initial_points(N_i_train, device)

    net_u_mixed.train(); net_mu_mixed.train()
    optimizer_mixed.zero_grad()
    
    # 构建仅含 IC + PDE 的损失（BC 权重设为0）
    x_f_train.requires_grad_(True); y_f_train.requires_grad_(True); t_f_train.requires_grad_(True)
    x_i_train.requires_grad_(False); y_i_train.requires_grad_(False); t_i_train.requires_grad_(False)
    
    u = net_u_mixed(x_f_train, y_f_train, t_f_train)
    mu = net_mu_mixed(x_f_train, y_f_train, t_f_train)

    # PDE1: u_t - Δμ = 0
    u_t = torch.autograd.grad(u, t_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    mu_x = torch.autograd.grad(mu, x_f_train, grad_outputs=torch.ones_like(mu), create_graph=True)[0]
    mu_y = torch.autograd.grad(mu, y_f_train, grad_outputs=torch.ones_like(mu), create_graph=True)[0]
    mu_xx = torch.autograd.grad(mu_x, x_f_train, grad_outputs=torch.ones_like(mu_x), create_graph=True)[0]
    mu_yy = torch.autograd.grad(mu_y, y_f_train, grad_outputs=torch.ones_like(mu_y), create_graph=True)[0]
    lap_mu = mu_xx + mu_yy
    loss_pde_1 = torch.mean((u_t - lap_mu) ** 2)

    # PDE2: μ = -ε²Δu + f'(u)
    u_x = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f_train, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_f_train, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    lap_u = u_xx + u_yy
    f_prime_u = u**3 - u
    mu_pred = -eps**2 * lap_u + f_prime_u
    loss_pde_2 = torch.mean((mu - mu_pred) ** 2)

    # IC
    u_i = net_u_mixed(x_i_train, y_i_train, t_i_train)
    ic_true = 0.05 * (torch.cos(4 * np.pi * x_i_train) + torch.cos(4 * np.pi * y_i_train))
    loss_ic = torch.mean((u_i - ic_true) ** 2) * lambda_ic

    loss = loss_ic + loss_pde_1 + loss_pde_2  # No BC

    torch.nn.utils.clip_grad_norm_(
    list(net_u_mixed.parameters()) + list(net_mu_mixed.parameters()),
    max_norm=1.0  # 关键！从 10.0 降到 1.0
)
    loss.backward()
    optimizer_mixed.step()
    scheduler_mixed.step()

    net_u_mixed.eval(); net_mu_mixed.eval()

    val_loss, _, _, _ = val_loss_fn(net_u_mixed, net_mu_mixed)
    loss_history_mixed.append(val_loss.item())

    if epoch % 100 == 0:
        print(f'[Mixed Stage2] Epoch {epoch}, Train Loss: {loss.item():.3e}, Val Loss: {val_loss.item():.3e}')

plot_2d_solution_snapshots(
    model=net_u_mixed,
    times=[0.0,2.5,5.0,7.5,10.0],
    nx=100, ny=100,
    x_range=(0, 1),
    y_range=(0, 1),
    filename_prefix="solution_2d_snapshots_t0_t1.0"
)

# ======================
# 阶段3：IC + PDE + BC
# ======================
def generate_fixed_boundary_points(N_b_per_edge=100):
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

N_b_train = x_b_train.shape[0]

x_b_train, y_b_train, t_b_train = generate_fixed_boundary_points(N_b_train)

print("🔹 Stage 3: Training Full Loss (IC + PDE + BC)")

num_epochs_stage3 = 40000

for epoch in range(num_epochs_stage3):
    if epoch % resample_freq == 0:
        N_f_train = x_f_train.shape[0]
        N_i_train = x_i_train.shape[0]
        x_f_train, y_f_train, t_f_train = resample_interior_points(N_f_train, device)
        x_i_train, y_i_train, t_i_train, ic_train = resample_initial_points(N_i_train, device)

    net_u_mixed.train(); net_mu_mixed.train()
    optimizer_mixed.zero_grad()
    torch.nn.utils.clip_grad_norm_(
    list(net_u_mixed.parameters()) + list(net_mu_mixed.parameters()),
    max_norm=1.0  # 关键！从 10.0 降到 1.0
)
    # 完整损失
    train_loss_fn = create_mixed_loss_ch(
        x_f_train, y_f_train, t_f_train,
        x_b_train, y_b_train, t_b_train,
        x_i_train, y_i_train, t_i_train,
        ic_train, lambda_bc, lambda_ic, eps
    )
    loss, loss_pde, loss_bc, loss_ic = train_loss_fn(net_u_mixed, net_mu_mixed)
    loss.backward()
    optimizer_mixed.step()
    scheduler_mixed.step()

    net_u_mixed.eval(); net_mu_mixed.eval()

    val_loss, _, _, _ = val_loss_fn(net_u_mixed, net_mu_mixed)
    loss_history_mixed.append(val_loss.item())

    if epoch % 100 == 0:
        print(f'[Mixed Stage3] Epoch {epoch}, Train Loss: {loss.item():.3e}, Val Loss: {val_loss.item():.3e}')

time_mixed = time.time() - start_time_mixed
plot_2d_solution_snapshots(
    model=net_u_mixed,
    times=[0.0,2.5,5.0,7.5,10.0],
    nx=100, ny=100,
    x_range=(0, 1),
    y_range=(0, 1),
    filename_prefix="solution_2d_snapshots_with_boundary_t0_t1.0"
)

print(f"✅ Mixed Residual PINN (3-stage) trained in {time_mixed:.2f} sec")

# ----------------------------
# 11. 主训练：Standard PINN（带动态采样）
# ----------------------------

net_u_std = PINN(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, activation='tanh').to(device)

optimizer_std = torch.optim.Adam(
    net_u_std.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)

scheduler_std = torch.optim.lr_scheduler.StepLR(optimizer_std, step_size=5000, gamma=0.9)

# 初始数据
x_f_train, y_f_train, t_f_train = data['x_f_train'], data['y_f_train'], data['t_f_train']
x_b_train, y_b_train, t_b_train = data['x_b_train'], data['y_b_train'], data['t_b_train']
x_i_train, y_i_train, t_i_train = data['x_i_train'], data['y_i_train'], data['t_i_train']
ic_train = data['ic_train']

x_f_val, y_f_val, t_f_val = data['x_f_val'], data['y_f_val'], data['t_f_val']
x_b_val, y_b_val, t_b_val = data['x_b_val'], data['y_b_val'], data['t_b_val']
x_i_val, y_i_val, t_i_val = data['x_i_val'], data['y_i_val'], data['t_i_val']
ic_val = data['ic_val']

# 验证损失函数（完整）
def val_loss_std():
    loss, _, _, _ = compute_standard_loss_ch(
            net_u_std,
            x_f_val, y_f_val, t_f_val,
            x_b_val, y_b_val, t_b_val,
            x_i_val, y_i_val, t_i_val,
            ic_val,
            lambda_bc, lambda_ic, eps
        )
    return loss.item()

loss_history_std = []
start_time_std = time.time()

# ======================
# 阶段1：仅训练初始条件
# ======================
print("🔹 Stage 1: Training Initial Condition Only (Standard PINN)")
num_epochs_stage1 = 10000
for epoch in range(num_epochs_stage1):

    net_u_std.train()
    optimizer_std.zero_grad()
    
    # 仅 IC 损失
    u_i = net_u_std(x_i_train, y_i_train, t_i_train)
    ic_true = 0.05 * (torch.cos(4 * np.pi * x_i_train) + torch.cos(4 * np.pi * y_i_train))
    loss_ic = torch.mean((u_i - ic_true) ** 2) * lambda_ic
    loss = loss_ic  # 忽略 PDE 和 BC
    
    loss.backward()
    optimizer_std.step()
    scheduler_std.step()

    net_u_std.eval()
    val_loss = val_loss_std()
    loss_history_std.append(val_loss)

    if epoch % 100 == 0:
        print(f'[Std Stage1] Epoch {epoch}, Train Loss (IC): {loss.item():.3e}, Val Loss: {val_loss:.3e}')
plot_2d_solution_snapshots(
    model=net_u_std,
    times=[0.0],
    nx=100, ny=100,
    x_range=(0, 1),
    y_range=(0, 1),
    filename_prefix="std_pinn_solution_2d_snapshots_t0"
)
# ======================
# 阶段2：IC + PDE（无 BC）
# ======================
print("🔹 Stage 2: Training IC + PDE (Standard PINN)")
num_epochs_stage2 = 40000
for epoch in range(num_epochs_stage2):
    if epoch % resample_freq == 0:
        N_f_train = x_f_train.shape[0]
        N_i_train = x_i_train.shape[0]
        x_f_train, y_f_train, t_f_train = resample_interior_points(N_f_train, device)
        x_i_train, y_i_train, t_i_train, ic_train = resample_initial_points(N_i_train, device)

    net_u_std.train()
    optimizer_std.zero_grad()
    
    # 手动计算 PDE 残差（但不加 BC）
    x_f_train.requires_grad_(True)
    y_f_train.requires_grad_(True)
    t_f_train.requires_grad_(True)

    u = net_u_std(x_f_train, y_f_train, t_f_train)

    u_t = torch.autograd.grad(u, t_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x_f_train, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_f_train, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    lap_u = u_xx + u_yy

    lap_u_x = torch.autograd.grad(lap_u, x_f_train, grad_outputs=torch.ones_like(lap_u), create_graph=True)[0]
    lap_u_y = torch.autograd.grad(lap_u, y_f_train, grad_outputs=torch.ones_like(lap_u), create_graph=True)[0]
    lap_u_xx = torch.autograd.grad(lap_u_x, x_f_train, grad_outputs=torch.ones_like(lap_u_x), create_graph=True)[0]
    lap_u_yy = torch.autograd.grad(lap_u_y, y_f_train, grad_outputs=torch.ones_like(lap_u_y), create_graph=True)[0]
    bi_lap_u = lap_u_xx + lap_u_yy

    nonlinear = u**3 - u
    nonlinear_x = torch.autograd.grad(nonlinear, x_f_train, grad_outputs=torch.ones_like(nonlinear), create_graph=True)[0]
    nonlinear_y = torch.autograd.grad(nonlinear, y_f_train, grad_outputs=torch.ones_like(nonlinear), create_graph=True)[0]
    nonlinear_xx = torch.autograd.grad(nonlinear_x, x_f_train, grad_outputs=torch.ones_like(nonlinear_x), create_graph=True)[0]
    nonlinear_yy = torch.autograd.grad(nonlinear_y, y_f_train, grad_outputs=torch.ones_like(nonlinear_y), create_graph=True)[0]
    lap_nonlinear = nonlinear_xx + nonlinear_yy

    pde_residual = u_t + eps**2 * bi_lap_u - lap_nonlinear
    loss_pde = torch.mean(pde_residual ** 2)

    # IC
    u_i = net_u_std(x_i_train, y_i_train, t_i_train)
    ic_true = 0.05 * (torch.cos(4 * np.pi * x_i_train) + torch.cos(4 * np.pi * y_i_train))
    loss_ic = torch.mean((u_i - ic_true) ** 2) * lambda_ic

    loss = loss_ic + loss_pde  # No BC

    loss.backward()
    optimizer_std.step()
    scheduler_std.step()

    net_u_std.eval()
    val_loss = val_loss_std()
    loss_history_std.append(val_loss)

    if epoch % 100 == 0:
        print(f'[Std Stage2] Epoch {epoch}, Train Loss: {loss.item():.3e}, Val Loss: {val_loss:.3e}')

plot_2d_solution_snapshots(
    model=net_u_std,
    times=[0.0,2.5,5.0,7.5,10.0],
    nx=100, ny=100,
    x_range=(0, 1),
    y_range=(0, 1),
    filename_prefix="std_pinn_solution_2d_snapshots_t0_t1"
)

# ======================
# 阶段3：完整损失（IC + PDE + BC）
# ======================

print("🔹 Stage 3: Training Full Loss (Standard PINN)")

num_epochs_stage3 = 40000

for epoch in range(num_epochs_stage3):

    if epoch % resample_freq == 0:
        
        N_f_train = x_f_train.shape[0]
        N_b_train = x_b_train.shape[0]
        N_i_train = x_i_train.shape[0]

        x_f_train, y_f_train, t_f_train = resample_interior_points(N_f_train, device)
        x_b_train, y_b_train, t_b_train = resample_boundary_points(N_b_train, device)
        x_i_train, y_i_train, t_i_train, ic_train = resample_initial_points(N_i_train, device)

    net_u_std.train()
    optimizer_std.zero_grad()
    
    # 完整损失（调用原函数）
    loss, loss_pde, loss_bc, loss_ic = compute_standard_loss_ch(
        net_u_std,
        x_f_train, y_f_train, t_f_train,
        x_b_train, y_b_train, t_b_train,
        x_i_train, y_i_train, t_i_train,
        ic_train,
        lambda_bc, lambda_ic, eps
    )
    
    loss.backward()
    
    optimizer_std.step()
    scheduler_std.step()

    net_u_std.eval()
    val_loss = val_loss_std()
    loss_history_std.append(val_loss)

    if epoch % 100 == 0:
        print(f'[Std Stage3] Epoch {epoch}, Train Loss: {loss.item():.3e}, Val Loss: {val_loss:.3e}')

time_std = time.time() - start_time_std
print(f"✅ Standard PINN (3-stage) trained in {time_std:.2f} sec")
plot_2d_solution_snapshots(
    model=net_u_std,
    times=[0.0,2.5,5.0,7.5,10.0],
    nx=100, ny=100,
    x_range=(0, 1),
    y_range=(0, 1),
    filename_prefix="std_pinn_solution_2d_snapshots_t0_t1_with_boundary"
)

# ----------------------------
# 12. 后续可视化与评估
# ----------------------------

L = 1.0
T_final = 10.0
x_test_line = np.linspace(0, L, 200)
y_test_line = np.full_like(x_test_line, 0.5)
t_test = np.array([0.0, 2.5, 5.0, 10.0])

def initial_condition_2d_ch(x, y):
    return 0.05 * (np.cos(4 * np.pi * x) + np.cos(4 * np.pi * y))

u_mixed_line = []
u_std_line = []

net_u_mixed.eval()
net_u_std.eval()


for t_val in t_test:
        t_tensor = torch.full((len(x_test_line), 1), t_val, dtype=torch.float32).to(device)
        x_tensor = torch.tensor(x_test_line[:, None], dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_test_line[:, None], dtype=torch.float32).to(device)
        u_pred_mixed = net_u_mixed(x_tensor, y_tensor, t_tensor).cpu().detach().numpy().flatten()
        u_pred_std = net_u_std(x_tensor, y_tensor, t_tensor).cpu().detach().numpy().flatten()
        u_mixed_line.append(u_pred_mixed)
        u_std_line.append(u_pred_std)

u_mixed_line = np.array(u_mixed_line)
u_std_line = np.array(u_std_line)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    ax.plot(x_test_line, u_mixed_line[i, :], 'b-', linewidth=2, label='Mixed Residual PINN')
    ax.plot(x_test_line, u_std_line[i, :], 'g--', linewidth=2, label='Standard PINN')
    if i == 0:
        ax.plot(x_test_line, initial_condition_2d_ch(x_test_line, y_test_line), 'r:', linewidth=2, label='Initial Condition')
    
    ax.set_title(f'$t = {t_test[i]}$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x, y=0.5, t)$')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('solution_comparison_2D_CH_line.png', dpi=150, bbox_inches='tight')


# Full 2D plot at t=0.0
t_plot = 0.0
x_grid = np.linspace(0, L, 100)
y_grid = np.linspace(0, L, 100)
X, Y = np.meshgrid(x_grid, y_grid)
x_flat = X.flatten()
y_flat = Y.flatten()
t_flat = np.full_like(x_flat, t_plot)

with torch.no_grad():
    x_tensor = torch.tensor(x_flat[:, None], dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_flat[:, None], dtype=torch.float32).to(device)
    t_tensor = torch.tensor(t_flat[:, None], dtype=torch.float32).to(device)
    u_mixed_2d = net_u_mixed(x_tensor, y_tensor, t_tensor).cpu().numpy().flatten()
    u_std_2d = net_u_std(x_tensor, y_tensor, t_tensor).cpu().numpy().flatten()

U_mixed_2d = u_mixed_2d.reshape(X.shape)
U_std_2d = u_std_2d.reshape(X.shape)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
im1 = axs[0].contourf(X, Y, U_mixed_2d, levels=50, cmap='viridis')
axs[0].set_title('Mixed Residual PINN (t=0.0)')
axs[0].set_xlabel('x'); axs[0].set_ylabel('y')
fig.colorbar(im1, ax=axs[0])
im2 = axs[1].contourf(X, Y, U_std_2d, levels=50, cmap='viridis')
axs[1].set_title('Standard PINN (t=0.0)')
axs[1].set_xlabel('x'); axs[1].set_ylabel('y')
fig.colorbar(im2, ax=axs[1])
plt.tight_layout()
plt.savefig('solution_comparison_2D_CH_contour.png', dpi=150, bbox_inches='tight')



# 假设 model_mixed 是你最终的模型
times_to_plot = [0.0,2.5,5.0,7.5,10.0]
plot_2d_solution_snapshots(
    model=net_u_mixed,
    times=times_to_plot,
    nx=100, ny=100,
    x_range=(0, 1),
    y_range=(0, 1),
    filename_prefix="mixed_solution_2d_snapshots_t0_t05_t1"
)
plot_2d_solution_snapshots(
    model=net_u_std,
    times=times_to_plot,
    nx=100, ny=100,
    x_range=(0, 1),
    y_range=(0, 1),
    filename_prefix="std_solution_2d_snapshots_t0_t05_t1"
)
# Mass conservation
from scipy.special import roots_legendre

def gauss_legendre_quadrature_2d(f, n=10):
    x_legendre, w_legendre = roots_legendre(n)
    x_scaled = (x_legendre + 1) / 2
    w_scaled = w_legendre / 2
    x_mesh, y_mesh = np.meshgrid(x_scaled, x_scaled, indexing='ij')
    w_mesh = np.outer(w_scaled, w_scaled)
    return np.sum(w_mesh * f(x_mesh, y_mesh))

# ----------------------------
# 修复版：初始质量（u0 的积分）
# ----------------------------
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


# ----------------------------
# 修复版：任意时刻的质量偏差（相对于初始质量）
# ----------------------------
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
    x_tensor = torch.tensor(x_flat, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_flat, dtype=torch.float32).to(device)

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

t_eval = np.linspace(0, T_final, 101)
u0_mean = compute_initial_mean_2d(gauss_n=20)
mass_mixed_2d = mass_history_2d(net_u_mixed, t_eval, u0_mean, gauss_n=20)
mass_std_2d = mass_history_2d(net_u_std, t_eval, u0_mean, gauss_n=20)

plt.figure(figsize=(7, 4))
plt.plot(t_eval, mass_mixed_2d, 'b-', linewidth=2, label='Mixed Residual PINN')
plt.plot(t_eval, mass_std_2d, 'g--', linewidth=2, label='Standard PINN')
plt.axhline(y=0.0, color='r', linestyle=':', linewidth=1.5, label='Theoretical (Zero)')
plt.title('2D Mass Conservation')
plt.xlabel('Time $t$')
plt.ylabel(r'$\iint (u - \bar{u}_0)\,dxdy$')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('mass_conservation_2d_gauss.png', dpi=150, bbox_inches='tight')

# Loss curve
plt.figure(figsize=(8, 5))
plt.semilogy(loss_history_mixed, 'b-', linewidth=2, label='Mixed Residual PINN (Val Loss)')
plt.semilogy(loss_history_std, 'g--', linewidth=2, label='Standard PINN (Val Loss)')
plt.title('Validation Loss During Training (2D Cahn-Hilliard)')
plt.xlabel('Epoch'); plt.ylabel('Loss (log scale)')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('loss_comparison_2D_CH.png', dpi=150, bbox_inches='tight')


# L2 error
def compute_L2_error_2d(u1, u2, x, y):
    dx = x[1] - x[0]; dy = y[1] - y[0]
    return np.sqrt(np.sum((u1 - u2)**2) * dx * dy)

L2_error_final = compute_L2_error_2d(U_mixed_2d, U_std_2d, x_grid, y_grid)
print(f"\n📊 Final Time L2 Error (Std vs Mixed): {L2_error_final:.3e}")

# Save metrics
with open('evaluation_metrics_2D_CH.txt', 'w') as f:
    f.write("实验1（2D）：Cahn-Hilliard方程 —— 混合残量模型 vs 标准PINN\n")
    f.write("="*60 + "\n")
    f.write(f"Final Time L2 Error (Std vs Mixed) = {L2_error_final:.3e}\n")
    params_mixed = sum(p.numel() for p in net_u_mixed.parameters()) + sum(p.numel() for p in net_mu_mixed.parameters())
    params_std = sum(p.numel() for p in net_u_std.parameters())
    f.write(f"  Mixed Residual PINN: {params_mixed:,} (u + μ networks)\n")
    f.write(f"  Standard PINN:       {params_std:,}\n")
    f.write(f"\nTraining Time:\n")
    f.write(f"  Mixed Residual PINN: {time_mixed:.2f} s\n")
    f.write(f"  Standard PINN:       {time_std:.2f} s\n")
    f.write(f"\nHyperparameter:\n")
    f.write(f"  λ_bc = {lambda_bc}, λ_ic = {lambda_ic}\n")
    f.write(f"  ε = {eps}\n")
    f.write(f"  Domain: [0,1]×[0,1], T = {T_final}\n")

print("✅ 所有实验完成！结果已保存至当前目录。")