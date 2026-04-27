"""
实验1：适配正则非线性项与Neumann边界的深度混合残量模型 vs 标准PINN
根据毕业论文第三章内容进行修改和增强
"""
# 实验1.py - 带 Early Stopping 的混合残量 PINN 与标准 PINN 对比（时间域修改为0到10）

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# 1. 定义 PINN 网络
# ----------------------------
class PINN(nn.Module):
    def __init__(self, hidden_dim=50, num_hidden_layers=4, activation='tanh'):
        super(PINN, self).__init__()
        layers = []
        input_dim = 2  # (x, t)
        output_dim = 1
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        if activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'relu':
            layers.append(nn.ReLU())
        # 隐藏层
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
        return self.net(X)


# ----------------------------
# 2. EarlyStopping 类
# ----------------------------
class EarlyStopping:
    def __init__(self, patience=500, min_delta=1e-6, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.6e}")
        else:
            self.counter += 1
            if self.verbose and self.counter % 100 == 0:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# ----------------------------
# 3. 生成训练/验证数据
# ----------------------------
def generate_data(N_f=10000, N_b=200, N_i=200, eps=0.01):
    # PDE 内部点 - 时间域改为0到10
    x_f = np.random.uniform(0, 1, N_f).reshape(-1, 1)
    t_f = np.random.uniform(0, 10, N_f).reshape(-1, 1)  # 时间范围修改

    # 边界点 (x=0 和 x=1) - 时间域改为0到10
    x_b_0 = np.zeros((N_b // 2, 1))
    x_b_1 = np.ones((N_b // 2, 1))
    x_b = np.vstack([x_b_0, x_b_1])
    t_b = np.random.uniform(0, 10, N_b).reshape(-1, 1)  # 时间范围修改

    # 初始点 (t=0)
    x_i = np.random.uniform(0, 1, N_i).reshape(-1, 1)
    t_i = np.zeros((N_i, 1))
    ic = np.sin(np.pi * x_i)  # u(x,0) = sin(pi x)

    # 转为 Tensor
    x_f = torch.tensor(x_f, dtype=torch.float32).to(device)
    t_f = torch.tensor(t_f, dtype=torch.float32).to(device)
    x_b = torch.tensor(x_b, dtype=torch.float32).to(device)
    t_b = torch.tensor(t_b, dtype=torch.float32).to(device)
    x_i = torch.tensor(x_i, dtype=torch.float32).to(device)
    t_i = torch.tensor(t_i, dtype=torch.float32).to(device)
    ic = torch.tensor(ic, dtype=torch.float32).to(device)

    # 分割训练/验证（8:2）
    split_f = int(0.8 * N_f)
    split_b = int(0.8 * N_b)
    split_i = int(0.8 * N_i)

    data = {
        'x_f_train': x_f[:split_f], 't_f_train': t_f[:split_f],
        'x_f_val': x_f[split_f:], 't_f_val': t_f[split_f:],
        'x_b_train': x_b[:split_b], 't_b_train': t_b[:split_b],
        'x_b_val': x_b[split_b:], 't_b_val': t_b[split_b:],
        'x_i_train': x_i[:split_i], 't_i_train': t_i[:split_i],
        'x_i_val': x_i[split_i:], 't_i_val': t_i[split_i:],
        'ic_train': ic[:split_i], 'ic_val': ic[split_i:]
    }
    return data


# ----------------------------
# 4. 损失函数定义
# ----------------------------
def compute_standard_loss(net_u, x_f, t_f, x_b, t_b, x_i, t_i, ic, lambda_bc, lambda_ic, eps):
    # PDE 残差
    x_f.requires_grad_(True)
    t_f.requires_grad_(True)
    u = net_u(x_f, t_f)
    u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    pde_residual = u_t - eps * u_xx
    loss_pde = torch.mean(pde_residual ** 2)

    # 边界条件：u(0,t)=u(1,t)=0
    u_b = net_u(x_b, t_b)
    loss_bc = torch.mean(u_b ** 2)

    # 初始条件
    u_i = net_u(x_i, t_i)
    loss_ic = torch.mean((u_i - ic) ** 2)

    total_loss = loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
    return total_loss, loss_pde, loss_bc, loss_ic


def create_mixed_loss(x_f, t_f, x_b, t_b, x_i, t_i, ic, lambda_bc, lambda_ic, eps, u0_mean):
    def loss_fn(net_u, net_mu):
        x_f.requires_grad_(True)
        t_f.requires_grad_(True)
        u = net_u(x_f, t_f)
        mu = net_mu(x_f, t_f)
        u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        pde_residual = u_t - eps * (u_xx + mu)
        loss_pde = torch.mean(pde_residual ** 2)

        u_b = net_u(x_b, t_b)
        loss_bc = torch.mean(u_b ** 2)

        u_i = net_u(x_i, t_i)
        loss_ic = torch.mean((u_i - ic) ** 2)

        #  修正：直接约束 u 的总质量
        u_integral = torch.mean(u)  # approx ∫₀¹ u dx
        loss_mass = (u_integral - u0_mean) ** 2

        total_loss = loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic + 1.0 * loss_mass
        return total_loss, loss_pde, loss_bc, loss_ic
    return loss_fn


# ----------------------------
# 5. 参数设置与超参数（λ_bc, λ_ic）网格搜索
# ----------------------------
hidden_dim = 512
num_hidden_layers = 4
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs_full = 10000  # 主实验训练轮次
num_epochs_search = 5000  # 超参数搜索时的简化训练轮次（平衡效率与效果）
eps = 0.01
u0_mean = 2 / np.pi  # ∫₀¹ sin(πx) dx = 2/π

# 超参数搜索范围（λ_bc 和 λ_ic 的候选组合，网格搜索）
lambda_candidates = [0.1, 1.0, 10.0, 100.0]  # 单个权重的候选值
lambda_combinations = [(bc, ic) for bc in lambda_candidates for ic in lambda_candidates]  # 所有组合（4x4=16种）


# # ----------------------------
# # 5.1 超参数搜索函数（轻量版训练+验证）
# # ----------------------------
# def search_best_lambda(data, lambda_combinations, num_epochs_search, device):
#     """
#     网格搜索最优的 (lambda_bc, lambda_ic) 组合
#     :param data: 训练/验证数据集（generate_data 输出）
#     :param lambda_combinations: 所有待测试的 λ 组合列表
#     :param num_epochs_search: 每个组合的训练轮次（简化版）
#     :param device: 计算设备（CPU/GPU）
#     :return: best_lambda (最优组合), lambda_performance (所有组合的验证损失)
#     """
#     lambda_performance = {}  # 存储每个组合的最终验证损失
#     x_f_train, t_f_train = data['x_f_train'], data['t_f_train']
#     x_f_val, t_f_val = data['x_f_val'], data['t_f_val']
#     x_b_train, t_b_train = data['x_b_train'], data['t_b_train']
#     x_b_val, t_b_val = data['x_b_val'], data['t_b_val']
#     x_i_train, t_i_train = data['x_i_train'], data['t_i_train']
#     x_i_val, t_i_val = data['x_i_val'], data['t_i_val']
#     ic_train, ic_val = data['ic_train'], data['ic_val']

#     print(f"🔍 开始超参数搜索：共 {len(lambda_combinations)} 种 (λ_bc, λ_ic) 组合")
#     for idx, (lambda_bc, lambda_ic) in enumerate(lambda_combinations, 1):
#         print(f"\n--- 搜索进度：{idx}/{len(lambda_combinations)} | 当前组合：(λ_bc={lambda_bc}, λ_ic={lambda_ic}) ---")
        
#         # 初始化轻量版标准PINN（搜索阶段用标准模型即可，避免混合模型的双重计算）
#         net_search = PINN(hidden_dim=256, num_hidden_layers=3, activation='tanh').to(device)  # 简化网络（减少参数量）
#         optimizer = torch.optim.Adam(net_search.parameters(), lr=1e-3, weight_decay=1e-4)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
#         #early_stopping = EarlyStopping(patience=200, min_delta=1e-6, verbose=False)  # 简化早停
        
#         best_val_loss = float('inf')
#         for epoch in range(num_epochs_search):
#             # 训练阶段
#             net_search.train()
#             optimizer.zero_grad()
#             train_loss, _, _, _ = compute_standard_loss(
#                 net_search, x_f_train, t_f_train, x_b_train, t_b_train, x_i_train, t_i_train, ic_train,
#                 lambda_bc, lambda_ic, eps
#             )
#             train_loss.backward()
#             optimizer.step()
#             scheduler.step()
            
#             # 验证阶段
#             net_search.eval()
#             val_loss, _, _, _ = compute_standard_loss(
#                     net_search, x_f_val, t_f_val, x_b_val, t_b_val, x_i_val, t_i_val, ic_val,
#                     lambda_bc, lambda_ic, eps
#                 )
#             val_loss = val_loss.item()
            
#             # 更新当前组合的最优验证损失
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
            
#             # # 早停判断（避免无效训练）
#             # early_stopping(val_loss)
#             # if early_stopping.early_stop:
#             #     print(f"  早停触发：epoch={epoch}, 最优验证损失={best_val_loss:.3e}")
#             #     break
        
#         # 记录当前组合的性能
#         lambda_performance[(lambda_bc, lambda_ic)] = best_val_loss
#         print(f"  组合完成：最终验证损失={best_val_loss:.3e}")
    
#     # 筛选最优组合（验证损失最小的）
#     best_lambda = min(lambda_performance.items(), key=lambda x: x[1])[0]
#     best_loss = lambda_performance[best_lambda]
#     print(f"\n🎉 超参数搜索完成！")
#     print(f"   最优 (λ_bc, λ_ic) 组合：{best_lambda}")
#     print(f"   对应最优验证损失：{best_loss:.3e}")
#     print(f"   所有组合性能：{lambda_performance}")
    
#     return best_lambda, lambda_performance


# ----------------------------
# 5.2 生成数据 + 执行超参数搜索
# ----------------------------
# 生成训练/验证数据（与主实验共用一套数据，保证一致性）
data = generate_data(N_f=10000, N_b=200, N_i=200, eps=eps)

# # 执行超参数搜索（若已搜索过，可注释此句直接指定 best_lambda）
# best_lambda, lambda_perf = search_best_lambda(
#     data=data,
#     lambda_combinations=lambda_combinations,
#     num_epochs_search=num_epochs_search,
#     device=device
# )

# ----------------------------
# 可选：超参数搜索结果可视化（热力图）
# ----------------------------
# import seaborn as sns

# # 构建性能矩阵（行：λ_bc，列：λ_ic，值：验证损失）
# perf_matrix = np.zeros((len(lambda_candidates), len(lambda_candidates)))
# for i, bc in enumerate(lambda_candidates):
#     for j, ic in enumerate(lambda_candidates):
#         perf_matrix[i, j] = lambda_perf[(bc, ic)]

# 绘制热力图
# plt.figure(figsize=(8, 6))
# # 修正2：将float列表转换为字符串列表，解决类型不匹配问题
# lambda_labels = [str(l) for l in lambda_candidates]
# sns.heatmap(
#     perf_matrix,
#     xticklabels=lambda_labels,
#     yticklabels=lambda_labels,
#     annot=True,  # 显示数值
#     fmt='.1e',   # 数值格式（科学计数法）
#     cmap='YlOrRd_r',  # 颜色映射（浅色表示损失小）
#     cbar_kws={'label': 'Validation Loss'}
# )
# plt.xlabel('λ_ic (Initial Condition Weight)')
# plt.ylabel('λ_bc (Boundary Condition Weight)')
# plt.title('Hyperparameter Search Result: Validation Loss of (λ_bc, λ_ic) Combinations')
# plt.tight_layout()
# plt.savefig('lambda_search_heatmap.png', dpi=150, bbox_inches='tight')

data = generate_data(N_f=10000, N_b=200, N_i=200, eps=eps)
x_f_train, t_f_train = data['x_f_train'], data['t_f_train']
x_f_val, t_f_val = data['x_f_val'], data['t_f_val']
x_b_train, t_b_train = data['x_b_train'], data['t_b_train']
x_b_val, t_b_val = data['x_b_val'], data['t_b_val']
x_i_train, t_i_train = data['x_i_train'], data['t_i_train']
x_i_val, t_i_val = data['x_i_val'], data['t_i_val']
ic_train, ic_val = data['ic_train'], data['ic_val']


# ----------------------------
# 6. 训练混合残量模型（主模型）+ Early Stopping
# ----------------------------
best_lambda = (0.1,10.0)
lambda_bc, lambda_ic = best_lambda
net_u_mixed = PINN(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, activation='tanh').to(device)
net_mu_mixed = PINN(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, activation='relu').to(device)
optimizer_mixed = torch.optim.Adam(
    list(net_u_mixed.parameters()) + list(net_mu_mixed.parameters()),
    lr=learning_rate, weight_decay=weight_decay
)
scheduler_mixed = torch.optim.lr_scheduler.StepLR(optimizer_mixed, step_size=5000, gamma=0.5)

#early_stopping = EarlyStopping(patience=800, min_delta=1e-7, verbose=True)
train_loss_fn = create_mixed_loss(x_f_train, t_f_train, x_b_train, t_b_train, x_i_train, t_i_train, ic_train, lambda_bc, lambda_ic, eps, u0_mean)
val_loss_fn = create_mixed_loss(x_f_val, t_f_val, x_b_val, t_b_val, x_i_val, t_i_val, ic_val, lambda_bc, lambda_ic, eps, u0_mean)

loss_history_mixed = []
start_time_mixed = time.time()
epoch = 0
for epoch in range(num_epochs_full):
    net_u_mixed.train(); net_mu_mixed.train()
    optimizer_mixed.zero_grad()
    loss, loss_pde, loss_bc, loss_ic = train_loss_fn(net_u_mixed, net_mu_mixed)
    loss.backward()

    optimizer_mixed.step()
    scheduler_mixed.step()

    # Validation
    net_u_mixed.eval(); net_mu_mixed.eval()

    val_loss, _, _, _ = val_loss_fn(net_u_mixed, net_mu_mixed)
    val_loss = val_loss.item()

    loss_history_mixed.append(val_loss)

    if epoch % 2000 == 0:
        print(f'[Mixed] Epoch {epoch}, Train Loss: {loss.item():.3e}, Val Loss: {val_loss:.3e}')

    # early_stopping(val_loss)
    # if early_stopping.early_stop:
    #     print("🛑 Early stopping triggered for Mixed Residual PINN.")
    #     break

time_mixed = time.time() - start_time_mixed
print(f"✅ Mixed Residual PINN trained in {time_mixed:.2f} sec (stopped at epoch {epoch})")


# ----------------------------
# 7. 训练标准 PINN（对比模型）+ Early Stopping
# ----------------------------
net_u_std = PINN(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, activation='tanh').to(device)
optimizer_std = torch.optim.Adam(net_u_std.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler_std = torch.optim.lr_scheduler.StepLR(optimizer_std, step_size=5000, gamma=0.5)

#early_stopping_std = EarlyStopping(patience=800, min_delta=1e-7, verbose=True)

loss_history_std = []
start_time_std = time.time()

for epoch in range(num_epochs_full):
    net_u_std.train()
    optimizer_std.zero_grad()
    loss, loss_pde, loss_bc, loss_ic = compute_standard_loss(
        net_u_std, x_f_train, t_f_train, x_b_train, t_b_train, x_i_train, t_i_train, ic_train,
        lambda_bc, lambda_ic, eps
    )
    loss.backward()
    optimizer_std.step()
    scheduler_std.step()

    # Validation
    net_u_std.eval()
    val_loss, _, _, _ = compute_standard_loss(
            net_u_std, x_f_val, t_f_val, x_b_val, t_b_val, x_i_val, t_i_val, ic_val,
            lambda_bc, lambda_ic, eps
        )
    val_loss = val_loss.item()

    loss_history_std.append(val_loss)

    if epoch % 2000 == 0:
        print(f'[Standard] Epoch {epoch}, Train Loss: {loss.item():.3e}, Val Loss: {val_loss:.3e}')

    # early_stopping_std(val_loss)
    # if early_stopping_std.early_stop:
    #     print("🛑 Early stopping triggered for Standard PINN.")
    #     break

time_std = time.time() - start_time_std
print(f"✅ Standard PINN trained in {time_std:.2f} sec (stopped at epoch {epoch})")

# ----------------------------
# 8. 实验结果可视化与评估
# ----------------------------

# 定义空间-时间网格用于预测 - 时间域改为0到10
L = 1.0
T_final = 10.0  # 时间范围修改
x_test = np.linspace(0, L, 200)
t_test = np.array([0.0, 2.5, 5.0, 10.0])  # 四个时间点对比（调整为0-10范围内的点）

def initial_condition(x):
    return np.sin(np.pi * x)

# 预测解
u_mixed = []
u_std = []

net_u_mixed.eval()
net_u_std.eval()

with torch.no_grad():
    for t_val in t_test:
        t_tensor = torch.full((len(x_test), 1), t_val, dtype=torch.float32).to(device)
        x_tensor = torch.tensor(x_test[:, None], dtype=torch.float32).to(device)
        
        u_pred_mixed = net_u_mixed(x_tensor, t_tensor).cpu().numpy().flatten()
        u_pred_std = net_u_std(x_tensor, t_tensor).cpu().numpy().flatten()
        
        u_mixed.append(u_pred_mixed)
        u_std.append(u_pred_std)

u_mixed = np.array(u_mixed)
u_std = np.array(u_std)

# 解对比图
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    ax.plot(x_test, u_mixed[i, :], 'b-', linewidth=2, label='Mixed Residual PINN')
    ax.plot(x_test, u_std[i, :], 'g--', linewidth=2, label='Standard PINN')
    if i == 0:
        ax.plot(x_test, initial_condition(x_test), 'r:', linewidth=2, label='Initial Condition $u_0(x)$')
    ax.set_title(f'$t = {t_test[i]}$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('solution_comparison.png', dpi=150, bbox_inches='tight')


# 质量守恒分析（对去均值后的场）
import numpy as np
from scipy.special import  roots_legendre

def gauss_legendre_quadrature(f, n=10):
    """
    使用Gauss-Legendre求积公式计算f(x)在[0,1]上的积分
    :param f: 被积函数（输入x为数组，输出函数值数组）
    :param n: 高斯点数（n越大精度越高，通常n=10已足够）
    :return: 积分结果
    """
    # 获取n阶Legendre多项式的根和权重（区间[-1,1]）
    x_legendre, w_legendre = roots_legendre(n)
    # 转换到区间[0,1]
    x_scaled = (x_legendre + 1) / 2  # 从[-1,1]映射到[0,1]
    w_scaled = w_legendre / 2        # 权重按区间长度缩放
    # 计算积分
    return np.sum(w_scaled * f(x_scaled))
def mass_history(net, t_vals, u0_mean, gauss_n=20):
    """
    使用Gauss求积计算质量守恒偏差：∫(u(x,t) - u0_mean)dx 在[0,1]上的积分
    :param net: 神经网络模型
    :param t_vals: 时间点数组
    :param u0_mean: 初始质量均值（2/π）
    :param gauss_n: 高斯点数（推荐n≥10）
    :return: 每个时间点的质量偏差
    """
    masses = []
    # 获取Gauss-Legendre采样点（[0,1]区间）
    x_legendre, _ = roots_legendre(gauss_n)
    x_scaled = (x_legendre + 1) / 2  # 映射到[0,1]
    x_tensor = torch.tensor(x_scaled[:, None], dtype=torch.float32).to(device)
    
    for t in t_vals:
        t_tensor = torch.full_like(x_tensor, t)
        with torch.no_grad():
            u = net(x_tensor, t_tensor).cpu().numpy().flatten()
        # 定义被积函数：u(x,t) - u0_mean
        def integrand(x):
            # 由于x_scaled已固定，直接使用网络输出的u即可
            return u - u0_mean
        # 用Gauss求积计算积分（区间[0,1]）
        mass = gauss_legendre_quadrature(integrand, n=gauss_n)
        masses.append(mass)
    return (np.array(masses))
# 质量守恒分析（使用Gauss求积）- 时间范围改为0到10
t_eval = np.linspace(0, T_final, 101)  # 时间范围修改
# 无需传入x_eval，Gauss点内部生成
mass_mixed = mass_history(net_u_mixed, t_eval, u0_mean, gauss_n=20)  # 20点高斯求积
mass_std = mass_history(net_u_std, t_eval, u0_mean, gauss_n=20)

# 绘图代码不变
plt.figure(figsize=(7, 4))
plt.plot(t_eval, mass_mixed, 'b-', linewidth=2, label='Mixed Residual PINN')
plt.plot(t_eval, mass_std, 'g--', linewidth=2, label='Standard PINN')
plt.axhline(y=0.0, color='r', linestyle=':', linewidth=1.5, label='Theoretical (Zero)')
plt.title('Mass Conservation: |$\\langle u(x,t) - \\bar{u}_0 \\rangle$| (Gauss Quadrature)')
plt.xlabel('Time $t$')
plt.ylabel(r'$\int_0^1 (u(x,t) - \bar{u}_0)\,dx$')  # 注意这里积分符号没有除以L（因为L=1）
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('mass_conservation_gauss.png', dpi=150, bbox_inches='tight')
# 损失曲线（验证损失）- 无最后点标注
plt.figure(figsize=(8, 5))

# 1. 绘制混合残量模型损失曲线
plt.semilogy(
    loss_history_mixed, 
    'b-', 
    linewidth=2, 
    label='Mixed Residual PINN (Val Loss)'
)

# 2. 绘制标准PINN模型损失曲线
plt.semilogy(
    loss_history_std, 
    'g--', 
    linewidth=2, 
    label='Standard PINN (Val Loss)'
)

# 3. 图表基础配置
plt.title('Validation Loss During Training', fontsize=12)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Loss (log scale)', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存并显示图表
plt.savefig('loss_comparison.png', dpi=150, bbox_inches='tight')
# 评估指标
def compute_L2_error(u1, u2, x):
    dx = x[1] - x[0]
    return np.sqrt(np.sum((u1 - u2)**2) * dx)

print("\n📊 评估指标:")
L2_errors = []
for i, t_val in enumerate(t_test):
    err = compute_L2_error(u_std[i, :], u_mixed[i, :], x_test)
    L2_errors.append(err)
    print(f"  t = {t_val}: L2 Error (Std vs Mixed): {err:.3e}")

# 质量守恒最大偏差（绝对值）
mass_error_mixed = np.max(np.abs(mass_mixed))
mass_error_std = np.max(np.abs(mass_std))

# 参数量统计
params_mixed = sum(p.numel() for p in net_u_mixed.parameters()) + sum(p.numel() for p in net_mu_mixed.parameters())
params_std = sum(p.numel() for p in net_u_std.parameters())

# 保存评估结果
with open('evaluation_metrics.txt', 'w') as f:
    f.write("实验1：混合残量模型 vs 标准PINN —— 评估指标汇总\n")
    f.write("="*50 + "\n")
    for i, t_val in enumerate(t_test):
        f.write(f"t = {t_val}: L2 Error (Std vs Mixed) = {L2_errors[i]:.3e}\n")
    f.write(f"\n质量守恒最大偏差（去均值后）:\n")
    f.write(f"  Mixed Residual PINN: {mass_error_mixed:.3e}\n")
    f.write(f"  Standard PINN:       {mass_error_std:.3e}\n")
    f.write(f"\n模型参数量:\n")
    f.write(f"  Mixed Residual PINN: {params_mixed:,} (u + μ networks)\n")
    f.write(f"  Standard PINN:       {params_std:,}\n")
    f.write(f"\n训练时间:\n")
    f.write(f"  Mixed Residual PINN: {time_mixed:.2f} s\n")
    f.write(f"  Standard PINN:       {time_std:.2f} s\n")

print("\n✅ 实验完成。所有结果已保存至：")
print("   - solution_comparison.png")
print("   - mass_conservation_gauss.png")
print("   - loss_comparison.png")
print("   - evaluation_metrics.txt")