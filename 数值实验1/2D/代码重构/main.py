# main.py
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import os  # 新增：用于创建结果目录
from config import *
from models import PINN
from data import (
    generate_data,
    resample_interior_points,
    resample_boundary_points,
    resample_initial_points,
    generate_fixed_boundary_points
)
from losses import (
    compute_standard_loss_ch,
    create_mixed_loss_ch,
    compute_boundary_loss
)
from utils import (
    set_seed,
    plot_2d_solution_snapshots,
    compute_initial_mean_2d,
    mass_history_2d
)

# 设置随机种子和设备
set_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.init()
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# ----------------------------
# 创建结果保存主目录
# ----------------------------
RESULTS_DIR = "results_2d_ch"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------
# 超参数搜索（使用混合模型）
# ----------------------------
def search_best_lambda(data, lambda_combinations, num_epochs_search, device):
    keys = ['x_f', 'y_f', 't_f', 'x_b', 'y_b', 't_b', 'x_i', 'y_i', 't_i', 'ic']
    train_data = {k+'_train': data[k+'_train'] for k in keys}
    val_data = {k+'_val': data[k+'_val'] for k in keys}

    print(f"🔍 Starting hyperparameter search: {len(lambda_combinations)} combinations")
    lambda_performance = {}

    for idx, (lambda_bc, lambda_ic) in enumerate(lambda_combinations, 1):
        print(f"\n--- Progress: {idx}/{len(lambda_combinations)} | Current: (λ_bc={lambda_bc}, λ_ic={lambda_ic}) ---")

        net_u = PINN(hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS, activation='tanh').to(device)
        net_mu = PINN(hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS, activation='tanh').to(device)

        optimizer = torch.optim.Adam(list(net_u.parameters()) + list(net_mu.parameters()), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

        loss_fn = create_mixed_loss_ch(
            train_data['x_f_train'], train_data['y_f_train'], train_data['t_f_train'],
            train_data['x_b_train'], train_data['y_b_train'], train_data['t_b_train'],
            train_data['x_i_train'], train_data['y_i_train'], train_data['t_i_train'],
            train_data['ic_train'],
            lambda_bc, lambda_ic, EPS
        )

        val_loss_fn = create_mixed_loss_ch(
            val_data['x_f_val'], val_data['y_f_val'], val_data['t_f_val'],
            val_data['x_b_val'], val_data['y_b_val'], val_data['t_b_val'],
            val_data['x_i_val'], val_data['y_i_val'], val_data['t_i_val'],
            val_data['ic_val'],
            lambda_bc, lambda_ic, EPS
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
        print(f" Done: Final Val Loss={best_val_loss:.3e}")

    best_lambda = min(lambda_performance.items(), key=lambda x: x[1])[0]
    print(f"\n🎉 Best (λ_bc, λ_ic): {best_lambda}")
    return best_lambda, lambda_performance

# ----------------------------
# 主执行流程
# ----------------------------
if __name__ == "__main__":
    # 1. 生成初始数据
    data = generate_data(N_f=1500, N_b=1500, N_i=1500)

    # 2. 执行超参搜索
    #best_lambda, lambda_perf = search_best_lambda(data, LAMBDA_COMBINATIONS, NUM_EPOCHS_SEARCH, DEVICE)

    # 3. 可视化超参搜索结果 (Heatmap)
    # lambda_candidates = [1.0, 10.0, 100.0]
    # ic_vals = [1.0, 10.0, 100.0]
    # perf_matrix = np.zeros((len(lambda_candidates), len(ic_vals)))
    # for i, bc in enumerate(lambda_candidates):
    #     for j, ic in enumerate(ic_vals):
    #         perf_matrix[i, j] = lambda_perf.get((bc, ic), np.nan)

    # plt.figure(figsize=(8, 6))
    # import seaborn as sns
    # sns.heatmap(
    #     perf_matrix,
    #     xticklabels=[str(ic) for ic in ic_vals],
    #     yticklabels=[str(bc) for bc in lambda_candidates],
    #     annot=True, fmt='.1e', cmap='YlOrRd_r',
    #     cbar_kws={'label': 'Validation Loss'}
    # )
    # plt.xlabel('λ_ic')
    # plt.ylabel('λ_bc')
    # plt.title('Hyperparameter Search Result (2D Cahn-Hilliard)')
    # plt.tight_layout()
    # plt.savefig(os.path.join(RESULTS_DIR, 'lambda_search_heatmap_2D_CH.png'), dpi=150, bbox_inches='tight')
    # plt.close()

    # ----------------------------
    # 4. 主训练：Mixed Residual Model（三阶段）
    # ----------------------------
    best_lambda = (1.0, 100.0)
    lambda_bc, lambda_ic = best_lambda
    net_u_mixed = PINN(hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS, activation='tanh').to(DEVICE)
    net_mu_mixed = PINN(hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS, activation='tanh').to(DEVICE)

    optimizer_mixed = torch.optim.Adam(
        list(net_u_mixed.parameters()) + list(net_mu_mixed.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4
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

    # 验证损失函数
    val_loss_fn = create_mixed_loss_ch(
        x_f_val, y_f_val, t_f_val,
        x_b_val, y_b_val, t_b_val,
        x_i_val, y_i_val, t_i_val,
        ic_val,
        lambda_bc, lambda_ic, EPS
    )

    loss_history_mixed = []
    start_time_mixed = time.time()

    # ======================
    # 阶段1：仅训练初始条件
    # ======================
    print("🔹 Stage 1: Training Initial Condition Only")
    num_epochs_stage1 = 10000
    optimizer_mixed1 = torch.optim.Adam(list(net_u_mixed.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler_mixed1 = torch.optim.lr_scheduler.StepLR(optimizer_mixed1, step_size=5000, gamma=0.9)

    for epoch in range(num_epochs_stage1):
        if epoch % RESAMPLE_FREQ == 0:
            N_f_train = x_f_train.shape[0]
            N_i_train = x_i_train.shape[0]
            x_f_train, y_f_train, t_f_train = resample_interior_points(N_f_train, DEVICE)
            x_i_train, y_i_train, t_i_train, ic_train = resample_initial_points(N_i_train, DEVICE)

        net_u_mixed.train()
        optimizer_mixed1.zero_grad()
        u_i = net_u_mixed(x_i_train, y_i_train, t_i_train)
        ic_true = 0.05 * (torch.cos(4 * torch.pi * x_i_train) + torch.cos(4 * torch.pi * y_i_train))
        loss_ic = torch.mean((u_i - ic_true) ** 2) * lambda_ic
        loss = loss_ic
        loss.backward()
        optimizer_mixed1.step()
        scheduler_mixed1.step()

        net_u_mixed.eval(); net_mu_mixed.eval()
        val_loss, _, _, _ = val_loss_fn(net_u_mixed, net_mu_mixed)
        loss_history_mixed.append(val_loss.item())

        if epoch % 100 == 0:
            print(f'[Mixed Stage1] Epoch {epoch}, Train Loss (IC): {loss.item():.3e}, Val Loss: {val_loss.item():.3e}')

    plot_2d_solution_snapshots(
            model=net_u_mixed, times=[0.0], nx=100, ny=100,
            filename_prefix=os.path.join(RESULTS_DIR, "solution_2d_snapshots_t0")
        )

    # ======================
    # 阶段2：IC + PDE
    # ======================
    print("🔹 Stage 2: Training IC + PDE")
    num_epochs_stage2 = 40000

    for epoch in range(num_epochs_stage2):
        if epoch % RESAMPLE_FREQ == 0:
            N_f_train = x_f_train.shape[0]
            N_i_train = x_i_train.shape[0]
            x_f_train, y_f_train, t_f_train = resample_interior_points(N_f_train, DEVICE)
            x_i_train, y_i_train, t_i_train, ic_train = resample_initial_points(N_i_train, DEVICE)

        net_u_mixed.train(); net_mu_mixed.train()
        optimizer_mixed.zero_grad()

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
        mu_pred = -EPS**2 * lap_u + f_prime_u
        loss_pde_2 = torch.mean((mu - mu_pred) ** 2)

        # IC
        u_i = net_u_mixed(x_i_train, y_i_train, t_i_train)
        ic_true = 0.05 * (torch.cos(4 * np.pi * x_i_train) + torch.cos(4 * np.pi * y_i_train))
        loss_ic = torch.mean((u_i - ic_true) ** 2) * lambda_ic

        loss = loss_ic + loss_pde_1 + loss_pde_2

        torch.nn.utils.clip_grad_norm_(
            list(net_u_mixed.parameters()) + list(net_mu_mixed.parameters()), max_norm=1.0
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
            model=net_u_mixed, times=[0.0, 5.0, 10.0], nx=100, ny=100,
            filename_prefix=os.path.join(RESULTS_DIR, "solution_2d_snapshots_t0_t1.0")
        )

    # ======================
    # 阶段3：IC + PDE + BC
    # ======================
    N_b_train = x_b_train.shape[0]
    x_b_train, y_b_train, t_b_train = generate_fixed_boundary_points(N_b_train // 4)

    print("🔹 Stage 3: Training Full Loss (IC + PDE + BC)")
    num_epochs_stage3 = 40000

    for epoch in range(num_epochs_stage3):
        if epoch % RESAMPLE_FREQ == 0:
            N_f_train = x_f_train.shape[0]
            N_i_train = x_i_train.shape[0]
            x_f_train, y_f_train, t_f_train = resample_interior_points(N_f_train, DEVICE)
            x_i_train, y_i_train, t_i_train, ic_train = resample_initial_points(N_i_train, DEVICE)

        net_u_mixed.train(); net_mu_mixed.train()
        optimizer_mixed.zero_grad()

        torch.nn.utils.clip_grad_norm_(
            list(net_u_mixed.parameters()) + list(net_mu_mixed.parameters()), max_norm=1.0
        )

        train_loss_fn = create_mixed_loss_ch(
            x_f_train, y_f_train, t_f_train,
            x_b_train, y_b_train, t_b_train,
            x_i_train, y_i_train, t_i_train,
            ic_train,
            lambda_bc, lambda_ic, EPS
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

    plot_2d_solution_snapshots(
            model=net_u_mixed, times=[0.0, 5.0, 10.0], nx=100, ny=100,
            filename_prefix=os.path.join(RESULTS_DIR, "solution_2d_snapshots_with_boundary_t0_t1.0")
        )

    time_mixed = time.time() - start_time_mixed
    print(f"✅ Mixed Residual PINN (3-stage) trained in {time_mixed:.2f} sec")

    # ----------------------------
    # 5. 主训练：Standard PINN（三阶段）
    # ----------------------------
    net_u_std = PINN(hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS, activation='tanh').to(DEVICE)
    optimizer_std = torch.optim.Adam(net_u_std.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler_std = torch.optim.lr_scheduler.StepLR(optimizer_std, step_size=5000, gamma=0.9)

    # 初始数据
    x_f_train, y_f_train, t_f_train = data['x_f_train'], data['y_f_train'], data['t_f_train']
    x_b_train, y_b_train, t_b_train = data['x_b_train'], data['y_b_train'], data['t_b_train']
    x_i_train, y_i_train, t_i_train = data['x_i_train'], data['y_i_train'], data['t_i_train']
    ic_train = data['ic_train']

    def val_loss_std():
        loss, _, _, _ = compute_standard_loss_ch(
            net_u_std,
            x_f_val, y_f_val, t_f_val,
            x_b_val, y_b_val, t_b_val,
            x_i_val, y_i_val, t_i_val,
            ic_val,
            lambda_bc, lambda_ic, EPS
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
        u_i = net_u_std(x_i_train, y_i_train, t_i_train)
        ic_true = 0.05 * (torch.cos(4 * np.pi * x_i_train) + torch.cos(4 * np.pi * y_i_train))
        loss_ic = torch.mean((u_i - ic_true) ** 2) * lambda_ic
        loss = loss_ic
        loss.backward()
        optimizer_std.step()
        scheduler_std.step()

        net_u_std.eval()
        val_loss = val_loss_std()
        loss_history_std.append(val_loss)

        if epoch % 100 == 0:
            print(f'[Std Stage1] Epoch {epoch}, Train Loss (IC): {loss.item():.3e}, Val Loss: {val_loss:.3e}')

    plot_2d_solution_snapshots(
            model=net_u_std, times=[0.0], nx=100, ny=100,
            filename_prefix=os.path.join(RESULTS_DIR, "std_pinn_solution_2d_snapshots_t0")
        )

    # ======================
    # 阶段2：IC + PDE（无 BC）
    # ======================
    print("🔹 Stage 2: Training IC + PDE (Standard PINN)")
    num_epochs_stage2 = 40000

    for epoch in range(num_epochs_stage2):
        if epoch % RESAMPLE_FREQ == 0:
            N_f_train = x_f_train.shape[0]
            N_i_train = x_i_train.shape[0]
            x_f_train, y_f_train, t_f_train = resample_interior_points(N_f_train, DEVICE)
            x_i_train, y_i_train, t_i_train, ic_train = resample_initial_points(N_i_train, DEVICE)

        net_u_std.train()
        optimizer_std.zero_grad()

        x_f_train.requires_grad_(True)
        y_f_train.requires_grad_(True)
        t_f_train.requires_grad_(True)

        u = net_u_std(x_f_train, y_f_train, t_f_train)

        # 手动计算 PDE 残差
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

        pde_residual = u_t + EPS**2 * bi_lap_u - lap_nonlinear
        loss_pde = torch.mean(pde_residual ** 2)

        # IC
        u_i = net_u_std(x_i_train, y_i_train, t_i_train)
        ic_true = 0.05 * (torch.cos(4 * np.pi * x_i_train) + torch.cos(4 * np.pi * y_i_train))
        loss_ic = torch.mean((u_i - ic_true) ** 2) * lambda_ic

        loss = loss_ic + loss_pde
        loss.backward()
        optimizer_std.step()
        scheduler_std.step()

        net_u_std.eval()
        val_loss = val_loss_std()
        loss_history_std.append(val_loss)

        if epoch % 100 == 0:
            print(f'[Std Stage2] Epoch {epoch}, Train Loss: {loss.item():.3e}, Val Loss: {val_loss:.3e}')

    plot_2d_solution_snapshots(
            model=net_u_std, times=[0.0, 5.0, 10.0], nx=100, ny=100,
            filename_prefix=os.path.join(RESULTS_DIR, "std_pinn_solution_2d_snapshots_t0_t1")
        )

    # ======================
    # 阶段3：完整损失（IC + PDE + BC）
    # ======================
    print("🔹 Stage 3: Training Full Loss (Standard PINN)")
    num_epochs_stage3 = 40000

    for epoch in range(num_epochs_stage3):
        if epoch % RESAMPLE_FREQ == 0:
            N_f_train = x_f_train.shape[0]
            N_b_train = x_b_train.shape[0]
            N_i_train = x_i_train.shape[0]
            x_f_train, y_f_train, t_f_train = resample_interior_points(N_f_train, DEVICE)
            x_b_train, y_b_train, t_b_train = resample_boundary_points(N_b_train, DEVICE)
            x_i_train, y_i_train, t_i_train, ic_train = resample_initial_points(N_i_train, DEVICE)

        net_u_std.train()
        optimizer_std.zero_grad()

        loss, loss_pde, loss_bc, loss_ic = compute_standard_loss_ch(
            net_u_std,
            x_f_train, y_f_train, t_f_train,
            x_b_train, y_b_train, t_b_train,
            x_i_train, y_i_train, t_i_train,
            ic_train,
            lambda_bc, lambda_ic, EPS
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
        model=net_u_std, times=[0.0, 5.0, 10.0], nx=100, ny=100,
        filename_prefix=os.path.join(RESULTS_DIR, "std_pinn_solution_2d_snapshots_t0_t1_with_boundary")
    )

    # ----------------------------
    # 6. 后续可视化与评估
    # ----------------------------
    L = DOMAIN_SIZE
    T_final = T_FINAL
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
        t_tensor = torch.full((len(x_test_line), 1), t_val, dtype=torch.float32).to(DEVICE)
        x_tensor = torch.tensor(x_test_line[:, None], dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y_test_line[:, None], dtype=torch.float32).to(DEVICE)

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
    plt.savefig(os.path.join(RESULTS_DIR, 'solution_comparison_2D_CH_line.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Full 2D plot at t=0.0
    t_plot = 0.0
    x_grid = np.linspace(0, L, 100)
    y_grid = np.linspace(0, L, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    t_flat = np.full_like(x_flat, t_plot)

    with torch.no_grad():
        x_tensor = torch.tensor(x_flat[:, None], dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y_flat[:, None], dtype=torch.float32).to(DEVICE)
        t_tensor = torch.tensor(t_flat[:, None], dtype=torch.float32).to(DEVICE)

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
    plt.savefig(os.path.join(RESULTS_DIR, 'solution_comparison_2D_CH_contour.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Mass conservation
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
    plt.savefig(os.path.join(RESULTS_DIR, 'mass_conservation_2d_gauss.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.semilogy(loss_history_mixed, 'b-', linewidth=2, label='Mixed Residual PINN (Val Loss)')
    plt.semilogy(loss_history_std, 'g--', linewidth=2, label='Standard PINN (Val Loss)')
    plt.title('Validation Loss During Training (2D Cahn-Hilliard)')
    plt.xlabel('Epoch'); plt.ylabel('Loss (log scale)')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'loss_comparison_2D_CH.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # L2 error
    def compute_L2_error_2d(u1, u2, x, y):
        dx = x[1] - x[0]; dy = y[1] - y[0]
        return np.sqrt(np.sum((u1 - u2)**2) * dx * dy)

    L2_error_final = compute_L2_error_2d(U_mixed_2d, U_std_2d, x_grid, y_grid)
    print(f"\n📊 Final Time L2 Error (Std vs Mixed): {L2_error_final:.3e}")

    # Save metrics
    with open(os.path.join(RESULTS_DIR, 'evaluation_metrics_2D_CH.txt'), 'w') as f:
        f.write("实验1（2D）：Cahn-Hilliard方程 —— 混合残量模型 vs 标准PINN\n")
        f.write("="*60 + "\n")
        f.write(f"Final Time L2 Error (Std vs Mixed) = {L2_error_final:.3e}\n")
        params_mixed = sum(p.numel() for p in net_u_mixed.parameters()) + sum(p.numel() for p in net_mu_mixed.parameters())
        params_std = sum(p.numel() for p in net_u_std.parameters())
        f.write(f" Mixed Residual PINN: {params_mixed:,} (u + μ networks)\n")
        f.write(f" Standard PINN: {params_std:,}\n")
        f.write(f"\nTraining Time:\n")
        f.write(f" Mixed Residual PINN: {time_mixed:.2f} s\n")
        f.write(f" Standard PINN: {time_std:.2f} s\n")
        f.write(f"\nHyperparameter:\n")
        f.write(f" λ_bc = {lambda_bc}, λ_ic = {lambda_ic}\n")
        f.write(f" ε = {EPS}\n")
        f.write(f" Domain: [0,1]×[0,1], T = {T_final}\n")

    print("✅ 所有实验完成！结果已保存至目录:", RESULTS_DIR)