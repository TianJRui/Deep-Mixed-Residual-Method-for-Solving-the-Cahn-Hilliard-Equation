# main.py
"""
CH方程的 PINN 求解器。
"""

import torch
import time
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
from config import *
from models import MLP
from data import generate_data, resample_interior_points, resample_boundary_points, resample_initial_points, resample_interior_points_adaptive
from losses import compute_standard_loss_CH,compute_drm_loss_CH
from utils import *
from initial_condition import compute_initial_condition as initial_condition_2d_ch
from optimizer import SALAdam  # 导入自定义优化器
# 设置随机种子和设备
set_seed(SEED)
print(f"Using device: {DEVICE}")

# 创建结果目录
RESULTS_DIR = "results_2d_CH"
os.makedirs(RESULTS_DIR, exist_ok=True)

if __name__ == "__main__":

    # 1.模型
    net_u_mixed = MLP(hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS, activation='tanh').to(DEVICE)
    net_mu_mixed = MLP(hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS, activation='tanh').to(DEVICE)
    
    optimizer_mixed = SALAdam(list(net_u_mixed.parameters()) + list(net_mu_mixed.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler_mixed = torch.optim.lr_scheduler.StepLR(optimizer_mixed, step_size=5000, gamma=0.999)

    # 2. 生成数据
    data = generate_data(N_f=800, N_b=800, N_i=1000, device=DEVICE,mode='reverse')
  
    # 3. 准备训练/验证数据
    x_f_train, y_f_train, t_f_train = data['x_f_train'], data['y_f_train'], data['t_f_train']
    x_b_train, y_b_train, t_b_train = data['x_b_train'], data['y_b_train'], data['t_b_train']
    x_i_train, y_i_train, t_i_train = data['x_i_train'], data['y_i_train'], data['t_i_train']
    ic_train = data['ic_train']
    
    x_f_val, y_f_val, t_f_val = data['x_f_val'], data['y_f_val'], data['t_f_val']
    x_b_val, y_b_val, t_b_val = data['x_b_val'], data['y_b_val'], data['t_b_val']
    x_i_val, y_i_val, t_i_val = data['x_i_val'], data['y_i_val'], data['t_i_val']
    ic_val = data['ic_val']


    def val_mixed_loss_fn():
        loss, _, _, _, _ = compute_drm_loss_CH(
                net_u_mixed, net_mu_mixed, x_f_val, y_f_val, t_f_val,
                x_b_val, y_b_val, t_b_val,
                x_i_val, y_i_val, t_i_val, ic_val,
                lambda_bc=LAMBDA_BC, lambda_ic=LAMBDA_IC,
            )
        return loss.item()
    
    # 4. 训练循环
    loss_history_mixed = []
    start_time = time.time()
    
    optimizer_mixed = SALAdam(list(net_u_mixed.parameters()) + list(net_mu_mixed.parameters()), lr=LEARNING_RATE_FINE_TUNING, weight_decay=1e-4)
    scheduler_mixed = torch.optim.lr_scheduler.StepLR(optimizer_mixed, step_size=5000, gamma=0.9)

    # 2. 生成数据
    data = generate_data(N_f=800, N_b=800, N_i=1000, device=DEVICE,mode='forward')
  
    # 3. 准备训练/验证数据
    x_f_train, y_f_train, t_f_train = data['x_f_train'], data['y_f_train'], data['t_f_train']
    x_b_train, y_b_train, t_b_train = data['x_b_train'], data['y_b_train'], data['t_b_train']
    x_i_train, y_i_train, t_i_train = data['x_i_train'], data['y_i_train'], data['t_i_train']
    ic_train = data['ic_train']
    
    x_f_val, y_f_val, t_f_val = data['x_f_val'], data['y_f_val'], data['t_f_val']
    x_b_val, y_b_val, t_b_val = data['x_b_val'], data['y_b_val'], data['t_b_val']
    x_i_val, y_i_val, t_i_val = data['x_i_val'], data['y_i_val'], data['t_i_val']
    ic_val = data['ic_val']

    
    # 4. 训练循环
    loss_history_mixed = []
    start_time = time.time()
    
    print("🔹 Starting training mixed PINN for log CH Equation...")
    for epoch in range(NUM_EPOCHS):

        if epoch % RESAMPLE_FREQ == 0:
            N_f_train = x_f_train.shape[0]
            N_b_train = x_b_train.shape[0]
            N_i_train = x_i_train.shape[0]
            x_f_train, y_f_train, t_f_train = resample_interior_points_adaptive(net_u_mixed,N_f_train, DEVICE,t_start=0.0, t_span=T_FINAL)
            x_b_train, y_b_train, t_b_train = resample_boundary_points(N_b_train, DEVICE,0.0, T_FINAL)
            x_i_train, y_i_train, t_i_train, ic_train = resample_initial_points(N_i_train, DEVICE)

        net_u_mixed.train()
        net_mu_mixed.train()
        optimizer_mixed.zero_grad()
            
        train_loss, loss_pde1, loss_pde2, loss_bc, loss_ic = compute_drm_loss_CH(
            net_u_mixed,net_mu_mixed,
            x_f_train, y_f_train, t_f_train,
            x_b_train, y_b_train, t_b_train,
            x_i_train, y_i_train, t_i_train,
            ic_train,
            lambda_bc=LAMBDA_BC,
            lambda_ic=LAMBDA_IC,
        )
        loss_history_mixed.append(train_loss.clone().to("cpu").item())
        

        train_loss.backward()
        optimizer_mixed.step()
        scheduler_mixed.step()

        if epoch % 100 == 0:
            net_u_mixed.eval()
            def val_loss_fn():
                loss, _, _, _,_ = compute_drm_loss_CH(
                net_u_mixed,net_mu_mixed,
                x_f_val, y_f_val, t_f_val,
                x_b_val, y_b_val, t_b_val,
                x_i_val, y_i_val, t_i_val, ic_val,
                lambda_bc=LAMBDA_BC, lambda_ic=LAMBDA_IC,
            )
                return loss.item()
            
            val_loss = val_loss_fn()

            
            print(f"mixed [Epoch {epoch}] Train Loss: {train_loss.item():.3e} "
                  f"(PDE1: {loss_pde1.item():.2e},PDE2: {loss_pde2.item():.2e}, BC: {loss_bc:.2e}, IC: {loss_ic.item():.2e}) | "
                  f"Val Loss: {val_loss:.3e}")
            if epoch % 1000 == 0:
                plot_2d_solution_snapshots(
                    model=net_u_mixed,
                    times=torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]) * T_FINAL,
                    nx=200, ny=200,
                    filename_prefix=os.path.join(RESULTS_DIR, f"mixed_log_CH_snapshot_epoch_{epoch}_forward")
                )


    mixed_training_time = time.time() - start_time
    print(f"✅mixed PINN method Training completed in {mixed_training_time:.2f} seconds.")


    # 1. 初始化模型
    net_u = MLP(
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        activation=ACTIVATION,
    ).to(DEVICE)
    
    optimizer = SALAdam(net_u.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)
    
    # 2. 生成数据
    data = generate_data(N_f=800, N_b=800, N_i=1000, device=DEVICE,mode='reverse')
  
    # 3. 准备训练/验证数据
    x_f_train, y_f_train, t_f_train = data['x_f_train'], data['y_f_train'], data['t_f_train']
    x_b_train, y_b_train, t_b_train = data['x_b_train'], data['y_b_train'], data['t_b_train']
    x_i_train, y_i_train, t_i_train = data['x_i_train'], data['y_i_train'], data['t_i_train']
    ic_train = data['ic_train']
    
    x_f_val, y_f_val, t_f_val = data['x_f_val'], data['y_f_val'], data['t_f_val']
    x_b_val, y_b_val, t_b_val = data['x_b_val'], data['y_b_val'], data['t_b_val']
    x_i_val, y_i_val, t_i_val = data['x_i_val'], data['y_i_val'], data['t_i_val']
    ic_val = data['ic_val']

    # 4. 训练循环
    loss_history_std = []
    start_time = time.time()

    optimizer = SALAdam(net_u.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)
    
    # 2. 生成数据
    data = generate_data(N_f=800, N_b=800, N_i=1000, device=DEVICE,mode='forward')
    
    # 3. 准备训练/验证数据
    x_f_train, y_f_train, t_f_train = data['x_f_train'], data['y_f_train'], data['t_f_train']
    x_b_train, y_b_train, t_b_train = data['x_b_train'], data['y_b_train'], data['t_b_train']
    x_i_train, y_i_train, t_i_train = data['x_i_train'], data['y_i_train'], data['t_i_train']
    ic_train = data['ic_train']
    
    x_f_val, y_f_val, t_f_val = data['x_f_val'], data['y_f_val'], data['t_f_val']
    x_b_val, y_b_val, t_b_val = data['x_b_val'], data['y_b_val'], data['t_b_val']
    x_i_val, y_i_val, t_i_val = data['x_i_val'], data['y_i_val'], data['t_i_val']
    ic_val = data['ic_val']

    # 4. 训练循环
    loss_history_std = []
    forward_start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):

        if epoch % RESAMPLE_FREQ == 0:
            N_f_train = x_f_train.shape[0]
            N_b_train = x_b_train.shape[0]
            N_i_train = x_i_train.shape[0]
            x_f_train, y_f_train, t_f_train = resample_interior_points_adaptive(net_u,N_f_train, DEVICE,t_start=0.0, t_span=T_FINAL)
            x_b_train, y_b_train, t_b_train = resample_boundary_points(N_b_train, DEVICE,t_start=0.0,t_span=T_FINAL)
            x_i_train, y_i_train, t_i_train, ic_train = resample_initial_points(N_i_train, DEVICE)

        net_u.train()
        optimizer.zero_grad()
            
        train_loss, loss_pde, loss_bc, loss_ic = compute_standard_loss_CH(
            net_u,
            x_f_train, y_f_train, t_f_train,
            x_b_train, y_b_train, t_b_train,
            x_i_train, y_i_train, t_i_train,
            ic_train,
            lambda_bc=LAMBDA_BC,
            lambda_ic=LAMBDA_IC,
        )
 
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history_std.append(train_loss.clone().to("cpu").item())
            
        if epoch % 100 == 0:
            net_u.eval()
            def val_loss_fn():
                loss, _, _, _ = compute_standard_loss_CH(
                net_u, x_f_val, y_f_val, t_f_val,
                x_b_val, y_b_val, t_b_val,
                x_i_val, y_i_val, t_i_val, ic_val,
                lambda_bc=LAMBDA_BC, lambda_ic=LAMBDA_IC,
            )
                return loss.item()
            val_loss = val_loss_fn()

            print(f"[Epoch {epoch}] Train Loss: {train_loss.item():.3e} "
                  f"(PDE: {loss_pde.item():.2e}, BC: {loss_bc:.2e}, IC: {loss_ic.item():.2e}) | "
                  f"Val Loss: {val_loss:.3e}")
            if epoch % 1000 == 0:
                plot_2d_solution_snapshots(
                    model=net_u,
                    times=torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]) * T_FINAL,
                    nx=200, ny=200,
                    filename_prefix=os.path.join(RESULTS_DIR, f"std_log_CH_snapshot_epoch_{epoch}_forward")
                )

    std_training_time = time.time() - forward_start_time
    
    print(f"✅std PINN method forward progress Training completed in {std_training_time:.2f} seconds.")

    print(f"✅std PINN method Training completed in {time.time() - start_time:.2f} seconds.")




    # 5. 最终可视化
    plot_2d_solution_snapshots(
        model=net_u_mixed,
        times=torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]) * T_FINAL,
        nx=200, ny=200,
        filename_prefix=os.path.join(RESULTS_DIR, "mixed_final_CH_solution")
    )
    # 5. 最终可视化
    plot_2d_solution_snapshots(
        model=net_u,
        times=torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]) * T_FINAL,
        nx=200, ny=200,
        filename_prefix=os.path.join(RESULTS_DIR, "std_final_CH_solution")
    )
    print(f"✅ Results saved to {RESULTS_DIR}")

    
    # ----------------------------
    # 6. 后续可视化与评估
    # ----------------------------
    L = DOMAIN_SIZE
    T_final = T_FINAL
    x_test_line = np.linspace(-L/2, L/2, 200)
    y_test_line = np.full_like(x_test_line, 0.5)
    t_test = np.array([0.0, 2.5, 5.0, 10.0])

    net_u_std=net_u
    u_mixed_line = []
    u_std_line = []
    net_u_mixed.eval()
    net_u_std.eval()

    for t_val in t_test:
        t_tensor = torch.full((len(x_test_line), 1), t_val, dtype=torch.float32).to(DEVICE)
        x_tensor = torch.tensor(x_test_line[:, None], dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y_test_line[:, None], dtype=torch.float32).to(DEVICE)

        # [修改] 应用 tanh 激活函数
        raw_mixed = net_u_mixed(x_tensor, y_tensor, t_tensor)
        u_pred_mixed = torch.tanh(raw_mixed).cpu().detach().numpy().flatten()
        
        raw_std = net_u_std(x_tensor, y_tensor, t_tensor)
        u_pred_std = torch.tanh(raw_std).cpu().detach().numpy().flatten()

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
    x_grid = np.linspace(-L/2, L/2, 100)
    y_grid = np.linspace(-L/2, L/2, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    t_flat = np.full_like(x_flat, t_plot)

    with torch.no_grad():
        x_tensor = torch.tensor(x_flat[:, None], dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y_flat[:, None], dtype=torch.float32).to(DEVICE)
        t_tensor = torch.tensor(t_flat[:, None], dtype=torch.float32).to(DEVICE)

        # [修改] 应用 tanh 激活函数
        u_mixed_2d = torch.tanh(net_u_mixed(x_tensor, y_tensor, t_tensor)).cpu().numpy().flatten()
        u_std_2d = torch.tanh(net_u_std(x_tensor, y_tensor, t_tensor)).cpu().numpy().flatten()

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

        
    t_eval = np.linspace(0, T_final, 101)

    # 计算 Mixed PINN 的总质量历史
    mass_total_mixed = []
    for t in t_eval:
        m = compute_total_mass(net_u_mixed, t)
        mass_total_mixed.append(m)
    mass_total_mixed = np.array(mass_total_mixed)

    # 计算 Standard PINN 的总质量历史 (如果需要对比)
    mass_total_std = []
    for t in t_eval:
        m = compute_total_mass(net_u_std, t)
        mass_total_std.append(m)
    mass_total_std = np.array(mass_total_std)

    # 初始总质量参考线
    ref_mass_mixed = mass_total_mixed[0]
    ref_mass_std = mass_total_std[0]

    plt.figure(figsize=(8, 5))

    # 绘制 Mixed PINN
    plt.plot(t_eval, mass_total_mixed, 'b-', linewidth=2, label='Mixed PINN (Total Mass)')
    plt.axhline(y=ref_mass_mixed, color='b', linestyle=':', alpha=0.5)

    # 绘制 Standard PINN
    plt.plot(t_eval, mass_total_std, 'g--', linewidth=2, label='Standard PINN (Total Mass)')
    plt.axhline(y=ref_mass_std, color='g', linestyle=':', alpha=0.5)

    plt.title('Total Mass Conservation ($M_{bulk} + M_{boundary}$)')
    plt.xlabel('Time $t$')
    plt.ylabel('Total Mass')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'total_mass_conservation.png'), dpi=150, bbox_inches='tight')
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
        f.write(f" Mixed Residual PINN: {mixed_training_time:.2f} s\n")
        f.write(f" Standard PINN: {std_training_time:.2f} s\n")
        f.write(f"\nHyperparameter:\n")
        f.write(f" Domain: [0,1]×[0,1], T = {T_final}\n")

    print("✅ 所有实验完成！结果已保存至目录:", RESULTS_DIR)