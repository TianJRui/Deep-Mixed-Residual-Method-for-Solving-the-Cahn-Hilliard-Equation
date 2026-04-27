# losses.py
import torch
import torch.nn as nn
from config import EPS

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
