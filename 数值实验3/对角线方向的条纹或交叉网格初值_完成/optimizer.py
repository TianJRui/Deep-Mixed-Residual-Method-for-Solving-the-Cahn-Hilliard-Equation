# optimizer.py
"""
改进版 SAL-Adam: 全域优化版本 (DS-SAL-Adam)
支持参数跨越零点，实现全空间优化。

参数化形式：W = tanh(phi) * exp(theta)
- theta: 控制对数尺度 (log-scale)
- phi:  控制符号与归一化幅度 (via tanh in [-1, 1])
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Callable, Any


class SALAdam(Optimizer):
    def __init__(self, params: Any, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0):
        """
        Args:
            params: 模型参数
            lr: 学习率
            betas: (beta1, beta2) for Adam
            eps: 数值稳定性
            weight_decay: L2 正则
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not (0.0 <= betas[0] < 1.0) or not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta: {betas}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(SALAdam, self).__init__(params, defaults)

        # 初始化双参数：theta (log-scale), phi (sign-param)
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # 初始化 theta: log(|p| + eps)
                    state['theta'] = torch.log(torch.abs(p.data) + eps).clone()
                    # 初始化 phi: 使得 tanh(phi) ≈ W / exp(theta) = sign(W) * (|W| / exp(theta)) ≈ sign(W)
                    # 因为 exp(theta) ≈ |W| + eps, 所以 W / exp(theta) ≈ sign(W) * (|W| / (|W|+eps)) ≈ sign(W)
                    # 所以我们设 tanh(phi) = W / exp(theta)，反解 phi = atanh(W / exp(theta))
                    W_over_scale = p.data / (torch.exp(state['theta']) + eps)
                    # clamp to avoid numerical issues in atanh
                    W_over_scale = torch.clamp(W_over_scale, -1 + eps, 1 - eps)
                    state['phi'] = torch.atanh(W_over_scale)

                    # 动量：在 theta 和 phi 空间分别维护
                    state['exp_avg_theta'] = torch.zeros_like(state['theta'])
                    state['exp_avg_sq_theta'] = torch.zeros_like(state['theta'])
                    state['exp_avg_phi'] = torch.zeros_like(state['phi'])
                    state['exp_avg_sq_phi'] = torch.zeros_like(state['phi'])

                    state['step'] = 0

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:#type:ignore
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss_val = closure()
                if isinstance(loss_val, torch.Tensor):
                    loss = loss_val.item()
                else:
                    loss = loss_val

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue

                grad_W = p.grad  # dL/dW
                if grad_W.is_sparse:
                    raise RuntimeError("DS_SALAdam does not support sparse gradients")

                state = self.state[p]
                theta = state['theta']
                phi = state['phi']
                beta1, beta2 = group['betas']
                lr = group['lr']
                eps = group['eps']
                weight_decay = group['weight_decay']
                step = state['step'] = state.get('step', 0) + 1

                # 1. 重构 W
                scale = torch.exp(theta)
                sign_factor = torch.tanh(phi)
                W = scale * sign_factor

                # 2. 计算 dW/dtheta 和 dW/dphi
                # dW/dtheta = scale * sign_factor = W
                dW_dtheta = W
                # dW/dphi = scale * (1 - tanh^2(phi))
                dW_dphi = scale * (1 - sign_factor ** 2)

                # 3. 链式法则：dL/dtheta = dL/dW * dW/dtheta
                grad_theta = grad_W * dW_dtheta
                grad_phi = grad_W * dW_dphi

                # 4. 权重衰减：作用于原始 W
                if weight_decay != 0:
                    # d(L2)/dtheta = d(L2)/dW * dW/dtheta = (weight_decay * W) * W = weight_decay * W^2
                    grad_theta = grad_theta + weight_decay * (W ** 2)
                    # d(L2)/dphi = (weight_decay * W) * dW/dphi
                    grad_phi = grad_phi + weight_decay * W * dW_dphi

                # 5. 在 theta 和 phi 空间分别执行 Adam 更新
                self._adam_step(theta, grad_theta, state['exp_avg_theta'], state['exp_avg_sq_theta'],
                                beta1, beta2, lr, eps, step)
                self._adam_step(phi, grad_phi, state['exp_avg_phi'], state['exp_avg_sq_phi'],
                                beta1, beta2, lr, eps, step)

                # 6. 重构 W 并写回 p.data
                W_new = torch.exp(theta) * torch.tanh(phi)
                p.data.copy_(W_new)

        return loss

    def _adam_step(self, param: torch.Tensor, grad: torch.Tensor,
                   exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor,
                   beta1: float, beta2: float, lr: float, eps: float, step: int):
        """辅助函数：执行标准 Adam 更新（带偏差修正）"""
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)