# models.py
import torch
import torch.nn as nn
from initial_condition import *
class MLP(nn.Module):

    """
    简单的多层感知机 (MLP)，用于物理信息神经网络 (PINN)。
    输入: (x, y, t) -> 输出: scalar (e.g., u or mu)
    """

    def __init__(self, input_dim=3, output_dim=1, hidden_dim=512, num_hidden_layers=4, activation='tanh'):
        """
        初始化 MLP。

        Args:
            input_dim (int): 输入维度，默认为 3 (x, y, t)。
            output_dim (int): 输出维度，默认为 1。
            hidden_dim (int): 隐藏层神经元数量。
            num_hidden_layers (int): 隐藏层的数量（不包括输入和输出层）。
            activation (str): 激活函数，支持 'tanh', 'relu', 'sigmoid'。
        """
        super(MLP, self).__init__()

        # 设置激活函数
        if activation == 'tanh':
            self.activation = nn.Tanh()#收敛到1e3
        elif activation == 'relu':
            self.activation = nn.ReLU()#收敛到1e1
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()#收敛到1e0
        elif activation == 'selu':
            self.activation = nn.SELU()##收敛到2e3
        elif activation == 'gelu':
            self.activation = nn.GELU()#
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'prelu':
            # PReLU 需要参数，这里用标量初始化（可学习）
            self.activation = nn.PReLU()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'swish' or activation == 'silu':
            self.activation = nn.SiLU()  # Swish = SiLU
        elif activation == 'mish':
            from torch.nn import Mish  # PyTorch >= 1.9 支持
            self.activation = Mish()
        elif activation == 'hardswish':
            self.activation = nn.Hardswish()
        elif activation == 'relu6':
            self.activation = nn.ReLU6()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        # 不收敛的原因很大程度上是因为在损失函数计算过程中，由于对数非线性项的存在，导致了损失函数对神经网络参数的变化过于敏感，只能依靠极小的学习率来微调参数，导致训练过程非常缓慢。
        # PDE的解由于初值条件的极端性，导致损失函数对初值条件的微小变化非常敏感，这会导致训练过程非常不稳定，从而使得模型无法收敛。
        #初值中+1相和-1相的占比过大，而过渡区域占比过小，导致模型在训练过程中难以学习到过渡区域的特征，从而使得模型在过渡区域的表现不佳，其表现为任意时刻神经网络输出的值都和初值保持相同。

        # 构建网络层
        layers = []
        in_features = input_dim

        # 添加隐藏层
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(self.activation)
            in_features = hidden_dim

        # 添加输出层（无激活函数）
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)


    def forward(self, x, y, t):
        """
        前向传播。

        Args:
            x, y, t (torch.Tensor): 形状为 (N, 1) 的张量。

        Returns:
            torch.Tensor: 形状为 (N, 1) 的输出。
        """
        # 将输入拼接为 (N, 3)
        input_tensor = torch.cat([x, y, t], dim=1)  # shape: (N, 3)
        output = self.net(input_tensor)
        return output
    
class MLP_hardcording_init(nn.Module):

    """
    简单的多层感知机 (MLP)，用于物理信息神经网络 (PINN)。
    输入: (x, y, t) -> 输出: scalar (e.g., u or mu)
    """

    def __init__(self, input_dim=3, output_dim=1, hidden_dim=512, num_hidden_layers=4, activation='tanh',init_condition=compute_initial_condition):
        """
        初始化 MLP。

        Args:
            input_dim (int): 输入维度，默认为 3 (x, y, t)。
            output_dim (int): 输出维度，默认为 1。
            hidden_dim (int): 隐藏层神经元数量。
            num_hidden_layers (int): 隐藏层的数量（不包括输入和输出层）。
            activation (str): 激活函数，支持 'tanh', 'relu', 'sigmoid'。
        """
        super(MLP_hardcording_init, self).__init__()

        # 设置激活函数
        if activation == 'tanh':
            self.activation = nn.Tanh()#收敛到1e3
        elif activation == 'relu':
            self.activation = nn.ReLU()#收敛到1e1
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()#收敛到1e0
        elif activation == 'selu':
            self.activation = nn.SELU()##收敛到2e3
        elif activation == 'gelu':
            self.activation = nn.GELU()#
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'prelu':
            # PReLU 需要参数，这里用标量初始化（可学习）
            self.activation = nn.PReLU()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'swish' or activation == 'silu':
            self.activation = nn.SiLU()  # Swish = SiLU
        elif activation == 'mish':
            from torch.nn import Mish  # PyTorch >= 1.9 支持
            self.activation = Mish()
        elif activation == 'hardswish':
            self.activation = nn.Hardswish()
        elif activation == 'relu6':
            self.activation = nn.ReLU6()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        # 不收敛的原因很大程度上是因为在损失函数计算过程中，由于对数非线性项的存在，导致了损失函数对神经网络参数的变化过于敏感，只能依靠极小的学习率来微调参数，导致训练过程非常缓慢。
        # PDE的解由于初值条件的极端性，导致损失函数对初值条件的微小变化非常敏感，这会导致训练过程非常不稳定，从而使得模型无法收敛。
        #初值中+1相和-1相的占比过大，而过渡区域占比过小，导致模型在训练过程中难以学习到过渡区域的特征，从而使得模型在过渡区域的表现不佳，其表现为任意时刻神经网络输出的值都和初值保持相同。
        self.init_condition = init_condition
        # 构建网络层
        layers = []
        in_features = input_dim

        # 添加隐藏层
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(self.activation)
            in_features = hidden_dim

        # 添加输出层（无激活函数）
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)


    def forward(self, x, y, t):
        """
        前向传播。

        Args:
            x, y, t (torch.Tensor): 形状为 (N, 1) 的张量。

        Returns:
            torch.Tensor: 形状为 (N, 1) 的输出。
        """
        # 将输入拼接为 (N, 3)
        input_tensor = torch.cat([x, y, t], dim=1)  # shape: (N, 3)
        output = self.net(input_tensor) * t + self.init_condition(x,y)
        return output

def compute_gradient_norm(model, norm_type=2):
    """
    计算模型所有参数的梯度范数
    :param model: 目标模型
    :param norm_type: 范数类型，默认L2范数（2），可选1（L1）、np.inf（无穷范数）
    :return: 总梯度范数、平均梯度范数、最大梯度范数
    """
    total_norm = 0.0
    grad_norms = []
    
    for param in model.parameters():
        if param.grad is not None:
            # 计算单个参数的梯度范数
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
            grad_norms.append(param_norm.item())
    
    # 计算总范数（L2范数需开平方）
    total_norm = total_norm ** (1. / norm_type)
    # 计算平均范数和最大范数（便于分析）
    avg_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    max_norm = max(grad_norms) if grad_norms else 0.0
    
    return total_norm, avg_norm, max_norm