# models.py
import torch
import torch.nn as nn
import numpy as np

class PINN(nn.Module):
    def __init__(self, hidden_dim=512, num_hidden_layers=5, activation='tanh', sigma=10.0, fourier_dim=64):
        super(PINN, self).__init__()
        self.fourier_dim = fourier_dim
        self.sigma = sigma
        self.register_buffer('B', torch.randn(2, fourier_dim) * sigma)
        input_dim = 2 * fourier_dim + 1
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
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        xy = torch.cat([x, y], dim=1)
        proj = torch.matmul(xy, self.B)#type: ignore
        fourier_feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        input_feat = torch.cat([fourier_feat, t], dim=1)
        return self.net(input_feat)