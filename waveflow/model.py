import torch.nn as nn
import torch


def mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class MLPVectorField(nn.Module):
    def __init__(self, hidden=[128, 128, 128, 128, 128, 128]):
        super().__init__()
        self.net = mlp([4] + hidden + [1])  # 4-D --> 1-D

    def forward(self, x_t, coords, t_path):
        return self.net(torch.cat([x_t, coords, t_path], dim=-1))
