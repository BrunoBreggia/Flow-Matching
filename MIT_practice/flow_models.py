from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import torch
from tqdm import tqdm

from flow_path import  GaussianCondPathZT
from flow_samples import WaveConditionalSampleable, Alpha, Beta
from flow_ODE import ODE, SDE
from flow_VF import LearnedVectorFieldODE

def build_mlp(dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU):
    mlp = []
    for idx in range(len(dims) - 1):
        mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            mlp.append(activation())
    return torch.nn.Sequential(*mlp)



class MLPVectorField(torch.nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """

    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - u_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


class MLPScore(torch.nn.Module):
    """
    MLP-parameterization of the learned score field
    """

    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - s_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


class MLPVectorFieldCond(torch.nn.Module):
    def __init__(self, dim_u=1, dim_z=2, h=[64, 64, 64]):
        super().__init__()
        self.net = build_mlp([dim_u + dim_z + 1] + h + [dim_u])

    def forward(self, u, z, t):
        inp = torch.cat([u, z, t], dim=-1)
        return self.net(inp)
    
    def train_one_k(pt_file, device="cuda"):
    # 1. datos y camino
    ds = WaveConditionalSampleable(pt_file, device)
    alpha, beta = Alpha(), Beta()
    path = GaussianCondPathZT(ds, alpha, beta)

    # 2. red y ODE
    net = MLPVectorFieldCond().to(device)
    ode = LearnedVectorFieldODE(net)

    # 3. loop FM
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(15_000):
        z, u0 = ds.sample(4096)  # (bs,2), (bs,1)
        t = torch.rand_like(u0)  # (bs,1)
        u_t = alpha(t) * u0 + beta(t) * torch.randn_like(u0)
        target = path.conditional_vector_field(u_t, z, t)
        pred = net(u_t, z, t)
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(net.state_dict(), pt_file.with_suffix(".fm.pt"))
