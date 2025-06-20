import torch
from .config import DEVICE
from .model import MLPVectorField

class ODE(torch.nn.Module):
    def drift_coefficient(self, u, t): ...

class LearnedODE(ODE):
    def __init__(self, net, coords):
        super().__init__(); self.net, self.coords = net, coords
    def drift_coefficient(self, u, t):
        return self.net(u, self.coords, t)

class EulerSim:
    def __init__(self, ode): self.ode = ode
    def step(self, u, t, h): return u + self.ode.drift_coefficient(u,t)*h
    @torch.no_grad()
    def simulate(self, u, ts):
        for k in range(ts.shape[1]-1):
            t = ts[:,k].unsqueeze(-1)
            h = (ts[:,k+1]-ts[:,k]).unsqueeze(-1)
            u = self.step(u, t, h)
        return u

def reconstruct(pt_file, model_file, N=401, steps=4_000):
    d   = torch.load(pt_file, map_location=DEVICE)
    xg  = torch.linspace(0, d["x"][-1], N, device=DEVICE)
    tg  = torch.linspace(0, d["t"][-1], N, device=DEVICE)
    xx, tt = torch.meshgrid(xg, tg, indexing="ij")
    coords = torch.stack([xx,tt], dim=-1).reshape(-1,2).float()
    M      = coords.size(0)
    net = MLPVectorField().to(DEVICE)
    net.load_state_dict(torch.load(model_file, map_location=DEVICE))
    ode   = LearnedODE(net.eval(), coords)
    solver= EulerSim(ode)
    u0 = torch.randn(M,1, device=DEVICE)
    ts = torch.linspace(0.0,1.0, steps+1, device=DEVICE).expand(M,-1)
    u1 = solver.simulate(u0, ts)
    return u1.view(N,N).cpu(), d["u"]
