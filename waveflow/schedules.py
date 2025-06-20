import torch
class LinearAlpha:
    def __call__(self, t):  return t
    def dt(self, t):        return torch.ones_like(t)
class SquareRootBeta:
    def __call__(self, t):  return torch.sqrt(1.0 - t)
    def dt(self, t):        return -0.5 / (torch.sqrt(1.0 - t) + 1e-8)
alpha, beta = LinearAlpha(), SquareRootBeta()
