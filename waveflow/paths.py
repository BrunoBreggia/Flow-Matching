import torch
from abc import ABC, abstractmethod
from .schedules import alpha, beta
from .data import Sampleable

class ConditionalProbabilityPath(ABC, torch.nn.Module):
    def __init__(self, p_data: Sampleable):
        super().__init__(); self.p_data = p_data
    @abstractmethod
    def sample_conditioning_variable(self, n): ...
    @abstractmethod
    def sample_conditional_path(self, vars, t): ...
    @abstractmethod
    def conditional_vector_field(self, x, vars, t): ...

class WaveConditionalPath(ConditionalProbabilityPath):
    def sample_conditioning_variable(self, n):
        return self.p_data.sample(n)
    def sample_conditional_path(self, vars, t):
        u_star, _ = vars
        eps = torch.randn_like(u_star)
        return alpha(t)*u_star + beta(t)*eps
    def conditional_vector_field(self, x, vars, t):
        u_star, _ = vars
        A = alpha.dt(t) - (beta.dt(t)/beta(t))*alpha(t)
        B = beta.dt(t)/beta(t)
        return A*u_star + B*x
