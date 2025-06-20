from pathlib import Path
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from .config import DEVICE

class Sampleable(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...
    @abstractmethod
    def sample(self, n): ...

class WaveTensorDataset(Sampleable, Dataset):
    _dim = 3
    def __init__(self, pt_file: Path, device=DEVICE):
        d      = torch.load(pt_file, map_location=device)
        xx, tt = torch.meshgrid(d["x"], d["t"], indexing="ij")
        self.coords = torch.stack([xx, tt], dim=-1).reshape(-1, 2).float().to(device)
        self.u_star = d["u"].T.reshape(-1, 1).float().to(device)
    def __len__(self):               return self.u_star.size(0)
    def __getitem__(self, idx):      return self.u_star[idx], self.coords[idx]
    @property
    def dim(self):                   return self._dim
    def sample(self, n):
        idx = torch.randint(0, len(self), (n,), device=self.u_star.device)
        return self.u_star[idx], self.coords[idx]
