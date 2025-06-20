import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 1_024
STEPS = 20_000
LR = 1e-3
EPS_T = 1e-3
