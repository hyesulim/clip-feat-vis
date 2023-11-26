import torch
import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))