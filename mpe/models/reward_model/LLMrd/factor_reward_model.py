import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from torch.distributions import Beta, Normal
class Factor_Reward_Model(nn.Module):
    def __init__(self, input_dim, output_dim=1, n_layers=5, hidden_dim=64,  device='cuda'):
        super().__init__()
        self.n_layers = n_layers
        if n_layers == 1:
            self.model = nn.Linear(input_dim, output_dim)
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_layers-2)],
                nn.Linear(hidden_dim, output_dim)
            )
        self.device = device
        self.to(device)
        # self.apply(self.init_weights)

    def forward(self, x):
        return self.model(x)
    
    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
