
from torch import nn
import torch
from torch.nn import functional as F
class Reward_Model(nn.Module):
    def __init__(self, input_dim, output_dim=1, n_layers=3, hidden_dim=256,  device='cuda'):
        super().__init__()
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

def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

class RNN_Reward_Model(nn.Module):
    def __init__(self, input_dim, output_dim=2, n_layers=1, hidden_dim=256,  device='cuda'):
        super().__init__()
        self.rnn_layer = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.hidden_dim = hidden_dim
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.register_parameter('max_logvar', nn.Parameter(torch.ones(1) * 0.5, requires_grad=True))
        self.register_parameter('min_logvar', nn.Parameter(torch.ones(1) * -10, requires_grad=True))

        self.hidden = None
        self.device = device
        self.to(device)

    def init_hidden(self):
        self.hidden = None

    def forward(self, input):
        batch_size, num_timesteps, _ = input.shape

        rnn_output, self.hidden = self.rnn_layer(input, self.hidden)
        rnn_output = rnn_output.reshape(-1, self.hidden_dim)
        output = self.output_layer(rnn_output)
        output = output.view(batch_size, num_timesteps, -1)
        mean, logvar = torch.chunk(output, 2, dim=-1)

        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        std = torch.sqrt(torch.exp(logvar))
        output = mean + torch.normal(0, 1, size=mean.shape, device=input.device)*std
        return output.view(batch_size, num_timesteps)
    
    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)