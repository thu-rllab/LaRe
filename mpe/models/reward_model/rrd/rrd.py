import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import wandb
from torch.distributions import Beta, Normal
class Reward_Model(nn.Module):
    def __init__(self, input_dim, output_dim=1, n_layers=5, hidden_dim=64,  device='cuda'):
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

class RRDRewardDecomposer(nn.Module):
    def __init__(self, args):
        super(RRDRewardDecomposer, self).__init__()
        self.args = args
        self.n_agents = self.args.n_agents
        self.K = self.args.rrd_k

        self.reward_model = Reward_Model(input_dim=self.args.obs_dim, n_layers=self.args.factor_reward_model_layers)
        self.reward_model.to('cuda')


    def forward(self, states, actions, cumulative_return,next_states=None, return_tensor_scores=False, only_inference=True):
        if self.args.use_next_state:
            states = next_states
        b,na,t,d = states.shape
        

        tensor_scores = torch.tensor(states).float().to(self.reward_model.device)
        rewards = self.reward_model(tensor_scores)#bs,na,t,-1 -> bs,na,t,1
        rewards = rewards.reshape(b,na,t,-1)
        
        if return_tensor_scores:
            return rewards, tensor_scores
        if only_inference:#only inference
            return rewards
        
        assert self.K<=t
        sampled_steps = np.random.choice(t, self.K, replace=False)
        no_sampled_steps = np.delete(np.arange(t), sampled_steps)
        rewards[:,:,no_sampled_steps,:] = 0
        rewards[:,:,sampled_steps,:] *= t/self.K

        sampled_rewards = rewards[:,:,sampled_steps,:].reshape(b,na,-1)#bs,na,K

        sampled_rewards_var = torch.sum(torch.square(sampled_rewards - torch.mean(sampled_rewards, dim=-1, keepdim=True)), dim=-1)/(self.K-1)#bs,na,1
        sampled_rewards_var = torch.mean(sampled_rewards_var.squeeze() * torch.tensor(1.0-self.K/t).to(sampled_rewards_var.device) / self.K)


        return rewards, sampled_rewards_var
