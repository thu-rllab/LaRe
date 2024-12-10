import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import wandb
from torch.distributions import Beta, Normal
from .module import Reward_Model

class RRDRewardDecomposer(object):
    def __init__(self, args):
        super(RRDRewardDecomposer, self).__init__()
        self.args = args
        self.K = self.args.rrd_k

        self.reward_model = Reward_Model(input_dim=self.args.obs_dim*2 + self.args.action_dim)
        self.reward_model.to('cuda')

        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=3e-4)
        self.loss_fn = nn.MSELoss(reduction='mean')


    def forward(self, states, actions, next_states):
        states = torch.cat([states, actions, states-next_states], dim=-1)

        rewards = self.reward_model(states)#bs,t,-1 -> bs,t,1

        return rewards

    
    def update(self, states, actions, next_states, episode_return, episode_length):
        self.optimizer.zero_grad()

        rewards = self.forward(states, actions, next_states)
        sampled_rewards = []
        var_coef = []
        for i in range(rewards.shape[0]):
            local_episode_length = int(episode_length[i].item())
            sampled_steps = np.random.choice(local_episode_length, self.K, replace=self.K>local_episode_length)
            sampled_rewards.append(rewards[i,sampled_steps])

            var_coef.append(1.0-self.K/local_episode_length)

        sampled_rewards = torch.stack(sampled_rewards, dim=0)#bs,k,1

        sampled_rewards_var = torch.sum(torch.square(sampled_rewards - torch.mean(sampled_rewards, dim=1, keepdim=True)), dim=1)/(self.K-1)#bs,1
        sampled_rewards_var = torch.mean(sampled_rewards_var.squeeze() * torch.tensor(var_coef).to(sampled_rewards_var.device) / self.K)

        pred_returns = sampled_rewards.mean(dim=1).reshape(-1)
        episode_return = episode_return.reshape(-1)/episode_length.reshape(-1)
        loss = self.loss_fn(pred_returns, episode_return)
        if self.args.rrd_unbiased:
            loss = loss - sampled_rewards_var
        loss.backward()
        self.optimizer.step()
        return loss.item()
