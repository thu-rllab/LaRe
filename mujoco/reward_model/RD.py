import numpy as np
import argparse
import torch as th
import torch
import random
from torch import nn
import wandb
from torch.distributions import kl_divergence
from .module import Reward_Model

class RDRewardDecomposer(nn.Module):
    def __init__(self, args):
        super(RDRewardDecomposer, self).__init__()
        self.args = args

        self.reward_model = Reward_Model(input_dim=self.args.obs_dim*2 + self.args.action_dim)
        self.reward_model.to('cuda')
        self.optimizer = torch.optim.Adam(list(self.reward_model.parameters()), lr=3e-4)
        self.loss_fn = nn.MSELoss(reduction='mean')


    def forward(self, states, actions, next_states):
        states = torch.cat([states, actions, states-next_states], dim=-1)
        rewards = self.reward_model(states)#bs,t,-1 -> bs,t,1
        return rewards

    
    def update(self, states,actions, next_states, episode_return, episode_length):
        self.optimizer.zero_grad()
        rewards = self.forward(states, actions,  next_states)
        for i in range(rewards.shape[0]):
            rewards[i,int(episode_length[i].item()):] = 0
        pred_returns = rewards.sum(dim=1).reshape(-1)/episode_length.reshape(-1)
        episode_return = episode_return.reshape(-1)/episode_length.reshape(-1)
        loss = self.loss_fn(pred_returns, episode_return)
        loss.backward()
        self.optimizer.step()
        return loss.item()
