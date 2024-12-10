import numpy as np
import argparse
import torch as th
import torch
import random
from torch import nn
import wandb
from torch.distributions import kl_divergence
from .module import Reward_Model

class VAE(nn.Module):
    def __init__(self, input_dim, inter_dim=256, latent_dim=4):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, latent_dim * 2),
        )

        self.decoder =  nn.Sequential(
            nn.Linear(latent_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, input_dim),
        )

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        raw_shape = x.shape
        x = x.reshape(-1, raw_shape[-1])

        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterise(mu, logvar)
        recon_x = self.decoder(z).reshape(raw_shape)

        return recon_x, mu, logvar, z
    
    def loss(self, x, recon_x, mu, logvar, kl_weight):
        x = x.reshape(-1, x.shape[-1])
        recon_x = recon_x.reshape(-1, recon_x.shape[-1])
        BCE = torch.nn.functional.mse_loss(recon_x, x)
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        return BCE + KLD * kl_weight, BCE, KLD


class IBRewardDecomposer(nn.Module):
    def __init__(self, args):
        super(IBRewardDecomposer, self).__init__()
        self.args = args
        self.K = self.args.rrd_k
        self.latent_dim = args.IB_latent_dim
        self.vae = VAE(input_dim=self.args.obs_dim*2+self.args.action_dim, latent_dim=self.latent_dim)
        self.reward_model = Reward_Model(input_dim=self.latent_dim)
        self.vae.to('cuda')
        self.reward_model.to('cuda')
        self.optimizer = torch.optim.Adam(list(self.reward_model.parameters())+list(self.vae.parameters()), lr=3e-4)
        self.loss_fn = nn.MSELoss(reduction='mean')


    def forward(self, states, actions, next_states, kl_loss =False):
        states = torch.cat([states, actions, states-next_states], dim=-1)
        recon_s, mu, logvar, z = self.vae(states)
        z = z.reshape(list(states.shape[:-1]) + [self.latent_dim])
        rewards = self.reward_model(z)
        if kl_loss:
            vae_loss, recon_loss, kl_loss = self.vae.loss(states, recon_s, mu, logvar, self.args.IB_kl_weight)
            return rewards, kl_loss
        return rewards

    
    def update(self, states, actions, next_states, episode_return, episode_length):
        self.optimizer.zero_grad()
        rewards, kl_loss = self.forward(states, actions, next_states, kl_loss=True)
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
        loss += self.args.IB_kl_weight * kl_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()
