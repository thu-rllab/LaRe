import numpy as np
import argparse
import torch as th
import torch
import random
from torch import nn
import wandb
from torch.distributions import kl_divergence

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

class VAE(nn.Module):
    def __init__(self, input_dim, inter_dim=256, latent_dim=4):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, inter_dim),
            nn.ReLU(),
            # nn.Linear(inter_dim, inter_dim),
            # nn.ReLU(),
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


class VAERewardDecomposer(nn.Module):
    def __init__(self, args):
        super(VAERewardDecomposer, self).__init__()
        self.args = args
        self.n_agents = self.args.n_agents
        self.latent_dim = args.IB_latent_dim

        # self.latent_dim = 4
        self.vae = VAE(input_dim=self.args.obs_dim, latent_dim=self.latent_dim)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.args.lr_reward)

        self.reward_model = Reward_Model(input_dim=self.latent_dim, n_layers=self.args.factor_reward_model_layers)
        self.vae.to('cuda')
        self.reward_model.to('cuda')


    def forward(self, states, actions, cumulative_return,next_states=None, return_tensor_scores=False, return_kl=False):
        if self.args.use_next_state:
            states = next_states
        b,na,t,d = states.shape
        assert na == self.n_agents

        recon_s, mu, logvar, z = self.vae(states)
        vae_loss, recon_loss, kl_loss = self.vae.loss(states, recon_s, mu, logvar, self.args.kl_weight)
        wandb.log({'vae_loss': vae_loss.item(), 'recon_loss': recon_loss.item(), 'kl_loss': kl_loss.item(), 'mu': mu.mean().item(), 'logvar': logvar.mean().item()})

        tensor_scores = z.reshape(b*na,t,-1)#.detach()
        rewards = self.reward_model(tensor_scores)#bs,na,t,-1 -> bs,na,t,1
        rewards = rewards.reshape(b,na,t,-1)
        tensor_scores = tensor_scores.reshape(b,na,t,-1)
        if return_kl:
            return rewards, kl_loss
        if return_tensor_scores:
            return rewards, tensor_scores
        return rewards
