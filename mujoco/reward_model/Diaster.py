import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import wandb
from torch.distributions import Beta, Normal
from .module import Reward_Model, RNN_Reward_Model

class DiasterRewardDecomposer(object):
    def __init__(self, args):
        super(DiasterRewardDecomposer, self).__init__()
        self.args = args

        self.sub_reward_model = RNN_Reward_Model(input_dim=self.args.obs_dim + self.args.action_dim)
        self.sub_reward_model.to('cuda')

        self.reward_model = Reward_Model(input_dim=self.args.obs_dim + self.args.action_dim)
        self.reward_model.to('cuda')
        self.device = 'cuda'
        self.return_scale = 10 if not 'HumanoidStandup' in self.args.env else 500

        self.update_t = 0

        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=3e-4)
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.sub_reward_optim = torch.optim.Adam(self.sub_reward_model.parameters(), lr=3e-4)
        self.reward_model_optim = torch.optim.Adam(self.reward_model.parameters(), lr=3e-4)


    def forward(self, states, actions, next_states):
        states = torch.cat([states, actions], dim=-1)

        rewards = self.reward_model(states)#bs,t,-1 -> bs,t,1

        return rewards
    
    def learn_sub_reward_from(self, s, a, s_, r, mask):
        # shape: (batch_size, time_steps, -1)
        bs, steps = s.shape[:2]
        s = s.view(bs*steps, -1)
        a = a.view(bs*steps, -1)
        s_ = s_.view(bs*steps, -1)
        r = r.sum(dim=1, keepdims=True)/self.return_scale  # episodic return
        mask = mask.view(bs, steps)
        mask_len = mask.sum(dim=-1).long()

        s_a = torch.cat((s, a), dim=-1).view(bs, steps, -1)

        total_red_loss = 0
        for it in range(steps):
            break_p = np.random.randint(steps)
            self.sub_reward_model.init_hidden()
            sub_r = self.sub_reward_model(s_a[:, :break_p+1])*mask[:, :break_p+1]
            if break_p < steps - 1:
                self.sub_reward_model.init_hidden()
                sub_r2 = self.sub_reward_model(s_a[:, break_p+1:])*mask[:, break_p+1:]
                sub_r = torch.cat((sub_r, sub_r2), dim=-1)
            sub_r = sub_r[torch.arange(bs), mask_len-1] + (mask_len > break_p+1).float()*sub_r[:, break_p]
            red_loss = ((sub_r.flatten() - r.flatten()).pow(2)).mean()
            self.sub_reward_optim.zero_grad()
            red_loss.backward()
            self.sub_reward_optim.step()
            total_red_loss += red_loss.item()
        return total_red_loss/(it+1)

    def learn_step_reward_from(self, s, a,s_, r, mask):
        # shape: (batch_size, time_steps, -1)
        bs, steps = s.shape[:2]
        s = s.view(bs*steps, -1)
        a = a.view(bs*steps, -1)
        s_ = s_.view(bs*steps, -1)
        r    = r.sum(dim=1)/self.return_scale  # episodic return
        mask = mask.view(bs, steps)
        mask_len = mask.sum(dim=-1).long()

        s_a = torch.cat((s, a), dim=-1).view(bs, steps, -1)
        with torch.no_grad():
            self.sub_reward_model.init_hidden()
            sub_r = self.sub_reward_model(s_a)*mask
            sub_r[torch.arange(bs), mask_len-1] = r.flatten()
            diff_r = sub_r - torch.cat((torch.zeros((bs, 1), device=self.device), sub_r[:, :-1]), dim=-1)

        total_steprewloss = 0
        for it in range(steps):
            indices = np.random.choice(np.arange(bs*steps), size=bs, 
                p=(mask.flatten()/mask.sum()).cpu().numpy(), replace=False)
            step_trgt = diff_r.flatten()[indices]
                
            step_r = self.reward_model(torch.cat([s[indices], a[indices]], dim=-1)).flatten()
            steprew_loss = (step_r-step_trgt).pow(2).mean()
            self.reward_model_optim.zero_grad()
            steprew_loss.backward()
            self.reward_model_optim.step()
            total_steprewloss += steprew_loss.item()
        return total_steprewloss/(it+1)

    
    def update(self, states, actions, next_states, episode_return, episode_length):
        #states bs,t,d
        #actions bs,t,d
        #next_states bs,t,d
        #episode_return bs,1
        #episode_length bs,1
        max_episode_length = int(max(episode_length).item())
        states = states[:, :max_episode_length].contiguous()
        actions = actions[:, :max_episode_length].contiguous()
        next_states = next_states[:, :max_episode_length].contiguous()
        mask = torch.tensor(np.repeat(np.expand_dims(np.arange(states.shape[1]), 0), repeats=states.shape[0], axis=0)).to(episode_length.device) < episode_length
        if self.update_t % 1000 == 0:
            sub_reward_loss = self.learn_sub_reward_from(states, actions, next_states, episode_return, mask)
            step_reward_loss = self.learn_step_reward_from(states, actions, next_states, episode_return, mask)
            wandb.log({'sub_reward_loss': sub_reward_loss, 'step_reward_loss': step_reward_loss})
        self.update_t += 1
            # epired_loss = self.learn_sub_reward_from(states, actions, next_states, episode_return, mask)
            # steprew_loss = self.learn_step_reward_from(states, actions, next_states, episode_return, mask)
        
        return 0
