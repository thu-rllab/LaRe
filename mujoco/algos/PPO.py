import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal
import numpy as np
import copy
import wandb

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size):
        self.s = np.zeros((max_size, state_dim))
        self.a = np.zeros((max_size, action_dim))
        self.a_logprob = np.zeros((max_size, action_dim))
        self.r = np.zeros((max_size, 1))
        self.dense_r = np.zeros((max_size, 1))
        self.s_ = np.zeros((max_size, state_dim))
        self.dw = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))
        self.size = 0

    def add(self, s, a, a_logprob, s_ ,r,dense_r, dw, done):
        self.s[self.size] = s
        self.a[self.size] = a
        self.a_logprob[self.size] = a_logprob
        self.r[self.size] = r
        self.dense_r[self.size] = dense_r
        self.s_[self.size] = s_
        self.dw[self.size] = dw
        self.done[self.size] = done
        self.size += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float).to(device)
        a = torch.tensor(self.a, dtype=torch.float).to(device)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(device)
        r = torch.tensor(self.r, dtype=torch.float).to(device)
        dense_r = torch.tensor(self.dense_r, dtype=torch.float).to(device)
        s_ = torch.tensor(self.s_, dtype=torch.float).to(device)
        dw = torch.tensor(self.dw, dtype=torch.float).to(device)
        done = torch.tensor(self.done, dtype=torch.float).to(device)

        return s, a, a_logprob, r,dense_r, s_, dw, done

class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO(object):
    def __init__(self,
        args, 
        state_dim,
		action_dim,
		max_action, discount=0.99, tau=0.005):
        self.args = args
        self.policy_dist = "Beta"#"Gaussian" #
        self.max_action = max_action
        args.state_dim = state_dim
        args.action_dim = action_dim
        args.hidden_width = 256
        args.use_tanh = True
        args.use_orthogonal_init = True
        # self.batch_size = args.batch_size
        self.mini_batch_size = 64
        self.max_train_steps = args.max_timesteps
        self.lr_a = 3e-4  # Learning rate of actor
        self.lr_c = 3e-4  # Learning rate of critic
        self.gamma = discount  # Discount factor
        self.lamda = 0.95  # GAE parameter
        self.epsilon = 0.2  # PPO clip parameter
        self.K_epochs = 10  # PPO parameter
        self.entropy_coef = 0.01  # Entropy coefficient
        self.set_adam_eps = 1
        self.use_grad_clip = 1
        self.use_lr_decay = 1
        self.use_adv_norm = 1
        self.use_state_norm = 0
        self.use_reward_scaling=0
        
        self.state_norm = Normalization(shape=args.state_dim)
        self.reward_scaling = RewardScaling(shape=1, gamma=self.gamma)
        self.reward_norm = Normalization(shape=1)


        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args).to(device)
        else:
            self.actor = Actor_Gaussian(args).to(device)
        self.critic = Critic(args).to(device)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        from reward_model import import_reward_model
        self.reward_model = import_reward_model(self.args)

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
        if self.policy_dist == "Beta":
            a = self.actor.mean(s).cpu().detach().numpy().flatten()
        else:
            a = self.actor(s).cpu().detach().numpy().flatten()
        if self.policy_dist == "Beta":
            a = 2 * (a - 0.5) * self.max_action  # [0,1]->[-max,max]
        else:
            a = a
        return a

    def select_action(self, s, log_prob=False, deterministic=False):
        if deterministic:
            with torch.no_grad():
                a = self.evaluate(s)
            return a
        
        # if self.use_state_norm:
        #     s = self.state_norm(s)
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        a = a.cpu().numpy().flatten()
        if self.policy_dist == "Beta":
            action = 2 * (a - 0.5) * self.max_action  # [0,1]->[-max,max]
        else:
            action = a
        if log_prob:
            return action, a_logprob.cpu().numpy().flatten()
        else:
            return action

    def train(self,old_replay_buffer, replay_buffer, batch_size=256, total_steps=0):
        s, a, a_logprob, r, dense_reward, s_, dw, done = old_replay_buffer.numpy_to_tensor()
        train_info = {}
        not_done = (1.-done)
        # s, a, s_, r, dense_reward, not_done, a_logprob = replay_buffer.sample(batch_size) #bs,d
        if self.policy_dist == "Beta":
            a = (a / self.max_action + 1) / 2
        if self.args.dense_r:
            r = dense_reward
        if self.reward_model is not None:
            r = self.reward_model.forward(s, a, s_)
            r = r.detach()
        train_info['reward_pred_err'] = F.mse_loss(r, dense_reward).item()
        

        # state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, nd in zip(reversed(deltas.cpu().flatten().numpy()), reversed(not_done.cpu().flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * nd
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(device)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))#bs,1

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

        train_info['actor_loss'] = actor_loss.mean().item()
        train_info['critic_loss'] = critic_loss.item()
        train_info['ppo_ratios'] = ratios.mean().item()
        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)
        
        return train_info
    
    def train_reward(self, replay_buffer, batch_size=256):
        train_info = {}
        reward_model_loss = 0
        if self.reward_model is not None and not self.args.direct_generate: 
            if 'RRD' in self.args.rd_method or self.args.rd_method=='VIB':
                traj_state, traj_action, traj_next_state, traj_reward, traj_dense_reward, traj_not_done, traj_episode_return, traj_episode_length,_ = replay_buffer.sample_traj(int(batch_size//self.args.rrd_k)) #bs,t,d
            else:
                traj_state, traj_action, traj_next_state, traj_reward, traj_dense_reward, traj_not_done, traj_episode_return, traj_episode_length,_ = replay_buffer.sample_traj(max(int(batch_size//np.mean(replay_buffer.episode_length)), 1)) #bs,t,d

            reward_model_loss = self.reward_model.update(traj_state, traj_action, traj_next_state, traj_episode_return, traj_episode_length)
            train_info['reward_model_loss'] = reward_model_loss
        return reward_model_loss

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps) + 1e-5
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps) + 1e-5
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_c_now

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        # self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        # self.actor_target = copy.deepcopy(self.actor)