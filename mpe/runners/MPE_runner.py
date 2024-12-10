import torch
import numpy as np
import numpy as np
import torch
from torch import nn
import argparse
from utils.replay_memory import ReplayBuffer
from utils.norm import Normalization
from utils.replay_memory import ReplayMemory_episode
from models.policy.ppo import *
from models.reward_model.mard.mard import STAS, STAS_ML
from models.reward_model.arel.transformers import Time_Agent_Transformer
from utils.util import *
import wandb
import time
import os


class Runner_STAS_MPE:
    def __init__(self, args, env_name, env, n_agents, n_action, obs_dim, device):
        self.args = args
        self.env_name = env_name
        self.env = env
        self.device = device
        self.n_agents = n_agents
        self.n_action = n_action
        self.obs_dim = obs_dim

        self.agent = PPO(self.args,self.obs_dim, self.n_action, self.n_agents, args.num_epochs, args.max_step_per_round,
                    args.num_episodes, args.lr, args.clip, args.minibatch_size, args.gamma, args.lamda,
                    args.loss_coeff_value, args.loss_coeff_entropy, args.schedule_clip, args.schedule_adam, args.value_norm, args.reward_norm,
                    args.advantage_norm, args.lossvalue_norm, args.layer_norm, args.return_decomposition, device)
        self.memory_e = ReplayMemory_episode(args.buffer_size, args.max_step_per_round, args.reward_norm)

        self.total_numsteps = 0
        self.running_episodes = 0
        self.episode = 0

        time_length, state_emb = args.max_step_per_round, self.obs_dim
        if args.reward_model_version == 'v1':
            self.reward_model = STAS(input_dim=state_emb, n_actions = self.n_action, emb_dim = args.hidden_size, 
                                    n_heads=args.n_heads, n_layer=args.n_layers, seq_length=time_length, 
                                    n_agents=self.n_agents, sample_num = args.nums_sample_from_coalition, device = device, dropout=0.3)
        else:
            self.reward_model = STAS_ML(input_dim=state_emb, n_actions = self.n_action, emb_dim = args.hidden_size, 
                                    n_heads=args.n_heads, n_layer=args.n_layers, seq_length=time_length, 
                                    n_agents=self.n_agents, sample_num = args.nums_sample_from_coalition, device = device, dropout=0.3)
        if torch.cuda.is_available() and args.cuda:
            self.reward_model.cuda()

        opt = torch.optim.Adam(lr=args.lr_reward, params=self.reward_model.parameters(), weight_decay=1e-5)
        loss_fn = nn.MSELoss(reduction='mean')

        # Creates the train_step function for our model, loss function and optimizer
        self.train_step = make_train_step(self.reward_model, loss_fn, opt, self.n_agents, device, args.reg, args.alpha)
    
    def run(self):
        evaluate_num = -1
        for i_episode in range(self.args.num_episodes):
            num_steps, ep_num = 0, 0

            while num_steps < self.args.batch_size:
                episode_reward, t,_ = self.run_episode_MPE(evaluate=False, i_episode=i_episode, ep_num=ep_num)

                ep_num += 1
                self.running_episodes += 1
                num_steps += (t + 1)
                self.total_numsteps += (t + 1)

            if len(self.memory_e)>self.args.policy_starts:
                policy_total_loss, loss_surr, loss_value, loss_entropy, r_statistics = self.agent.update_parameters(self.agent.memory, self.reward_model, i_episode, use_next_state=self.args.use_next_state)
                wandb.log({'policy_total_loss': policy_total_loss, 'loss_surr':loss_surr, 
                        'loss_value': loss_value, 'loss_entropy': loss_entropy, 
                        'episodes':self.episode, 'epoch':i_episode, 'total_t':self.total_numsteps}, step = self.episode)
                
            if (i_episode+1) % self.args.reward_model_update_freq == 0 and (len(self.memory_e)>self.args.reward_model_starts) and len(self.memory_e)>=self.args.rewardbatch_size:
                reward_model_train = 1
                epoch_train_total_reward_loss = []
                if self.args.reward_norm:
                    self.reward_model.module.reward_normalizer.update(self.memory_e.get_update_data())
                    self.memory_e.reset()
                for ii in range(self.args.updates_per_step):
                    if not self.args.use_next_state:
                        states, actions, episode_return, episode_reward, episode_length = self.memory_e.sample_trajectory(n_trajectories=self.args.rewardbatch_size)
                        next_states = None
                    else:
                        states, actions, episode_return, episode_reward, episode_length, next_states = self.memory_e.sample_trajectory(n_trajectories=self.args.rewardbatch_size, use_next_state=True)
                        next_states = next_states.to(self.device)
                    states = states.to(self.device)#bs,t,na,d
                    actions = actions.to(self.device)#bs,t,na
                    episode_return = episode_return.to(self.device)#bs
                    episode_length = episode_length.to(self.device)#bs
                    if self.args.factor_reward_model_layers == 0:
                        loss = 0
                    else:
                        loss = self.train_step(states, actions, episode_return, episode_length, next_states=next_states)
                    epoch_train_total_reward_loss.append(loss)

                if self.args.wandb:
                    wandb.log({'reward_model_loss': np.mean(epoch_train_total_reward_loss)}, step=self.episode)

            self.episode += ep_num
            if self.episode // self.args.eval_freq > evaluate_num:
                self.evaluate_policy(i_episode)
                evaluate_num += 1
 
            self.agent.reset_memory()

        self.env.close()
        self.agent.save_model(self.args.model_save_path)
        time.sleep(5)

    def run_episode_MPE(self, evaluate, i_episode, ep_num):
        self.agent.save_model(self.args.model_save_path)
        obs_n = self.env.reset()
        if self.args.obs_agent_id:
            obs_n = np.concatenate([np.array(obs_n), np.eye(self.n_agents)], axis=-1)
        if self.args.state_norm:
            obs_n = self.agent.state_norm(obs_n)
        
        episode_reward = 0
        x_e, action_e, value_e, mask_e, x_next_e, logproba_e, reward_e, info_e, sparse_reward_e = [], [], [], [], [], [], [], [], []
        for t in range(self.args.max_step_per_round):
            action_n, logproba_n, value_n = self.agent.select_action(torch.Tensor(obs_n).to(self.device), None, False)
            next_obs_n, reward_n, done_n, info_n = self.env.step(action_n)
            if self.args.obs_agent_id:
                next_obs_n = np.concatenate([np.array(next_obs_n), np.eye(self.n_agents)], axis=-1)

            if self.args.state_norm:
                next_obs_n = self.agent.state_norm(next_obs_n)

            episode_reward += np.sum(reward_n)
            
            x_e.append(np.array(obs_n))
            action_e.append(action_n.reshape(1,-1))
            value_e.append(value_n.reshape(1,-1))
            mask_e.append(np.array([[not done if t<self.args.max_step_per_round-1 else False for done in done_n]]))
            x_next_e.append(np.array(next_obs_n))
            logproba_e.append(logproba_n.reshape(1,-1))
            sparse_reward_e.append(np.array([[0]*self.n_agents]))
            reward_e.append(np.array([reward_n]))
            info_e.append(np.array(info_n))
            
            if all(done_n) or t == self.args.max_step_per_round-1:
                if not self.args.dense_r:
                    sparse_reward_e[-1] = np.array([[episode_reward/self.n_agents]*self.n_agents])
                    reward_e = sparse_reward_e
                mode = 'train' if not evaluate else 'eval'
                print(mode, ' episode reward: ', episode_reward, ', episode/ep: {}/{}'.format(i_episode, ep_num))
                if not evaluate:
                    self.memory_e.push(x_e, action_e, mask_e, x_next_e, reward_e)
                    self.agent.memory.push(x_e, value_e, action_e, logproba_e, mask_e,
                            x_next_e, reward_e, info_e)
                break
            obs_n = next_obs_n
        
        return episode_reward, t, info_e

    def evaluate_policy(self, i_episode):
        evaluate_rewards = 0
        episode_steps = []

        for _ in range(self.args.num_eval_runs):
            episode_reward, t,infos = self.run_episode_MPE(evaluate=True, i_episode=i_episode, ep_num=_)
            episode_steps.append(t)
            evaluate_rewards += episode_reward
        
        if self.args.wandb:
            wandb.log({'agent_reward': evaluate_rewards / self.args.num_eval_runs, 'episode_steps':np.mean(episode_steps), 'epoch':i_episode, 'episodes':self.episode, 'total_t':self.total_numsteps}, step=self.episode)
        else:
            print('eval policy, agent reward: {:.4f}'.format(evaluate_rewards / self.args.num_eval_runs))
    
class Runner_PPO_MPE(Runner_STAS_MPE):
    def __init__(self, args, env_name, env, n_agents, n_action, obs_dim, device):
        super(Runner_PPO_MPE, self).__init__(args, env_name, env, n_agents, n_action, obs_dim, device)
        self.args.reward_model_update_freq = self.args.num_episodes + 100
class Runner_AREL_MPE(Runner_STAS_MPE):
    def __init__(self, args, env_name, env, n_agents, n_action, obs_dim, device):
        self.args = args
        self.env_name = env_name
        self.env = env
        self.device = device
        self.n_agents = n_agents
        self.n_action = n_action
        self.obs_dim = obs_dim
        self.agent = AREL_PPO(self.args, self.obs_dim, self.n_action, self.n_agents, args.num_epochs, args.max_step_per_round,
                    args.num_episodes, args.lr, args.clip, args.minibatch_size, args.gamma, args.lamda,
                    args.loss_coeff_value, args.loss_coeff_entropy, args.schedule_clip, args.schedule_adam, args.value_norm, args.reward_norm,
                    args.advantage_norm, args.lossvalue_norm, args.layer_norm, args.return_decomposition, device)
        self.memory_e = ReplayMemory_episode(args.buffer_size, args.max_step_per_round, args.reward_norm)

        self.total_numsteps = 0
        self.running_episodes = 0
        self.episode = 0

        time_length, state_emb = args.max_step_per_round, self.obs_dim
        n_heads = 3
        depth = 3
        self.reward_model = Time_Agent_Transformer(emb=state_emb, heads=n_heads, 
                            depth=depth, seq_length=time_length, 
                            n_agents=n_agents, agent=True, dropout=0.0)

        if torch.cuda.is_available() and args.cuda:
            self.reward_model.cuda()

        opt = torch.optim.Adam(lr=args.lr_reward, params=self.reward_model.parameters(), weight_decay=1e-5)
        loss_fn = nn.MSELoss(reduction='mean')
        self.train_step = make_arel_train_step(self.reward_model, loss_fn, opt, self.n_agents, device, args.reg, args.alpha)

class Runner_Factor_LLMrd_MPE(Runner_STAS_MPE):
    def __init__(self, args, env_name, env, n_agents, n_action, obs_dim, device):
        self.args = args
        self.env_name = env_name
        self.env = env
        self.device = device
        self.n_agents = n_agents
        self.n_action = n_action
        self.obs_dim = obs_dim

        self.agent = PPO(args, self.obs_dim, self.n_action, self.n_agents, args.num_epochs, args.max_step_per_round,
                    args.num_episodes, args.lr, args.clip, args.minibatch_size, args.gamma, args.lamda,
                    args.loss_coeff_value, args.loss_coeff_entropy, args.schedule_clip, args.schedule_adam, args.value_norm, args.reward_norm,
                    args.advantage_norm, args.lossvalue_norm, args.layer_norm, args.return_decomposition, device)
        self.memory_e = ReplayMemory_episode(args.buffer_size, args.max_step_per_round, args.reward_norm)

        self.total_numsteps = 0
        self.running_episodes = 0
        self.episode = 0

        time_length, state_emb = args.max_step_per_round, self.obs_dim
        from models.reward_model.LLMrd.factor_reward_decompose import FactorRewardDecomposer
        args.obs_dim = self.obs_dim
        self.reward_model = FactorRewardDecomposer(args)

        opt = torch.optim.Adam(lr=args.lr_reward, params=self.reward_model.reward_model.parameters(), weight_decay=1e-5)
        loss_fn = nn.MSELoss(reduction='mean')

        # Creates the train_step function for our model, loss function and optimizer
        self.train_step = make_train_step(self.reward_model, loss_fn, opt, self.n_agents, device, args.reg, args.alpha)


class Runner_RRD_MPE(Runner_STAS_MPE):
    def __init__(self, args, env_name, env, n_agents, n_action, obs_dim, device):
        self.args = args
        self.env_name = env_name
        self.env = env
        self.device = device
        self.n_agents = n_agents
        self.n_action = n_action
        self.obs_dim = obs_dim

        self.agent = PPO(self.args,self.obs_dim, self.n_action, self.n_agents, args.num_epochs, args.max_step_per_round,
                    args.num_episodes, args.lr, args.clip, args.minibatch_size, args.gamma, args.lamda,
                    args.loss_coeff_value, args.loss_coeff_entropy, args.schedule_clip, args.schedule_adam, args.value_norm, args.reward_norm,
                    args.advantage_norm, args.lossvalue_norm, args.layer_norm, args.return_decomposition, device)
        self.memory_e = ReplayMemory_episode(args.buffer_size, args.max_step_per_round, args.reward_norm)

        self.total_numsteps = 0
        self.running_episodes = 0
        self.episode = 0

        time_length, state_emb = args.max_step_per_round, self.obs_dim
        from models.reward_model.rrd.rrd import RRDRewardDecomposer
        args.obs_dim = self.obs_dim
        self.reward_model = RRDRewardDecomposer(args)

        opt = torch.optim.Adam(lr=args.lr_reward, params=self.reward_model.parameters(), weight_decay=1e-5)
        loss_fn = nn.MSELoss(reduction='mean')

        # Creates the train_step function for our model, loss function and optimizer
        self.train_step = make_rrd_train_step(args, self.reward_model, loss_fn, opt, self.n_agents, device, args.reg, args.alpha)


class Runner_VIB_MPE(Runner_STAS_MPE):
    def __init__(self, args, env_name, env, n_agents, n_action, obs_dim, device):
        self.args = args
        self.env_name = env_name
        self.env = env
        self.device = device
        self.n_agents = n_agents
        self.n_action = n_action
        self.obs_dim = obs_dim

        self.agent = VAE_PPO(args, self.obs_dim, self.n_action, self.n_agents, args.num_epochs, args.max_step_per_round,
                    args.num_episodes, args.lr, args.clip, args.minibatch_size, args.gamma, args.lamda,
                    args.loss_coeff_value, args.loss_coeff_entropy, args.schedule_clip, args.schedule_adam, args.value_norm, args.reward_norm,
                    args.advantage_norm, args.lossvalue_norm, args.layer_norm, args.return_decomposition, device)
        self.memory_e = ReplayMemory_episode(args.buffer_size, args.max_step_per_round, args.reward_norm)

        self.total_numsteps = 0
        self.running_episodes = 0
        self.episode = 0

        time_length, state_emb = args.max_step_per_round, self.obs_dim
        from models.reward_model.vae.vae import VAERewardDecomposer
        args.obs_dim = self.obs_dim
        self.reward_model = VAERewardDecomposer(args)

        opt = torch.optim.Adam(lr=args.lr_reward, params=list(self.reward_model.reward_model.parameters())+list(self.reward_model.vae.parameters()), weight_decay=1e-5)
        loss_fn = nn.MSELoss(reduction='mean')

        # Creates the train_step function for our model, loss function and optimizer
        self.train_step = make_IB_train_step(args, self.reward_model, loss_fn, opt, self.n_agents, device, args.reg, args.alpha)
