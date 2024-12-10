import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		args,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):
		self.args = args
		self.state_dim = state_dim
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0

		from reward_model import import_reward_model
		self.reward_model = import_reward_model(self.args)
		


	def select_action(self, state, deterministic=False):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer,batch_size=256):
		self.total_it += 1
		train_info = {}

		# Sample replay buffer 
		start_time = time.time()
		state, action, next_state, reward, dense_reward, not_done = replay_buffer.sample(batch_size) #bs,d
		sample_time = time.time() - start_time

		if self.args.dense_r:
			reward = dense_reward
		if self.reward_model is not None:
			reward = self.reward_model.forward(state, action, next_state)
			reward = reward.detach()
		train_info['reward_pred_err'] = F.mse_loss(reward, dense_reward).item()
		
		start_time = time.time()
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q
		target_time = time.time() - start_time

		# Get current Q estimates
		start_time = time.time()
		current_Q1, current_Q2 = self.critic(state, action)
		critic_time = time.time() - start_time

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		start_time = time.time()
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		critic_update_time = time.time() - start_time
		train_info['critic_loss'] = critic_loss.item()

		
		if self.reward_model is not None and not self.args.direct_generate:
			start_time = time.time()
			if 'RRD' in self.args.rd_method or self.args.rd_method=='VIB':
				traj_state, traj_action, traj_next_state, traj_reward, traj_dense_reward, traj_not_done, traj_episode_return, traj_episode_length= replay_buffer.sample_traj(int(self.args.batch_size//self.args.rrd_k)) #bs,t,d
			elif self.args.rd_method=='Diaster':
				traj_state, traj_action, traj_next_state, traj_reward, traj_dense_reward, traj_not_done, traj_episode_return, traj_episode_length = replay_buffer.sample_traj(batch_size//4, length_priority=True) #bs,t,d
			else:
				traj_state, traj_action, traj_next_state, traj_reward, traj_dense_reward, traj_not_done, traj_episode_return, traj_episode_length = replay_buffer.sample_traj(max(int(batch_size//np.mean(replay_buffer.episode_length)), 1)) #bs,t,d
			reward_sample_time = time.time() - start_time
			start_time = time.time()
			reward_model_loss = self.reward_model.update(traj_state, traj_action, traj_next_state, traj_episode_return, traj_episode_length)
			reward_update_time = time.time() - start_time
			train_info['reward_model_loss'] = reward_model_loss

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			policy_start_time = time.time()

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			policy_update_time = time.time() - policy_start_time

			train_info['actor_loss'] = actor_loss.item()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			
		# return {'reward_sample_time':reward_sample_time,'reward_update_time':reward_update_time ,'sample_time': sample_time, 'target_time': target_time, 'critic_time': critic_time, 'critic_update_time': critic_update_time, 'policy_update_time': policy_update_time}
		return train_info




	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		