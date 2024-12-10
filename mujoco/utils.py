import numpy as np
import torch
import copy
class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_traj_len, max_size=int(1e6), log_prob=False):
		self.max_traj_len = max_traj_len
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.log_prob = log_prob

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.dense_reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.a_log_prob = np.zeros((max_size, action_dim))

		self.traj_head_id = []
		self.traj_end_id = []
		self.episode_return = []
		self.episode_length = []

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	def save_data(self, path):
		np.savez(path, state=self.state, action=self.action, next_state=self.next_state, reward=self.reward, dense_reward=self.dense_reward, not_done=self.not_done, traj_head_id=self.traj_head_id, traj_end_id=self.traj_end_id, episode_return=self.episode_return, episode_length=self.episode_length, ptr=self.ptr, size=self.size)


	def add_traj(self, state, action, next_state, reward,dense_reward, done, episode_return, episode_length, a_log_prob=None):
		assert len(state) == episode_length

		self.traj_head_id.append(self.ptr)
		self.traj_end_id.append((self.ptr + episode_length) % self.max_size)
		self.episode_return.append(episode_return)
		self.episode_length.append(episode_length)

		if len(self.traj_head_id)>1 and (self.traj_head_id[-1]>self.traj_end_id[-1]):
			for i in range(len(self.traj_head_id)):
				if self.traj_head_id[i] >= self.traj_end_id[-1]:
					break
			for j in range(i):
				self.traj_head_id.pop(0)
				self.traj_end_id.pop(0)
				self.episode_return.pop(0)
				self.episode_length.pop(0)
		if  len(self.traj_head_id)>1 and self.traj_head_id[-1]<=self.traj_head_id[0]:
			for i in range(len(self.traj_head_id)):
				if self.traj_head_id[i] >= self.traj_end_id[-1] or self.traj_head_id[i] < self.traj_head_id[-1]:
					break
			for j in range(i):
				self.traj_head_id.pop(0)
				self.traj_end_id.pop(0)
				self.episode_return.pop(0)
				self.episode_length.pop(0)

		if self.ptr+episode_length > self.max_size:
			self.state[self.ptr: self.max_size] = state[:self.max_size-self.ptr]
			self.action[self.ptr: self.max_size] = action[:self.max_size-self.ptr]
			self.next_state[self.ptr: self.max_size] = next_state[:self.max_size-self.ptr]
			self.reward[self.ptr: self.max_size] = reward[:self.max_size-self.ptr]
			self.dense_reward[self.ptr: self.max_size] = dense_reward[:self.max_size-self.ptr]
			self.not_done[self.ptr: self.max_size] = 1. - done[:self.max_size-self.ptr]
			if a_log_prob is not None:
				self.a_log_prob[self.ptr: self.max_size] = a_log_prob[:self.max_size-self.ptr]
				self.a_log_prob[: (self.ptr+episode_length)% self.max_size] = a_log_prob[self.max_size-self.ptr:]

			self.state[: (self.ptr+episode_length)% self.max_size] = state[self.max_size-self.ptr:]
			self.action[: (self.ptr+episode_length)% self.max_size] = action[self.max_size-self.ptr:]
			self.next_state[: (self.ptr+episode_length)% self.max_size] = next_state[self.max_size-self.ptr:]
			self.reward[: (self.ptr+episode_length)% self.max_size] = reward[self.max_size-self.ptr:]
			self.dense_reward[: (self.ptr+episode_length)% self.max_size] = dense_reward[self.max_size-self.ptr:]
			self.not_done[: (self.ptr+episode_length)% self.max_size] = 1. - done[self.max_size-self.ptr:]
		else:
			self.state[self.ptr: (self.ptr+episode_length)] = state
			self.action[self.ptr: (self.ptr+episode_length)] = action
			self.next_state[self.ptr: (self.ptr+episode_length)] = next_state
			self.reward[self.ptr: (self.ptr+episode_length)] = reward
			self.dense_reward[self.ptr: (self.ptr+episode_length)] = dense_reward
			self.not_done[self.ptr: (self.ptr+episode_length)] = 1. - done
			if a_log_prob is not None:
				self.a_log_prob[self.ptr: (self.ptr+episode_length)] = a_log_prob

		self.ptr = (self.ptr + episode_length) % self.max_size
		self.size = min(self.size + episode_length, self.max_size)


	def sample(self, batch_size):
		# ind = np.random.randint(0, self.size, size=batch_size)
		ind = np.random.choice(self.size, batch_size, replace=False)
		if self.log_prob:
			return (
				torch.FloatTensor(self.state[ind]).to(self.device),
				torch.FloatTensor(self.action[ind]).to(self.device),
				torch.FloatTensor(self.next_state[ind]).to(self.device),
				torch.FloatTensor(self.reward[ind]).to(self.device),
				torch.FloatTensor(self.dense_reward[ind]).to(self.device),
				torch.FloatTensor(self.not_done[ind]).to(self.device),
				torch.FloatTensor(self.a_log_prob[ind]).to(self.device)
			)
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.dense_reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
		)

	def sample_traj(self, traj_num, length_priority=False):
		if not length_priority:
			ind = np.random.randint(0, len(self.traj_head_id), size=traj_num)
		else:
			ind = np.random.choice(np.arange(len(self.traj_head_id)), p=np.array(self.episode_length)/np.sum(self.episode_length), size=traj_num)
		sampled_states = np.zeros((traj_num, self.max_traj_len, self.state_dim))
		sampled_actions = np.zeros((traj_num, self.max_traj_len, self.action_dim))
		sampled_next_states = np.zeros((traj_num, self.max_traj_len, self.state_dim))
		sampled_rewards = np.zeros((traj_num, self.max_traj_len, 1))
		sampled_dense_rewards = np.zeros((traj_num, self.max_traj_len, 1))
		sampled_not_dones = np.zeros((traj_num, self.max_traj_len, 1))
		sampled_a_log_prob = np.zeros((traj_num, self.max_traj_len, self.action_dim))
		sampled_episode_returns = np.zeros((traj_num, 1))
		sampled_episode_lengths = np.zeros((traj_num, 1))

		for i, idx in enumerate(ind):
			traj_head = self.traj_head_id[idx]
			traj_end = self.traj_end_id[idx]
			episode_return = self.episode_return[idx]
			episode_length = self.episode_length[idx]
			sampled_episode_returns[i] = episode_return
			sampled_episode_lengths[i] = episode_length
			if traj_head < traj_end:
				assert traj_end - traj_head == episode_length
				sampled_states[i, :traj_end-traj_head] = self.state[traj_head:traj_end]
				sampled_actions[i, :traj_end-traj_head] = self.action[traj_head:traj_end]
				sampled_next_states[i, :traj_end-traj_head] = self.next_state[traj_head:traj_end]
				sampled_rewards[i, :traj_end-traj_head] = self.reward[traj_head:traj_end]
				sampled_dense_rewards[i, :traj_end-traj_head] = self.dense_reward[traj_head:traj_end]
				sampled_not_dones[i, :traj_end-traj_head] = self.not_done[traj_head:traj_end]
				if self.log_prob:
					sampled_a_log_prob[i, :traj_end-traj_head] = self.a_log_prob[traj_head:traj_end]
			else:
				sampled_states[i, :self.max_size-traj_head] = self.state[traj_head:]
				sampled_actions[i, :self.max_size-traj_head] = self.action[traj_head:]
				sampled_next_states[i, :self.max_size-traj_head] = self.next_state[traj_head:]
				sampled_rewards[i, :self.max_size-traj_head] = self.reward[traj_head:]
				sampled_dense_rewards[i, :self.max_size-traj_head] = self.dense_reward[traj_head:]
				sampled_not_dones[i, :self.max_size-traj_head] = self.not_done[traj_head:]

				sampled_states[i, self.max_size-traj_head: episode_length] = self.state[:traj_end]
				sampled_actions[i, self.max_size-traj_head: episode_length] = self.action[:traj_end]
				sampled_next_states[i, self.max_size-traj_head: episode_length] = self.next_state[:traj_end]
				sampled_rewards[i, self.max_size-traj_head: episode_length] = self.reward[:traj_end]
				sampled_dense_rewards[i, self.max_size-traj_head: episode_length] = self.dense_reward[:traj_end]
				sampled_not_dones[i, self.max_size-traj_head: episode_length] = self.not_done[:traj_end]
				if self.log_prob:
					sampled_a_log_prob[i, self.max_size-traj_head: episode_length] = self.a_log_prob[:traj_end]
					sampled_a_log_prob[i, :self.max_size-traj_head] = self.a_log_prob[traj_head:]

		if self.log_prob:
			return (
				torch.FloatTensor(sampled_states).to(self.device),
				torch.FloatTensor(sampled_actions).to(self.device),
				torch.FloatTensor(sampled_next_states).to(self.device),
				torch.FloatTensor(sampled_rewards).to(self.device),
				torch.FloatTensor(sampled_dense_rewards).to(self.device),
				torch.FloatTensor(sampled_not_dones).to(self.device),
				torch.FloatTensor(sampled_episode_returns).to(self.device),
				torch.FloatTensor(sampled_episode_lengths).to(self.device),
				torch.FloatTensor(sampled_a_log_prob).to(self.device)
			)
		return (
			torch.FloatTensor(sampled_states).to(self.device),
			torch.FloatTensor(sampled_actions).to(self.device),
			torch.FloatTensor(sampled_next_states).to(self.device),
			torch.FloatTensor(sampled_rewards).to(self.device),
			torch.FloatTensor(sampled_dense_rewards).to(self.device),
			torch.FloatTensor(sampled_not_dones).to(self.device),
			torch.FloatTensor(sampled_episode_returns).to(self.device),
			torch.FloatTensor(sampled_episode_lengths).to(self.device)
		)



class HeapReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_length, max_size=int(10)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, max_length, state_dim))
		self.action = np.zeros((max_size, max_length, action_dim))
		self.next_state = np.zeros((max_size, max_length, state_dim))
		self.reward = np.zeros((max_size, max_length, 1))
		self.dense_reward = np.zeros((max_size, max_length, 1))
		self.not_done = np.zeros((max_size, max_length, 1))

		self.episode_return = np.zeros((max_size))
		self.episode_length = np.zeros((max_size))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.all_episode_length = []

	def add_traj(self, state, action, next_state, reward, dense_reward, done, episode_return, episode_length):
		if self.size >= self.max_size:
			self.ptr = np.argmin(self.episode_return)
			self.min_R = self.episode_return[self.ptr]
			if episode_return <= self.min_R:
				return
			else:
				self.state[self.ptr] *= 0
				self.action[self.ptr] *= 0
				self.next_state[self.ptr] *= 0
				self.reward[self.ptr] *= 0
				self.dense_reward[self.ptr] *= 0
				self.not_done[self.ptr] *= 0
				self.episode_return[self.ptr] = 0
				self.episode_length[self.ptr] = 0
		self.state[self.ptr, :episode_length] = state
		self.action[self.ptr, :episode_length] = action
		self.next_state[self.ptr, :episode_length] = next_state
		self.reward[self.ptr, :episode_length] = episode_return/episode_length
		self.dense_reward[self.ptr, :episode_length] = dense_reward
		self.not_done[self.ptr, :episode_length] = 1. - done
		self.episode_return[self.ptr] = episode_return
		self.episode_length[self.ptr] = episode_length

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample_transition(self, traj_id):
		ind = np.random.randint(0, self.episode_length[traj_id], size=1)
		return (
			self.state[traj_id, ind],
			self.action[traj_id, ind],
			self.next_state[traj_id, ind],
			self.reward[traj_id, ind],
			self.dense_reward[traj_id, ind],
			self.not_done[traj_id, ind],
		)
	
	def sample(self, batch_size):
		sampled_s = []
		sampled_a = []
		sampled_ns = []
		sampled_r = []
		sampled_dr = []
		sampled_nd = []

		for i in range(batch_size):
			random_traj_idx = np.random.randint(0, self.size)
			s,a,ns,r,dr,nd = self.sample_transition(random_traj_idx)
			sampled_s.append(s)
			sampled_a.append(a)
			sampled_ns.append(ns)
			sampled_r.append(r)
			sampled_dr.append(dr)
			sampled_nd.append(nd)

		return (
			torch.FloatTensor(np.concatenate(sampled_s,axis=0)).to(self.device),
			torch.FloatTensor(np.concatenate(sampled_a,axis=0)).to(self.device),
			torch.FloatTensor(np.concatenate(sampled_ns,axis=0)).to(self.device),
			torch.FloatTensor(np.concatenate(sampled_r,axis=0)).to(self.device),
			torch.FloatTensor(np.concatenate(sampled_dr,axis=0)).to(self.device),
			torch.FloatTensor(np.concatenate(sampled_nd,axis=0)).to(self.device),
		)

	

class IRCRReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_traj_len, max_size=int(1e6)):
		self.max_traj_len = max_traj_len
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.state_dim = state_dim
		self.action_dim = action_dim

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.dense_reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.traj_head_id = []
		self.traj_end_id = []
		self.episode_return = []
		self.episode_length = []

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.heap_replay_buffer = HeapReplayBuffer(state_dim, action_dim, max_traj_len)
		self.R_max = -1e10
		self.R_min = 1e10


	def update_heap(self, state, action, next_state, reward,dense_reward, done, episode_return, episode_length):
		self.heap_replay_buffer.add_traj(state, action, next_state, reward, dense_reward, done, episode_return, episode_length)
		self.R_max = max(np.array(self.episode_return)/np.array(self.episode_length))
		self.R_min = min(np.array(self.episode_return)/np.array(self.episode_length))

	def add_traj(self, state, action, next_state, reward,dense_reward, done, episode_return, episode_length):
		assert len(state) == episode_length
		self.traj_head_id.append(self.ptr)
		self.traj_end_id.append((self.ptr + episode_length) % self.max_size)
		self.episode_return.append(episode_return)
		self.episode_length.append(episode_length)

		if len(self.traj_head_id)>1 and (self.traj_head_id[-1]>self.traj_end_id[-1]):
			for i in range(len(self.traj_head_id)):
				if self.traj_head_id[i] >= self.traj_end_id[-1]:
					break
			for j in range(i):
				self.traj_head_id.pop(0)
				self.traj_end_id.pop(0)
				self.episode_return.pop(0)
				self.episode_length.pop(0)
		if  len(self.traj_head_id)>1 and self.traj_head_id[-1]<=self.traj_head_id[0]:
			for i in range(len(self.traj_head_id)):
				if self.traj_head_id[i] >= self.traj_end_id[-1] or self.traj_head_id[i] < self.traj_head_id[-1]:
					break
			for j in range(i):
				self.traj_head_id.pop(0)
				self.traj_end_id.pop(0)
				self.episode_return.pop(0)
				self.episode_length.pop(0)

		self.update_heap(state, action, next_state, reward, dense_reward, done, episode_return, episode_length)

		if self.ptr+episode_length > self.max_size:
			self.state[self.ptr: self.max_size] = state[:self.max_size-self.ptr]
			self.action[self.ptr: self.max_size] = action[:self.max_size-self.ptr]
			self.next_state[self.ptr: self.max_size] = next_state[:self.max_size-self.ptr]
			self.reward[self.ptr: self.max_size] = episode_return/episode_length
			self.dense_reward[self.ptr: self.max_size] = dense_reward[:self.max_size-self.ptr]
			self.not_done[self.ptr: self.max_size] = 1. - done[:self.max_size-self.ptr]

			self.state[: (self.ptr+episode_length)% self.max_size] = state[self.max_size-self.ptr:]
			self.action[: (self.ptr+episode_length)% self.max_size] = action[self.max_size-self.ptr:]
			self.next_state[: (self.ptr+episode_length)% self.max_size] = next_state[self.max_size-self.ptr:]
			self.reward[: (self.ptr+episode_length)% self.max_size] = episode_return/episode_length
			self.dense_reward[: (self.ptr+episode_length)% self.max_size] = dense_reward[self.max_size-self.ptr:]
			self.not_done[: (self.ptr+episode_length)% self.max_size] = 1. - done[self.max_size-self.ptr:]
		else:
			self.state[self.ptr: (self.ptr+episode_length)] = state
			self.action[self.ptr: (self.ptr+episode_length)] = action
			self.next_state[self.ptr: (self.ptr+episode_length)] = next_state
			self.reward[self.ptr: (self.ptr+episode_length)] = episode_return/episode_length
			self.dense_reward[self.ptr: (self.ptr+episode_length)] = dense_reward
			self.not_done[self.ptr: (self.ptr+episode_length)] = 1. - done

		self.ptr = (self.ptr + episode_length) % self.max_size
		self.size = min(self.size + episode_length, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size//2)
		heat_sampled_state, heap_sampled_action, heap_sampled_next_state, heap_sampled_reward, heap_sampled_dense_reward, heap_sampled_not_done = self.heap_replay_buffer.sample(batch_size//2)

		return (
			torch.cat([torch.FloatTensor(self.state[ind]).to(self.device), heat_sampled_state], dim=0),
			torch.cat([torch.FloatTensor(self.action[ind]).to(self.device), heap_sampled_action], dim=0),
			torch.cat([torch.FloatTensor(self.next_state[ind]).to(self.device), heap_sampled_next_state], dim=0),
			(torch.cat([torch.FloatTensor(self.reward[ind]).to(self.device), heap_sampled_reward], dim=0)-self.R_min)/(self.R_max-self.R_min),
			torch.cat([torch.FloatTensor(self.dense_reward[ind]).to(self.device), heap_sampled_dense_reward], dim=0),
			torch.cat([torch.FloatTensor(self.not_done[ind]).to(self.device), heap_sampled_not_done], dim=0),
		)
