import random
from collections import namedtuple
import numpy as np
import torch

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

Transition_e = namedtuple(
    'Transition_episode', ('states', 'actions', 'masks', 'next_states', 'rewards'))

globals()["Transition"] = Transition
globals()["Transition_episode"] = Transition_e

class ReplayMemory_episode(object):

    def __init__(self, capacity, max_step_per_round, reward_norm=False):
        self.max_step_per_round = max_step_per_round
        self.capacity = capacity
        self.memory = []
        self.position = 0

        self.data = []
        self.reward_norm = reward_norm

    def padding(self, seq, pad):
        padded_seq = [np.full(seq[0].shape, pad) for i in range(self.max_step_per_round-len(seq))]
        return seq+padded_seq

    def _push(self, episode, memory, position, length):
        if len(memory) < self.capacity:
            memory.append(None)
        memory[position] = (episode, length)
        position = (position + 1) % self.capacity
        return position

    def push(self, *args):
        """Saves a transition."""
        # episode = Transition_e(*args)
        l = len(args[0])
        episode = [self.padding(arg, 0) if i!=1 else self.padding(arg, 4) for i, arg in enumerate(args) ]
        self.position = self._push(Transition(*episode), self.memory, self.position, l)
        if self.reward_norm:
            self.data.append(Transition_e(*args))

    def sample_trajectory(self, n_trajectories=128, use_next_state=False):
        sample_traj = random.sample(self.memory, n_trajectories)
        sample_traj, sample_length = zip(*sample_traj)
        episode_length = torch.Tensor(sample_length).long()
        samples = Transition_e(*zip(*sample_traj))

        states = np.array(samples.states).transpose((0,2,1,3))
        states = torch.from_numpy(states).float()
        actions = np.squeeze(np.array(samples.actions), axis=2).transpose((0,2,1))
        actions = torch.from_numpy(actions)

        # rewards: [B, T, N_a]
        rewards = np.squeeze(np.array(samples.rewards), axis=2)
        # episode_return = np.sum(rewards[:,:,:], axis=(1,2))
        episode_return = np.sum(rewards[:,:,0], axis=1)
        episode_return = torch.from_numpy(episode_return).float()
        # episode_reward = rewards[:,:,:]
        episode_reward = rewards[:,:,0]
        episode_reward = torch.from_numpy(episode_reward).float()
        if use_next_state:
            next_states = np.array(samples.next_states).transpose((0,2,1,3))
            next_states = torch.from_numpy(next_states).float()
            return states, actions, episode_return, episode_reward, episode_length, next_states
        else:
            return states, actions, episode_return, episode_reward, episode_length

    def reset(self):
        self.data = []
    
    def get_update_data(self):
        data = Transition_e(*zip(*self.data))
        rewards = np.sum(np.squeeze(np.array(data.rewards), axis=2)[:,:,0], axis=1)
        return rewards

    def shuffle(self):
        random.shuffle(self.memory)

    def __len__(self):
        return len(self.memory)


class ReplayBuffer:
    def __init__(self, args, n_agents, obs_dim, n_action, state_dim, device):
        self.N = n_agents
        self.action_dim = n_action
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.episode_limit = args.max_step_per_round
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.current_size = 0
        self.device = device
        self.buffer = {'obs_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.obs_dim]),
                       's': np.zeros([self.buffer_size, self.episode_limit + 1, self.state_dim]),
                       'avail_a_n': np.ones([self.buffer_size, self.episode_limit + 1, self.N, self.action_dim]),  # Note: We use 'np.ones' to initialize 'avail_a_n'
                       'last_onehot_a_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.action_dim]),
                       'a_n': np.zeros([self.buffer_size, self.episode_limit, self.N]),
                       'r': np.zeros([self.buffer_size, self.episode_limit, 1]),
                       'dw': np.ones([self.buffer_size, self.episode_limit, 1]),  # Note: We use 'np.ones' to initialize 'dw'
                       'active': np.zeros([self.buffer_size, self.episode_limit, 1])
                       }
        self.episode_len = np.zeros(self.buffer_size)

    def store_transition(self, episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.buffer['last_onehot_a_n'][self.episode_num][episode_step] = last_onehot_a_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['r'][self.episode_num][episode_step] = r
        self.buffer['dw'][self.episode_num][episode_step] = dw

        self.buffer['active'][self.episode_num][episode_step] = 1.0

    def store_last_step(self, episode_step, obs_n, s, avail_a_n, last_onehot_a_n):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.buffer['last_onehot_a_n'][self.episode_num][episode_step] = last_onehot_a_n
        self.episode_len[self.episode_num] = episode_step  # Record the length of this episode
        self.episode_num = (self.episode_num + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, batch_size=None):
        # Randomly sampling
        index = np.random.choice(self.current_size, size=self.batch_size if batch_size is None else batch_size, replace=False)
        max_episode_len = int(np.max(self.episode_len[index]))
        batch = {}
        for key in self.buffer.keys():
            if key == 'obs_n' or key == 's' or key == 'avail_a_n' or key == 'last_onehot_a_n':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len + 1], dtype=torch.float32).to(self.device)
            elif key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.long).to(self.device)
            else:
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.float32).to(self.device)

        return batch, max_episode_len

class TransReplayBuffer(object):

    def __init__(self, size):
        self.size = size
        self.buffer = []

    def get_single(self, index):
        return self.buffer[index]

    def offset(self):
        self.buffer.pop(0)

    def get_batch(self, batch_size):
        length = len(self.buffer)
        indices = np.random.choice(length, batch_size, replace=False)
        batch_buffer = [self.buffer[i] for i in indices]
        return batch_buffer

    def add_experience(self, trans):
        est_len = 1 + len(self.buffer)
        if est_len > self.size:
            self.offset()
        self.buffer.append(trans)

    def clear(self):
        self.buffer = []



class EpisodeReplayBuffer(object):

    def __init__(self, size):
        self.size = size
        self.buffer = []

    def get_single(self, index):
        return self.buffer[index]

    def offset(self):
        self.buffer.pop(0)

    def get_batch(self, batch_size):
        length = len(self.buffer)
        indices = np.random.choice(length, batch_size, replace=False)
        batch_buffer = []
        for i in indices:
            batch_buffer.extend(self.buffer[i])
        return batch_buffer

    def add_experience(self, episode):
        est_len = 1 + len(self.buffer)
        if est_len > self.size:
            self.offset()
        self.buffer.append(episode)
