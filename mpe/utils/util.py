import numpy as np
import torch
import random
import os

import torch.linalg


def setup_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def copy_policy(s_agent, t_agent):
        print('o'*50)
        state_dict = s_agent.network.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        t_agent.network.load_state_dict(state_dict)

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def back(input):
    output = input.detach().cpu().numpy if type(input) == torch.tensor else input
    return output

def mask_select(input, input_length, device):
    '''
    select part of the trajectory according to its original length
    input: [B, N_a, T] or [B, N_a, T, E], 
    input_length: [B, N_a]
    output: [N] or [N, E], where N is the sequence length of all original trajectories
    '''

    if input.dim()==3:
        B, N_a, T = input.shape
        input = input.reshape(-1, T).to(device)
        input_length = input_length.reshape(-1, 1).to(device)
        mask = (torch.arange(T)[None, :].to(device) < input_length[:, None]).to(device).squeeze()
        output = input[mask==1]
    else:
        B, N_a, T, E = input.shape
        input = input.reshape(-1, T, E).to(device)
        input_length = input_length.reshape(-1, 1).to(device)
        mask = (torch.arange(T)[None, :].to(device) < input_length[:, None]).to(device).squeeze()
        output = input[mask==1]
    return output

def generate_shuffle_indices(n):
    indices = torch.randperm(n)
    _, back_indices = indices.sort()

    return indices, back_indices

def n_actions(action_spaces):
    """
    :param action_space: list
    :return: n_action: list
    """
    n_actions = []
    from gym import spaces
    from multiagent.environment import MultiDiscrete
    for action_space in action_spaces:
        if isinstance(action_space, spaces.discrete.Discrete):
            n_actions.append(action_space.n)
        elif isinstance(action_space, MultiDiscrete):
            total_n_action = 0
            one_agent_n_action = 0
            for h, l in zip(action_space.high, action_space.low):
                total_n_action += int(h - l + 1)
                one_agent_n_action += int(h - l + 1)
            n_actions.append(one_agent_n_action)
        else:
            raise NotImplementedError
    return n_actions

from gym import spaces

class GymWrapper(object):

    def __init__(self, env):
        self.env = env
        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space

    def __call__(self):
        return self.env

    def get_num_of_agents(self):
        return self.env.n

    def get_shape_of_obs(self):
        obs_shapes = []
        for obs in self.obs_space:
            if isinstance(obs, spaces.Box):
                obs_shapes.append(obs.shape)
        assert len(self.obs_space) == len(obs_shapes)
        return obs_shapes

    def get_output_shape_of_act(self):
        act_shapes = []
        for act in self.act_space:
            if isinstance(act, spaces.Discrete):
                act_shapes.append(act.n)
            elif isinstance(act, spaces.MultiDiscrete):
                act_shapes.append(act.high - act.low + 1)
            elif isinstance(act, spaces.Boxes):
                assert act.shape == 1
                act_shapes.append(act.shape)
        return act_shapes

    def get_dtype_of_obs(self):
        return [obs.dtype for obs in self.obs_space]

    def get_input_shape_of_act(self):
        act_shapes = []
        for act in self.act_space:
            if isinstance(act, spaces.Discrete):
                act_shapes.append(act.n)
            elif isinstance(act, spaces.MultiDiscrete):
                act_shapes.append(act.shape)
            elif isinstance(act, spaces.Boxes):
                assert act.shape == 1
                act_shapes.append(act.shape)
        return act_shapes

def make_env(scenario_name, discrete_action_input=True, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.info if scenario_name.startswith('simple_tag') else None,
                            seed_callback=scenario.seed, cam_range=scenario.world_radius, discrete_action_input=discrete_action_input)
    # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=True, 
    #                         done_callback=scenario.episode_over if scenario_name=='simple_tag' else None, discrete_action_input=discrete_action_input)
    # env = GymWrapper(env)
    # if benchmark:
    #     env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,
    #                         seed_callback=scenario.seed, cam_range=scenario.world_radius)
    # else:
        # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
        #                     seed_callback=scenario.seed, cam_range=scenario.world_radius)
    return env

def mask_mse(input, target, mask):
    mse_loss = torch.square(input-target)*mask
    mse_loss = torch.sum(mse_loss)/torch.sum(mask)
    return mse_loss

def make_train_step(model, loss_fn, optimizer, n_agent, device, reg=False, alpha=None):
    # Builds function that performs a step in the train loop
    if reg: assert alpha is not None
    def train_step(states, actions, episode_return, episode_length, next_states=None):
        #states: b,na,t,d
        # actions: b,na,t
        # episode_return: b
        # episode_length: b
        # Sets model to TRAIN mode
        model.train()

        # Makes predictions: pred_return: [batch_size, n_agent, time length]
        if next_states is not None:
            pred_rewards = model(states, actions, episode_length, next_states).squeeze()
        else:
            pred_rewards = model(states, actions, episode_length).squeeze()
        # Computes loss

        main_mask = (torch.arange(pred_rewards.shape[-1])[None, :].to(device) <= (episode_length-1).reshape(-1,1).squeeze()[:, None]).to(device).squeeze()
        main_mask = main_mask.unsqueeze(1).repeat(1,n_agent,1)
        loss = loss_fn(episode_return, (pred_rewards*main_mask).sum(dim=[1, 2]))

        if reg:
            pred_rewards = pred_rewards.sum(dim=1)
            main_mask = main_mask[:,0,:]
            pred_rewards_mean = torch.sum(pred_rewards*main_mask, dim=-1)/torch.sum(main_mask, dim=-1)
            pred_rewards_mean = pred_rewards_mean.unsqueeze(-1).repeat(1,pred_rewards.shape[-1])
            pred_rewards_std = torch.square(pred_rewards-pred_rewards_mean)
            reg_loss = torch.mean(torch.sum(pred_rewards_std*main_mask, dim=-1)/torch.sum(main_mask, dim=-1))

            loss += alpha*reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return train_step

def make_arel_train_step(model, loss_fn, optimizer, n_agent, device, reg=False, alpha=None):
    # Builds function that performs a step in the train loop
    def train_step(states, actions, episode_return, episode_length, next_states=None):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat, y_time_hat = model(next_states)
        # Computes loss
        loss = loss_fn(episode_return, yhat)
        loss_3 = 20 * torch.mean(torch.var(y_time_hat, dim=1))
        
        loss_total = loss + loss_3
        # Computes gradients
        loss_total.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return train_step

def make_rrd_train_step(args, model, loss_fn, optimizer, n_agent, device, reg=False, alpha=None):
    # Builds function that performs a step in the train loop
    if reg: assert alpha is not None
    def train_step(states, actions, episode_return, episode_length, next_states=None):
        model.train()

        # Makes predictions: pred_return: [batch_size, n_agent, time length]
        if next_states is not None:
            pred_rewards, sampled_rewards_var = model(states, actions, episode_length, next_states,only_inference=False)
            pred_rewards = pred_rewards.squeeze()
        else:
            pred_rewards, sampled_rewards_var = model(states, actions, episode_length,only_inference=False)
            pred_rewards = pred_rewards.squeeze()
        # Computes loss

        main_mask = (torch.arange(pred_rewards.shape[-1])[None, :].to(device) <= (episode_length-1).reshape(-1,1).squeeze()[:, None]).to(device).squeeze()
        main_mask = main_mask.unsqueeze(1).repeat(1,n_agent,1)
        loss = loss_fn(episode_return, (pred_rewards*main_mask).sum(dim=[1, 2]))
        if args.rrd_unbiased:
            loss -= sampled_rewards_var

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return train_step

def make_IB_train_step(args, model, loss_fn, optimizer, n_agent, device, reg=False, alpha=None):
    # Builds function that performs a step in the train loop
    if reg: assert alpha is not None
    def train_step(states, actions, episode_return, episode_length, next_states=None):
        # Sets model to TRAIN mode
        b,n,t,d = states.shape
        model.train()

        # Makes predictions: pred_return: [batch_size, n_agent, time length]
        input_episode_return = episode_return.unsqueeze(-1).unsqueeze(-1).repeat(1,n,t).unsqueeze(-1).to(states.device)/(n*t)
        if next_states is not None:
            pred_rewards, kl_loss = model(states, actions, input_episode_return, next_states, return_kl=True)
            pred_rewards = pred_rewards.squeeze()
        else:
            pred_rewards, kl_loss = model(states, actions, input_episode_return, return_kl=True)
            pred_rewards = pred_rewards.squeeze()
        # Computes loss

        main_mask = (torch.arange(pred_rewards.shape[-1])[None, :].to(device) <= (episode_length-1).reshape(-1,1).squeeze()[:, None]).to(device).squeeze()
        main_mask = main_mask.unsqueeze(1).repeat(1,n_agent,1)
        loss = loss_fn(episode_return, (pred_rewards*main_mask).sum(dim=[1, 2]))
        loss += args.kl_weight*kl_loss

        if reg:
            pred_rewards = pred_rewards.sum(dim=1)
            main_mask = main_mask[:,0,:]
            pred_rewards_mean = torch.sum(pred_rewards*main_mask, dim=-1)/torch.sum(main_mask, dim=-1)
            pred_rewards_mean = pred_rewards_mean.unsqueeze(-1).repeat(1,pred_rewards.shape[-1])
            pred_rewards_std = torch.square(pred_rewards-pred_rewards_mean)
            reg_loss = torch.mean(torch.sum(pred_rewards_std*main_mask, dim=-1)/torch.sum(main_mask, dim=-1))

            loss += alpha*reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return train_step


def from_return_to_reward(pred_return, device):
    b, t, n_a = pred_return.shape
    agent_reward = pred_return.reshape(b, -1)
    _agent_reward = agent_reward[:, :-1]
    padding = torch.zeros(b,1).to(device)
    _agent_reward = torch.cat((padding, _agent_reward), dim=1)
    pred_agent_reward = (agent_reward-_agent_reward).reshape(b, t, n_a).permute(0, 2, 1)

    return pred_agent_reward

def draw_reward_statistics(exp_name, writer, reward_statistics, n_agent, i_episode):
    # predicted: [B, N_a, T], real: [B, N_a, T]
    predicted, real, _ = reward_statistics
    predicted = torch.from_numpy(predicted)
    real = torch.from_numpy(real)
    # wandb.log({'predicted_all_rewards': predicted, 'predicted_rewards_mean': torch.mean(predicted), \
    #     'predicted_all_step_rewards': torch.sum(predicted, dim=1), 'real_all_step_rewards': real[:,0,:], \
    #     'real_step_rewards_mean': torch.mean(real[:,0,:])})

    # draw total statistics
    writer.add_histogram(exp_name + '_predicted_all_rewards', predicted, i_episode)
    writer.add_scalar(exp_name + '_predicted_rewards_mean', torch.mean(predicted), i_episode)
    writer.add_scalar(exp_name + '_predicted_rewards_var', torch.var(predicted), i_episode)

    writer.add_histogram(exp_name + '_predicted_all_step_rewards', torch.sum(predicted, dim=1), i_episode)
    writer.add_scalar(exp_name + '_predicted_step_rewards_mean', torch.mean(torch.sum(predicted, dim=1)), i_episode)
    writer.add_scalar(exp_name + '_predicted_step_rewards_var', torch.var(torch.sum(predicted, dim=1)), i_episode)
    writer.add_histogram(exp_name + '_real_all_step_rewards', real[:,0,:], i_episode)
    writer.add_scalar(exp_name + '_real_step_rewards_mean', torch.mean(real[:,0,:]), i_episode)
    writer.add_scalar(exp_name + '_real_step_rewards_var', torch.var(real[:,0,:]), i_episode)

    # draw along agent
    for i in range(n_agent):
        writer.add_histogram(exp_name + '_predicted_agent_{}_reward'.format(i), predicted[:,i,:], i_episode)

    # draw along time
    writer.add_histogram(exp_name + '_predicted_first_5_steps', predicted[:,:,:5], i_episode)
    writer.add_histogram(exp_name + '_predicted_last_5_steps', predicted[:,:,-5:], i_episode)

def rewards_process(decomposed_r, real_r, info):
    # ipdb.set_trace()
    info[info<0] = 0
    key_epi = info.sum((1,2))>0
    if key_epi.any() == False:
        return None, None, None
    decomposed_r = decomposed_r[key_epi]
    real_r = real_r[key_epi]
    info = info[key_epi]

    idx = np.random.choice(decomposed_r.shape[0], 1)
    decomposed_r = np.squeeze(decomposed_r[idx], axis=0)
    real_r = np.squeeze(real_r[idx], axis=0)
    info = np.squeeze(info[idx], axis=0)

    return decomposed_r, real_r, info