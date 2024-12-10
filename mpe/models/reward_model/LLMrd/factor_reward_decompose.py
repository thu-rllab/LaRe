import requests
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prompt_template import get_prompt
import numpy as np
import argparse
import torch as th
from factor_chat_with_gpt import callgpt, hetero_callgpt
import torch
import random
from factor_reward_model import Factor_Reward_Model
from torch import nn
import copy

class FactorRewardDecomposer(nn.Module):
    def __init__(self, args):
        super(FactorRewardDecomposer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.rd_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),self.args.llm_response_dir, self.args.scenario, str(self.args.seed))
        self.id = 0
        
        self.prompt = get_prompt(args.env_name, args.scenario, factor_decomp=True)
        if not args.only_s:
            self.load_rd_functions()
            self.get_factor_num()
        agent_id_num = 0 if not self.args.obs_agent_id else self.n_agents
        print(self.factor_num, agent_id_num)
        if args.only_s:
            self.reward_model = Factor_Reward_Model(self.args.obs_dim, n_layers=self.args.factor_reward_model_layers, device = 'cuda')
        else:
            self.reward_model = Factor_Reward_Model(self.factor_num+agent_id_num, n_layers=self.args.factor_reward_model_layers, device = 'cuda')
        if 'hetero_tag' in self.args.scenario:
            self.hetero_reward_model = Factor_Reward_Model(self.factor_num_1+agent_id_num, n_layers=self.args.factor_reward_model_layers, device = 'cuda')

    
    def load_rd_functions(self):
        if not os.path.exists(self.rd_save_dir):
            os.makedirs(self.rd_save_dir)
        if 'hetero_tag' in self.args.scenario:
            while not os.path.exists(os.path.join(self.rd_save_dir, f'response_{self.id}_1.npy')):
                hetero_callgpt(self.args.env_name, self.args.scenario, self.rd_save_dir,  True, self.id, n=self.args.llm_n, port=self.args.port, seed=self.args.seed)
            self.rd_functions = []
            for i in range(2):
                rd_responses = np.load(os.path.join(self.rd_save_dir, f'response_{self.id}_{i}.npy'), allow_pickle=True)
                rd_functions = []
                for i in range(len(rd_responses)):
                    func = json.loads(rd_responses[i])['Functions']
                    rd_functions.append(func)
                self.rd_functions.append(rd_functions)
            # self.rd_functions = rd_functions
        else:
            while not os.path.exists(os.path.join(self.rd_save_dir, f'response_{self.id}.npy')):
                callgpt(self.args.env_name, self.args.scenario, self.rd_save_dir,  True, self.id, n=self.args.llm_n, port=self.args.port, seed=self.args.seed)

            rd_responses = np.load(os.path.join(self.rd_save_dir, f'response_{self.id}.npy'), allow_pickle=True)
            rd_functions = []
            for i in range(len(rd_responses)):
                func = json.loads(rd_responses[i])['Functions']
                rd_functions.append(func)
            self.rd_functions = rd_functions


    def get_factor_num(self):
        if 'hetero_tag' in self.args.scenario:
            self.factor_num = np.load(os.path.join(self.rd_save_dir, f'factor_num_{self.id}_0.npy'), allow_pickle=True)
            self.factor_num_1 = np.load(os.path.join(self.rd_save_dir, f'factor_num_{self.id}_1.npy'), allow_pickle=True)
        else:
            self.factor_num = np.load(os.path.join(self.rd_save_dir, f'factor_num_{self.id}.npy'), allow_pickle=True)
    
    def func_forward(self, obs, func_str, device):
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        bs,n_agents,t,_ = obs.shape
        array_obs = obs.reshape(-1, obs.shape[-1])
        func = func_str
        namespace = {}
        exec(func,namespace)
        evaluation_func = namespace['evaluation_func']
        factor_scores = evaluation_func(array_obs)
        cat_factor_scores = np.concatenate(factor_scores, axis=-1)#bs,nfactors
        tensor_scores = th.tensor(cat_factor_scores.reshape(bs, n_agents, t, -1)).float().to(device)
        return tensor_scores
    
    def hetero_forward(self, states, actions, episode_length,next_states=None, return_tensor_scores=False):
        n_agent_1, n_agent_2 = self.prompt.n_advs, self.prompt.n_preys
        state1, state2 = states[:,:n_agent_1], states[:,n_agent_1:]
        next_state1, next_state2 = next_states[:,:n_agent_1], next_states[:,n_agent_1:]
        func_1 = self.rd_functions[0][0]
        func_2 = self.rd_functions[1][0]

        b,na,t,d = states.shape
        if self.args.only_s:
            tensor_scores = torch.tensor(states).float().to(self.reward_model.device)
            if self.args.use_next_state:
                next_tensor_scores = torch.tensor(next_states).float().to(self.reward_model.device)
            if self.args.use_next_state:
                tensor_scores = next_tensor_scores# - tensor_scores
            tensor_scores = tensor_scores.reshape(b*na,t,-1)
            rewards = self.reward_model(tensor_scores)#bs,na,t,-1 -> bs,na,t,1
            rewards = rewards.reshape(b,na,t,-1)
            tensor_scores = tensor_scores.reshape(b,na,t,-1)
        else:
            tensor_scores1 = self.func_forward(state1[:,:,:,:-self.n_agents], func_1, self.reward_model.device)
            tensor_scores1 = torch.cat([tensor_scores1, state1[:,:,:,-self.n_agents:]], dim=-1)
            if self.args.use_next_state:
                next_tensor_scores1 = self.func_forward(next_state1[:,:,:,:-self.n_agents], func_1, self.reward_model.device)
                next_tensor_scores1 = torch.cat([next_tensor_scores1, next_state1[:,:,:,-self.n_agents:]], dim=-1)
            reward_1 = self.reward_model(tensor_scores1)

            tensor_scores2 = self.func_forward(state2[:,:,:,:-self.n_agents], func_2, self.reward_model.device)
            tensor_scores2 = torch.cat([tensor_scores2, state2[:,:,:,-self.n_agents:]], dim=-1)
            if self.args.use_next_state:
                next_tensor_scores2 = self.func_forward(next_state2[:,:,:,:-self.n_agents], func_2, self.reward_model.device)
                next_tensor_scores2 = torch.cat([next_tensor_scores2, next_state2[:,:,:,-self.n_agents:]], dim=-1)
            reward_2 = self.hetero_reward_model(tensor_scores2)
            rewards = torch.cat([reward_1, reward_2], dim=1)
            tensor_scores = next_tensor_scores1

        if return_tensor_scores:
            return rewards, tensor_scores
        else:
            return rewards
        

    def forward(self, states, actions, episode_length,next_states=None, return_tensor_scores=False):
        if 'hetero_tag' in self.args.scenario:
            return self.hetero_forward(states, actions, episode_length, next_states, return_tensor_scores)
        func = self.rd_functions[0]
        b,na,t,d = states.shape
        assert na == self.n_agents
        if self.args.only_s:
            tensor_scores = torch.tensor(states).float().to(self.reward_model.device)
            if self.args.use_next_state:
                next_tensor_scores = torch.tensor(next_states).float().to(self.reward_model.device)
        else:
            tensor_scores = self.func_forward(states, func, self.reward_model.device)
            if self.args.use_next_state:
                next_tensor_scores = self.func_forward(next_states, func, self.reward_model.device)

        if self.args.use_next_state:
            tensor_scores = next_tensor_scores# - tensor_scores
        tensor_scores = tensor_scores.reshape(b*na,t,-1)
        rewards = self.reward_model(tensor_scores)#bs,na,t,-1 -> bs,na,t,1
        rewards = rewards.reshape(b,na,t,-1)
        tensor_scores = tensor_scores.reshape(b,na,t,-1)
        if return_tensor_scores:
            return rewards, tensor_scores
        else:
            return rewards
        






