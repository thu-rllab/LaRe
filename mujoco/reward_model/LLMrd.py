import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import wandb
from torch.distributions import Beta, Normal
from .module import Reward_Model
from .chat_with_gpt import callgpt
from prompt_template import Mujoco_prompt
import os
import json
from .rrd import RRDRewardDecomposer
from .RD import RDRewardDecomposer

class LLMRewardDecomposer(RRDRewardDecomposer):
    def __init__(self, args):
        super(LLMRewardDecomposer, self).__init__(args)
        self.args = args
        self.K = self.args.rrd_k
        self.rd_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),self.args.llm_response_dir, self.args.env, str(self.args.seed))
        self.id = 0
        self.prompt = Mujoco_prompt(self.args.env)
        self.load_rd_functions()
        self.get_factor_num()

        self.reward_model = Reward_Model(input_dim=self.factor_num)
        self.reward_model.to('cuda')

        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=3e-4)
        self.loss_fn = nn.MSELoss(reduction='mean')

    def load_rd_functions(self):
        if not os.path.exists(self.rd_save_dir):
            os.makedirs(self.rd_save_dir)
        while not os.path.exists(os.path.join(self.rd_save_dir, f'response_{self.id}.npy')):
            callgpt(self.args.env, self.rd_save_dir,  True, self.id, n=self.args.llm_n, port=self.args.port, seed=self.args.seed, factor=(not self.args.direct_generate))

        rd_responses = np.load(os.path.join(self.rd_save_dir, f'response_{self.id}.npy'), allow_pickle=True)
        rd_functions = []
        for i in range(len(rd_responses)):
            func = json.loads(rd_responses[i])['Functions']
            rd_functions.append(func)
        self.rd_functions = rd_functions

    def get_factor_num(self):
        self.factor_num = np.load(os.path.join(self.rd_save_dir, f'factor_num_{self.id}.npy'), allow_pickle=True)

    def func_forward(self, states, actions, next_states):
        func = self.rd_functions[0]
        device = states.device
        raw_shape = states.shape
        states = states.cpu().numpy().reshape(-1,states.shape[-1])
        actions = actions.cpu().numpy().reshape(-1,actions.shape[-1])
        next_states = next_states.cpu().numpy().reshape(-1,states.shape[-1])
        namespace = {}
        exec(func,namespace)
        evaluation_func = namespace['evaluation_func']
        factor_scores = evaluation_func(states, actions)#, next_states)
        cat_factor_scores = np.concatenate(factor_scores, axis=-1)#bs,nfactors
        if len(raw_shape) == 3:
            tensor_scores = torch.FloatTensor(cat_factor_scores).to(device).reshape(raw_shape[0], raw_shape[1], -1)
        else:
            tensor_scores = torch.FloatTensor(cat_factor_scores).to(device).reshape(raw_shape[0], -1)
        return tensor_scores
    
    def forward(self, states, actions, next_states):
        states = self.func_forward(states, actions, next_states)
        if self.args.direct_generate:
            rewards = states
        else:
            rewards = self.reward_model(states)#bs,t,-1 -> bs,t,1
        return rewards

class LLMRDRewardDecomposer(RDRewardDecomposer):
    def __init__(self, args):
        super(LLMRDRewardDecomposer, self).__init__(args)
        self.args = args
        self.K = self.args.rrd_k
        self.rd_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),self.args.llm_response_dir, self.args.env, str(self.args.seed))
        self.id = 0
        self.prompt = Mujoco_prompt(self.args.env)
        self.load_rd_functions()
        self.get_factor_num()

        self.reward_model = Reward_Model(input_dim=self.factor_num)
        self.reward_model.to('cuda')

        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=3e-4)
        self.loss_fn = nn.MSELoss(reduction='mean')

    def load_rd_functions(self):
        if not os.path.exists(self.rd_save_dir):
            os.makedirs(self.rd_save_dir)
        while not os.path.exists(os.path.join(self.rd_save_dir, f'response_{self.id}.npy')):
            callgpt(self.args.env, self.rd_save_dir,  True, self.id, n=self.args.llm_n, port=self.args.port, seed=self.args.seed)

        rd_responses = np.load(os.path.join(self.rd_save_dir, f'response_{self.id}.npy'), allow_pickle=True)
        rd_functions = []
        for i in range(len(rd_responses)):
            func = json.loads(rd_responses[i])['Functions']
            rd_functions.append(func)
        self.rd_functions = rd_functions

    def get_factor_num(self):
        self.factor_num = np.load(os.path.join(self.rd_save_dir, f'factor_num_{self.id}.npy'), allow_pickle=True)

    def func_forward(self, states, actions, next_states):
        func = self.rd_functions[0]
        device = states.device
        raw_shape = states.shape
        states = states.cpu().numpy().reshape(-1,states.shape[-1])
        actions = actions.cpu().numpy().reshape(-1,actions.shape[-1])
        next_states = next_states.cpu().numpy().reshape(-1,states.shape[-1])


        namespace = {}
        exec(func,namespace)
        evaluation_func = namespace['evaluation_func']
        factor_scores = evaluation_func(states, actions)#, next_states)
        cat_factor_scores = np.concatenate(factor_scores, axis=-1)#bs,nfactors
        if len(raw_shape) == 3:
            tensor_scores = torch.FloatTensor(cat_factor_scores).to(device).reshape(raw_shape[0], raw_shape[1], -1)
        else:
            tensor_scores = torch.FloatTensor(cat_factor_scores).to(device).reshape(raw_shape[0], -1)
        return tensor_scores

    def forward(self, states, actions, next_states):
        states = self.func_forward(states, actions, next_states)
        rewards = self.reward_model(states)#bs,t,-1 -> bs,t,1
        return rewards