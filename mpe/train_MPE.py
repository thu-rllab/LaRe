from runners.MPE_runner import *
from utils.util import setup_seed, make_env, n_actions
import sys
command = ' '.join(sys.argv)
from arguments import *
import wandb
import numpy as np
import torch
torch.set_num_threads(5)

if args.seed==-1:
    args.seed = np.random.randint(0, 1000000)
setup_seed(args.seed)
if args.run_name is not None:
    args.run_name = args.run_name + '_'+str(args.seed)

env = make_env(args.scenario, discrete_action_input=False if args.method_name=='SQDDPG' else True)
n_agents = env.n
n_action = n_actions(env.action_space)[0]
obs_dim = env.observation_space[0].shape[0]
args.n_agents = n_agents

if 'hetero_tag' in args.scenario:
    args.obs_agent_id = True
    obs_dim = obs_dim + n_agents

if args.method_name == 'RRD':
    args.rrd_unbiased = False
if args.method_name == 'RRD_unbiased':
    args.rrd_unbiased = True
if args.method_name == 'RD':
    args.only_s = True

if args.method_name == 'PPO':
    args.return_decomposition = False
    args.dense_r = False
if args.method_name == 'PPO_dense':
    args.return_decomposition = False
    args.dense_r = True

if args.wandb:
    wandb.init(project = args.wandb_project, group = args.env_name + '-' + args.scenario + '-' + args.method_name, config=vars(args), name=getattr(args, 'run_name', None))

if args.method_name == 'STAS':
    args.use_next_state = False
    runner = Runner_STAS_MPE(args, args.env_name, env, n_agents, n_action, obs_dim, device)
elif args.method_name == 'AREL':
    args.use_next_state = True
    runner = Runner_AREL_MPE(args, args.env_name, env, n_agents, n_action, obs_dim, device)
elif args.method_name == 'LaRe' or args.method_name == 'RD':
    args.use_next_state = True
    runner = Runner_Factor_LLMrd_MPE(args, args.env_name, env, n_agents, n_action, obs_dim, device)
elif args.method_name == 'RRD' or args.method_name == 'RRD_unbiased':
    args.use_next_state = True
    runner = Runner_RRD_MPE(args, args.env_name, env, n_agents, n_action, obs_dim, device)

elif args.method_name == 'PPO':
    runner = Runner_PPO_MPE(args, args.env_name, env, n_agents, n_action, obs_dim, device)
elif args.method_name == 'PPO_dense':
    runner = Runner_PPO_MPE(args, args.env_name, env, n_agents, n_action, obs_dim, device)

elif args.method_name == 'VIB':
    runner = Runner_VIB_MPE(args, args.env_name, env, n_agents, n_action, obs_dim, device)

runner.run()