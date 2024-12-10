import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import wandb
import utils
import algos.TD3 as TD3
import algos.DDPG as DDPG
import algos.SAC as SAC
import algos.PPO as PPO
from algos.PPO import Normalization, RewardScaling
torch.set_num_threads(5)
# os.environ["WANDB_MODE"] = "offline"
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, state_norm=None):
	eval_env = gym.make(env_name)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, _ = eval_env.reset()
		if state_norm is not None:
			state = state_norm(state, update=False)
		done=False
		while not done:
			action = policy.select_action(np.array(state), deterministic=True)
			state, reward, terminated, truncated,_ = eval_env.step(action)
			done = terminated or truncated
			if state_norm is not None:
				state = state_norm(state, update=False)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="PPO")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="HalfCheetah-v4")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=0, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment

	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument('--buffer_size', type=int,default=1000000)   
	parser.add_argument("--adaptive_alpha", default=True, action="store_false")  #SAC alpha
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--save_data", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument('--target_update_freq', type=int,default=2)

	parser.add_argument("--dense_r", default=False, action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--rrd_unbiased", default=False, action="store_true")        # rrd_unbiased
	parser.add_argument("--rd_method", default='None')        # Save model and optimizer parameters #IRCR, RRD, RRD_unbiased, VIB, RD, LLMrd
	parser.add_argument("--exp_name", default='None')       
	parser.add_argument("--run_name", default='None')        
	parser.add_argument("--rrd_k", default=64, type=int)       #from rrd paper     

	parser.add_argument("--llm_response_dir", type=str,default='factor_reward_decomp_response_data')           
	parser.add_argument('--llm_n', type=int,default=5) #n==1, no llm self-prompting 
	parser.add_argument('--port', type=int,default=8000)      
	
	parser.add_argument("--direct_generate", default=False, action="store_true")  #use llm reward design
	
	parser.add_argument("--state_norm", default=True, action="store_false")    
	parser.add_argument("--reward_scale", default=False, action="store_false")    
	      
	args = parser.parse_args()
	# args.max_timesteps = 1e6

	if args.rd_method == 'IRCR':
		if args.env == 'Humanoid-v4':
			args.adaptive_alpha = False
		args.buffer_size = 300000
		args.tau = 0.001
		args.batch_size = 512
	if args.policy=='PPO':
		args.batch_size = 2048

	if 'unbiased' in args.rd_method:
		args.rrd_unbiased = True
	else:
		args.rrd_unbiased = False

	if args.exp_name == 'None':
		args.exp_name = args.rd_method + '_' + args.policy
	if args.run_name == 'None':
		args.run_name = args.exp_name+'_'+args.env+'_'+str(args.seed)

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	wandb.init(project = 'RD_MUJOCO', config=vars(args), name=getattr(args, 'run_name', None))


	env = gym.make(args.env)
	# Set seeds
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	max_length = env._max_episode_steps
	args.obs_dim = state_dim
	args.action_dim = action_dim
	args.max_length = max_length
	args.max_action = max_action

	kwargs = {
		"args": args,
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	# if args.policy == "TD3":
	# 	# Target policy smoothing is scaled wrt the action scale
	# 	kwargs["policy_noise"] = args.policy_noise * max_action
	# 	kwargs["noise_clip"] = args.noise_clip * max_action
	# 	kwargs["policy_freq"] = args.policy_freq
	# 	policy = TD3.TD3(**kwargs)
	# elif args.policy == "DDPG":
	# 	policy = DDPG.DDPG(**kwargs)
	# elif args.policy == "SAC":
	# 	policy = SAC.SAC(**kwargs)
	assert args.policy == "PPO":
	policy = PPO.PPO(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	if args.rd_method != 'IRCR':
		replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_length, args.buffer_size, log_prob=(args.policy == "PPO"))
	else:
		replay_buffer = utils.IRCRReplayBuffer(state_dim, action_dim, max_length, args.buffer_size)

	old_replay_buffer = PPO.ReplayBuffer(state_dim, action_dim, args.batch_size+1)
	state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
	reward_scaling = RewardScaling(shape=1, gamma=0.99)
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed, state_norm=state_norm if args.state_norm else None)]
	wandb.log({"eval_reward": evaluations[0], "t":0})
	
	

	state, _ = env.reset()
	raw_state = state
	if args.state_norm:
		state = state_norm(state)
	reward_scaling.reset()
	done = False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	last_train = 0
	states, actions, next_states, rewards, dense_rewards, dones, a_log_probs = [], [], [], [], [], [], []
	raw_states, raw_next_states = [], []

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		action, log_prob = policy.select_action(np.array(state), log_prob=True)

		# Perform action
		next_state, reward, terminated, truncated, info = env.step(action) 
		raw_next_state = next_state
		if args.state_norm:
			next_state = state_norm(next_state)
		raw_reward = reward
		if args.reward_scale:
			reward = reward_scaling(reward)[0]
		done = terminated or truncated
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		old_replay_buffer.add(state, action, log_prob, next_state, 0 if not done else episode_reward, reward, done_bool, done)
		states.append(state)
		raw_states.append(raw_state)
		actions.append(action)
		next_states.append(next_state)
		raw_next_states.append(raw_next_state)
		rewards.append([0 if not done else episode_reward])
		dense_rewards.append([reward])
		dones.append([done_bool])
		a_log_probs.append(log_prob)
		

		# Store data in replay buffer
		episode_reward += raw_reward
		state = next_state
		raw_state = raw_next_state
		

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			total_loss_log = {}
			if replay_buffer.size >= 10e3:
				reward_model_loss = policy.train_reward(replay_buffer)
				total_loss_log["reward_model_loss"] = reward_model_loss
			if old_replay_buffer.size >= args.batch_size or args.policy != 'PPO':
				loss_log = policy.train(old_replay_buffer, replay_buffer, args.batch_size)
				if args.policy == 'PPO':
					old_replay_buffer.size = 0
				loss_log["t"] = t
				for key in loss_log:
					if key not in total_loss_log:
						total_loss_log[key] = 0
					total_loss_log[key] += loss_log[key]
				if t-last_train >= 100:
					wandb.log(total_loss_log)
					last_train = t

		if done: 
			replay_buffer.add_traj(np.array(raw_states), np.array(actions), np.array(raw_next_states), np.array(rewards), np.array(dense_rewards), np.array(dones), episode_reward, episode_timesteps, np.array(a_log_probs) if args.policy == 'PPO' else None)

			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			wandb.log({"train_reward": episode_reward, "episode_steps": episode_timesteps, "t":t, "episodes":episode_num})
			# Reset environment
			state, _ = env.reset()
			raw_state = state
			if args.state_norm:
				state = state_norm(state)
			reward_scaling.reset()
			done = False
			states, actions, next_states, rewards, dense_rewards, dones, a_log_probs = [], [], [], [], [], [], []
			raw_states, raw_next_states = [], []
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed, state_norm=state_norm if args.state_norm else None))
			wandb.log({"eval_reward": evaluations[-1], "t":t})
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")
	if args.save_data:
		replay_buffer.save_data(f"./results/{file_name}_buffer.npz")
