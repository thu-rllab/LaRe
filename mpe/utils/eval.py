import numpy as np
import contextlib
import torch
import os
import csv
import time
from utils.util import make_env
from arguments import *
# from envs.alice_and_bob.alice_and_bob import Alice_and_Bob

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def dict2csv(output_dict, f_name):
    with open(f_name, mode='w') as f:
        writer = csv.writer(f, delimiter=",")
        for k, v in output_dict.items():
            v = [k] + v
            writer.writerow(v)

def eval_policy(test_q, done_training, args):
    plot = {'rewards': [], 'steps': [], 'p_loss': [], 'final': [], 'abs': [], 'episodes': []}
    best_eval_reward = -np.inf
    while True:
        if not test_q.empty():
            print('=================== start eval ===================')
            eval_env = None
            if args.env_name == 'Alice_and_Bob' or args.env_name == 'Alice_and_Bob_simple':
                np.random.seed(args.seed + 10)
                torch.manual_seed(args.seed + 10)
                if args.env_name == 'Alice_and_Bob':
                    from envs.alice_and_bob.alice_and_bob_original import Alice_and_Bob
                    obs_dim = 13   # The dimensions of an agent's observation space
                else:
                    from envs.alice_and_bob.alice_and_bob_simple import Alice_and_Bob
                    obs_dim = 19   # The dimensions of an agent's observation space
                
                args.state_dim = 13 * 9

                eval_env = Alice_and_Bob()

                n_action = 4   # The dimensions of an agent's action space
            elif args.env_name == 'MPE':
                eval_env = make_env(args.scenario)
                n_action = eval_env.n
                # eval_env.seed(args.seed + 10)
            
            eval_rewards = []
            good_eval_rewards = []
            agent, tr_log = test_q.get()
            with temp_seed(args.seed):
                for n_eval in range(args.num_eval_runs):
                    obs_n = eval_env.reset()
                    if args.state_norm:
                        obs_n = agent.state_norm(obs_n)
                    episode_reward = 0
                    episode_step = 0
                    n_agents = n_action
                    # if args.env_name == 'Alice_and_Bob' or args.env_name == 'Alice_and_Bob_simple':
                    #     n_agents = 4
                    # elif args.env_name == 'MPE':
                    #     n_agents = eval_env.n
                    agents_rew = [[] for _ in range(n_agents)]
                    while True:
                        action_n, logproba_n, value_n = agent.select_action(torch.Tensor(obs_n).to(device))
                        next_obs_n, reward_n, done_n, _ = eval_env.step(action_n)
                        
                        if args.env_name == 'Alice_and_Bob' or args.env_name == 'Alice_and_Bob_simple':
                            reward_n = [reward_n for na in range(n_agents)]
                            done_n = [done_n for na in range(n_agents)]
                            # info_n = {'n': info_n}

                        if args.state_norm:
                            next_obs_n = agent.state_norm(next_obs_n)

                        episode_step += 1
                        episode_reward += np.sum(reward_n)
                        for i, r in enumerate(reward_n):
                            agents_rew[i].append(r)
                        obs_n = next_obs_n
                        if done_n[0] or episode_step >= args.max_step_per_round-1:
                            eval_rewards.append(episode_reward)
                            agents_rew = [np.sum(rew) for rew in agents_rew]
                            good_reward = np.sum(agents_rew)
                            good_eval_rewards.append(good_reward)
                            if n_eval % 100 == 0:
                                print('test reward', episode_reward)
                            break
                if np.mean(eval_rewards) > best_eval_reward:
                    best_eval_reward = np.mean(eval_rewards)
                    torch.save({'agents': agent}, os.path.join(tr_log['model_save_path'], 'agents_best.ckpt'))

                plot['rewards'].append(np.mean(eval_rewards))
                plot['steps'].append(tr_log['total_numsteps'])
                plot['episodes'].append(tr_log['i_episode'])
                plot['p_loss'].append(tr_log['policy_loss'])
                print("========================================================")
                print(
                    "Episode: {}, total numsteps: {}, {} eval runs, total time: {} s".
                        format(tr_log['i_episode'], tr_log['total_numsteps'], args.num_eval_runs,
                               time.time() - tr_log['start_time']))
                print("GOOD reward: avg {} std {}, average reward: {}, best reward {}".format(np.mean(eval_rewards),
                                                                                              np.std(eval_rewards),
                                                                                              np.mean(plot['rewards'][
                                                                                                      -10:]),
                                                                                              best_eval_reward))
                plot['final'].append(np.mean(plot['rewards'][-10:]))
                plot['abs'].append(best_eval_reward)
                dict2csv(plot, os.path.join(tr_log['log_save_path'], 'train_curve.csv'))
                # eval_env.close()
        if args.method == 'STAS':
            if done_training.value and test_q.empty():
                torch.save({'agents': agent}, os.path.join(tr_log['model_save_path'], 'agents.ckpt'))
                # test_q.terminate()
                # print('stop process')
                # test_q.join()
                break
        else:
            if test_q.empty():
                torch.save({'agents': agent}, os.path.join(tr_log['model_save_path'], 'agents.ckpt'))
                # test_q.terminate()
                # print('stop process')
                # test_q.join()
                break
                