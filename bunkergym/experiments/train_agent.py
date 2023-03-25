import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from datetime import datetime
import torch
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from multiprocessing import Process
from bunkergym.experiments.callbacks import *
from bunkergym.env import SutcoEnv
from gym.wrappers import FlattenObservation
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment specific arguments
    parser.add_argument("--config-file", type=str, default='1bunker_1press.json',
                        help="The name of the config file for the env")
    parser.add_argument("--budget", type=int, default=10000,
                        help="total number of timesteps of the experiments")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="total number of timesteps of the experiments")
    parser.add_argument("--n-seeds", type=int, default=4,
                        help="total number of seeds to run the experiment")
    parser.add_argument("--RL-agent", type=str, default='PPO',
                        help="The name of the agent to train the env")
    args = parser.parse_args()

    return args


def train(seed,args):
    
    """
    Trains an agent on the benchmark environment.

    Attributes
    ----------
    seed : int
        Seed used for the experiment
    args : argparse.Namespace
        Arguments passed to the script
    """
    
    config_file = args.config_file
    budget = args.budget
    ent_coef = 0.
    gamma = 0.99
    n_steps = args.n_steps

    name = f"{args.RL_agent}_" + config_file.replace(".json", "") \
           + '_seed_' + str(seed) + '_budget_' + str(budget) + "_n_steps_" + str(n_steps)

    log_dir = os.path.dirname(os.path.abspath(__file__)) + '/logs/' + name + '/'
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = SutcoEnv.from_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../configs/" + config_file))
    env = FlattenObservation(env)
    env = Monitor(env, log_dir)

    # Create the callback: check every 1000 steps
    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    # model = PPO("MultiInputPolicy", env, seed=seed, verbose=0, ent_coef=ent_coef, gamma=gamma, n_steps=n_steps)
    if args.RL_agent == 'PPO':
        model = PPO("MlpPolicy", env, seed=seed, verbose=0, ent_coef=ent_coef, gamma=gamma, n_steps=n_steps)
    elif args.RL_agent == 'TRPO':
        model = TRPO("MlpPolicy", env, seed=seed, verbose=0, n_steps=n_steps)
    elif args.RL_agent == 'A2C':
        model = A2C("MlpPolicy", env, seed=seed, verbose=0, n_steps = n_steps)
    elif args.RL_agent == 'DQN':
        model = DQN("MlpPolicy", env, seed=seed, verbose=0)

    start = datetime.now()

    model.learn(total_timesteps=budget, callback=auto_save_callback, tb_log_name="seed_" + str(seed))

    print("Total training time: ", datetime.now() - start)

    # Plot Training reward
    results_plotter.plot_results([log_dir], budget, results_plotter.X_TIMESTEPS, f"{args.RL_agent}")

    del model
    if args.RL_agent == 'PPO':
        model = PPO.load(log_dir + "best_model.zip", env=env)
    elif args.RL_agent == 'TRPO':
        model = TRPO.load(log_dir + "best_model.zip", env=env)
    elif args.RL_agent == 'A2C':
        model = A2C.load(log_dir + "best_model.zip", env=env)
    elif args.RL_agent == 'DQN':
        model = DQN.load(log_dir + "best_model.zip", env=env)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic=True)
    print("Average episodic reward: {:} \u00B1 {:}".format(mean_reward, std_reward))
    
if __name__ == "__main__":
    
    # get the arguments
    args = parse_args()
    
    seeds = range(1, args.n_seeds + 1)
    processes = [Process(target=train, args=(s, args)) for s in seeds]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
