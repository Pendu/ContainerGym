import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import argparse
from distutils.util import strtobool
from multiprocessing import Process

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor

from containergym.env import ContainerEnv
from containergym.experiments.plot_avg_cum_rew_n_rollouts import *

os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(1)


def parse_args():

    """
    The parse_args function is used to parse the command line arguments.

    :return: A namespace object that contains the arguments passed to the script
    """
    parser = argparse.ArgumentParser()

    # Experiment specific arguments
    parser.add_argument(
        "--config-file",
        type=str,
        default="1bunker_1press.json",
        help="The name of the config file for the env",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=10000,
        help="total number of timesteps of the experiments",
    )
    parser.add_argument(
        "--inf-eplen",
        type=int,
        default=600,
        help="total number of timesteps of the experiments (will be overwritten if it differs from the timesteps set from config file)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="total number of timesteps of the experiments",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=4,
        help="total number of seeds to run the experiment",
    )
    parser.add_argument(
        "--RL-agent",
        type=str,
        default="PPO",
        help="The name of the agent to train the env",
    )
    parser.add_argument(
        "--ent-coeff",
        type=float,
        default=0.0,
        help="Entropy coefficient for the loss calculation",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    args = parser.parse_args()

    return args


def average_cumulative_reward(log_dir = None , config_file=None, overwrite_episode_length= None, algorithm= None, n_rollouts=15, deterministic_policy=True):
    """Performs n_rollouts rollouts of a policy and prints the average cumulative reward and the standard deviation.
    """

    env = ContainerEnv.from_json(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../configs/" + config_file
        )
    )
    if args.inf_eplen:
        env.max_episode_length = args.inf_eplen

    env = Monitor(env)

    if args.RL_agent == "PPO":
        model = PPO.load(log_dir + "best_model.zip")
    elif args.RL_agent == "TRPO":
        model = TRPO.load(log_dir + "best_model.zip")
    elif args.RL_agent == "A2C":
        model = A2C.load(log_dir + "best_model.zip")
    elif args.RL_agent == "DQN":
        model = DQN.load(log_dir + "best_model.zip")
    elif args.RL_agent == "rulebased":
        model = rulebased_agent()
        peak_vols = model.set_env(env)


    # Keep track of volumes, actions and rewards
    volumes = []
    actions = []
    cumulative_rewards = []

    # Perform rollouts
    for _ in range(n_rollouts):
        obs = env.reset()
        volumes.append(obs["Volumes"].copy())  # Save initial volumes
        step = 0
        episode_length = 0
        cumulative_reward = 0

        while True:
            episode_length += 1
            action, _ = model.predict(obs, deterministic=deterministic_policy)
            actions.append(action)
            obs, reward, done, info = env.step(action)
            if done:
                break
            else:
                volumes.append(obs["Volumes"].copy())  # Save all volumes but the last ones

            cumulative_reward += reward
            step += 1

        cumulative_rewards.append(cumulative_reward)

    print("Average cumul. reward: {:.2f} \u00B1 {:.2f}".format(np.mean(cumulative_rewards), np.std(cumulative_rewards)))


if __name__ == '__main__':

    # get the arguments
    args = parse_args()

    for seed in range(1, args.n_seeds + 1):
        run_name = f"{args.RL_agent}_{args.config_file.replace('.json', '')}_seed_{seed}_budget_{args.budget}_ent-coef_{args.ent_coeff}_gamma_{args.gamma}_n_steps_{args.n_steps}"
        log_dir = os.path.dirname(os.path.abspath(__file__)) + '/logs/' + run_name + '/'
        average_cumulative_reward(log_dir = log_dir, config_file=args.config_file, overwrite_episode_length=args.inf_eplen,
                                  algorithm=args.RL_agent, n_rollouts=15,
                                  deterministic_policy=True)