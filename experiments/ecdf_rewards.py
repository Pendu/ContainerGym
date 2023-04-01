import glob
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from statsmodels.distributions.empirical_distribution import ECDF
import torch
from stable_baselines3 import PPO, DQN
from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from containergym.env import ContainerEnv
from rule_based_agent import RuleBasedAgent


def episodic_reward(log_dir, config_file='1container_1press.json', overwrite_episode_length=False,
                    overwrite_timestep=False, algorithm='ppo', deterministic_policy=True, n_rollouts=15):
    """Returns a list of episodic rewards collected over n_rollouts of the agent in log_dir.

    Parameters
    ----------
    log_dir : str
        log directory in which the agent was saved
    config_file : str, optional
        name of the config file used for the environment, by default '1container_1press.json'
    overwrite_episode_length : int, optional
        if a different episode length is desired in inference, set that length here
    deterministic_policy : bool, optional
        whether to sample from deterministic policy, by default True
    """

    env = ContainerEnv.from_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../containergym/configs/" + config_file))
    if overwrite_episode_length:
        env.max_episode_length = overwrite_episode_length
    if overwrite_timestep:
        env.timestep = overwrite_timestep
    env = Monitor(env)
    if algorithm == 'ppo':
        model = PPO.load(log_dir + "best_model.zip")
    elif algorithm == 'trpo':
        model = TRPO.load(log_dir + "best_model.zip")
    elif algorithm == 'dqn':
        model = DQN.load(log_dir + "best_model.zip")
    elif algorithm == 'rulebased':
        model = RuleBasedAgent(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "../containergym/configs/" + config_file))
    # Keep track of rewards
    rewards = []

    for _ in range(n_rollouts):
        obs = env.reset()
        step = 0
        episode_length = 0
        # Run episode
        while True:
            episode_length += 1
            action, _ = model.predict(obs, deterministic=deterministic_policy)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                break
            step += 1

    return rewards


def plot_ecdf_rewards(dict_rewards, fig_name='', format='png'):
    """Plots ECDFs of episodic rewards passed as dictionary, where the key corresponds to an algorithm name.
    """

    # Plot reward ECDFs
    plt.rcParams.update({'font.size': 13})
    line_width = 2
    ls_cycler = plt.cycler(ls=['-', '-.', '--'], color=3 * ['blue'])
    plt.gca().set_prop_cycle(ls_cycler)
    for k, r in dict_rewards.items():
        ecdf = ECDF(r)
        plt.plot(ecdf.x, ecdf.y, linewidth=line_width, label=k)

    plt.title("ECDF of reward per timestep")
    plt.grid()
    plt.xlabel("Reward")
    xmin, xmax = -1, 1  # min([min(v) for v in dict_rewards.values()]), max([max(v) for v in dict_rewards.values()])
    plt.xlim(xmin, xmax)

    plt.legend(loc='best')
    plt.savefig(fig_name + '.%s' % format, dpi='figure', format=format)
    plt.show()


def main():
    # Reward ECDFs for PPO, TRPO and DQN
    fig_format = 'svg'
    paths = ["./logs_best_seeds/ppo_11containers_11presses_5_seed_3_budget_5000000_ent-coef_0.0_gamma_0.99_steps_6144",
             "./logs_best_seeds/trpo_11containers_11presses_5_seed_4_budget_5000000_gamma_0.99_steps_6144",
             "./logs_best_seeds/dqn_11containers_11presses_5_seed_10_budget_5000000_gamma_0.99"]

    dict_rewards = {}

    for path in paths:
        # Extract run and config info from path
        run_name = path.split(os.sep)[-1]
        config_name = run_name.split("_seed_")[0].split("_", maxsplit=1)[-1]
        config_file = config_name + ".json"
        algorithm = run_name.split('_')[0]
        log_dir = os.path.dirname(os.path.abspath(__file__)) + '/logs_best_seeds/' + run_name + '/'
        dict_rewards[algorithm.upper()] = episodic_reward(log_dir, config_file=config_file,
                                                          overwrite_episode_length=600, overwrite_timestep=120,
                                                          algorithm=algorithm, deterministic_policy=True, n_rollouts=15)
    # Reward ECDFs
    fig_name = 'reward_ecdfs_' + config_name
    plot_ecdf_rewards(dict_rewards, fig_name=fig_name, format=fig_format)

    # Reward ECDFs for rule-based controller
    dict_rewards = {}
    algorithm = "rulebased"
    labels = [r"$n = 5, m = 2$", r"$n = 11, m = 2$", r"$n = 11, m = 11$"]
    for i, config_file in enumerate(["5containers_2presses_2.json", "11containers_2presses_3.json",
                                     "11containers_11presses_5.json"]):
        dict_rewards[labels[i]] = episodic_reward(log_dir, config_file=config_file,
                                                  overwrite_episode_length=600,overwrite_timestep=120,
                                                  algorithm=algorithm, deterministic_policy=True, n_rollouts=15)
    fig_name = 'reward_ecdfs_rule_based_controller'
    plot_ecdf_rewards(dict_rewards, fig_name=fig_name, format=fig_format)


if __name__ == "__main__":
    main()
