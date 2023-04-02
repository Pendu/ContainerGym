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


def episodic_reward(
    log_dir,
    config_file="1container_1press.json",
    overwrite_episode_length=False,
    overwrite_timestep=False,
    algorithm="ppo",
    deterministic_policy=True,
    n_rollouts=15,
    include_action_zero=False
):
    """
    Returns a list of episodic rewards collected over n_rollouts of the agent in log_dir.

    Parameters:
    -----------
    log_dir: str
        log directory in which the agent was saved
    config_file: str, optional
        name of the config file used for the environment, by default '1container_1press.json'
    overwrite_episode_length: int, optional
        if a different episode length is desired in inference, set that length here
    overwrite_timestep : int, optional
        if a different timestep length is desired in inference, set that length here
    deterministic_policy: bool, optional
        whether to sample from deterministic policy, by default True
    algorithm: str, optional
        the RL algorithm used, can be 'ppo', 'trpo', 'dqn', or 'rulebased', by default 'ppo'
    n_rollouts: int, optional
        number of rollouts used to collect rewards, by default 15
    include_action_zero : bool, optional
        if True, rewards obtained when action 0 (do nothing) is performed are included in the sample, default is False

    Returns:
    --------
    rewards: list
        list of rewards collected over the rollouts
    """

    env = ContainerEnv.from_json(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../containergym/configs/" + config_file,
        )
    )
    if overwrite_episode_length:
        env.max_episode_length = overwrite_episode_length
    if overwrite_timestep:
        env.timestep = overwrite_timestep
    env = Monitor(env)
    if algorithm == "ppo":
        model = PPO.load(log_dir + "best_model.zip")
    elif algorithm == "trpo":
        model = TRPO.load(log_dir + "best_model.zip")
    elif algorithm == "dqn":
        model = DQN.load(log_dir + "best_model.zip")
    elif algorithm == "rulebased":
        model = RuleBasedAgent(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../containergym/configs/" + config_file,
            )
        )
    # Keep track of rewards and ratio of emptying actions
    rewards = []
    rewards_all = []  # Needed to compute ratio of rollouts that end prematurely
    ratio_emptying_actions = 0
    total_actions = 0

    for _ in range(n_rollouts):
        obs = env.reset()
        step = 0
        episode_length = 0
        # Run episode
        while True:
            episode_length += 1
            action, _ = model.predict(obs, deterministic=deterministic_policy)
            total_actions += 1
            obs, reward, done, info = env.step(action)

            rewards_all.append(reward)

            if action:
                rewards.append(reward)
                ratio_emptying_actions += 1
            else:
                if include_action_zero:
                    rewards.append(reward)

            if done:
                break
            step += 1

    print("Ratio of {}'s rollouts ending prematurely: {:.4}".format(algorithm, len([r for r in rewards_all if r == -1])
                                                                    / n_rollouts))
    print("Ratio of emptying actions for {}: {:.4f}".format(algorithm, ratio_emptying_actions / total_actions))
    print("------------------------------------------------------------------------------------------")
    return rewards


def plot_ecdf_rewards(dict_rewards, fig_name="", format="png"):
    """
    Plots ECDFs of episodic rewards passed as dictionary, where the key corresponds to an algorithm name.

    Plots empirical cumulative distribution functions (ECDFs) of episodic rewards for different algorithms.

    Parameters
    ----------
    dict_rewards : dict
        Dictionary of episodic rewards where the keys are algorithm names.
    fig_name : str, optional
        Name of the output file. Default is ''.
    format : str, optional
        Format of the output file. Default is 'png'.

    Returns
    -------
    None

    """

    # Plot reward ECDFs
    plt.rcParams.update({"font.size": 13})
    line_width = 2
    ls_cycler = plt.cycler(ls=["-", "-.", "--"], color=3 * ["blue"])
    plt.gca().set_prop_cycle(ls_cycler)
    for k, r in dict_rewards.items():
        ecdf = ECDF(r)
        plt.plot(ecdf.x, ecdf.y, linewidth=line_width, label=k)

    plt.title("ECDF of reward per emptying action")
    plt.grid()
    plt.xlabel("Reward")
    xmin, xmax = -1, 1  # min([min(v) for v in dict_rewards.values()]), max([max(v) for v in dict_rewards.values()])
    plt.xlim(xmin, xmax)

    plt.legend(loc=2)
    plt.savefig(fig_name + ".%s" % format, dpi="figure", format=format)
    plt.show()


def produce_ecdf_rewards():
    """
    Runs the main experiment by computing the episodic rewards for PPO, TRPO, DQN and rule-based controller,
    and then plotting their reward ECDFs.

    Returns:
    --------
    None
    """
    fig_format = "svg"
    paths = [
        "./logs_best_seeds/ppo_11containers_11presses_5_seed_3_budget_5000000_ent-coef_0.0_gamma_0.99_steps_6144",
        "./logs_best_seeds/trpo_11containers_11presses_5_seed_4_budget_5000000_gamma_0.99_steps_6144",
        "./logs_best_seeds/dqn_11containers_11presses_5_seed_10_budget_5000000_gamma_0.99",
    ]

    dict_rewards = {}

    for path in paths:
        # Extract run and config info from path
        run_name = path.split("/")[-1]
        print(run_name)
        config_name = run_name.split("_seed_")[0].split("_", maxsplit=1)[-1]
        config_file = config_name + ".json"
        algorithm = run_name.split("_")[0]
        log_dir = (
            os.path.dirname(os.path.abspath(__file__))
            + "/logs_best_seeds/"
            + run_name
            + "/"
        )
        print(config_file)
        dict_rewards[algorithm.upper()] = episodic_reward(
            log_dir,
            config_file=config_file,
            overwrite_episode_length=600,
            overwrite_timestep=120,
            algorithm=algorithm,
            deterministic_policy=True,
            n_rollouts=15,
            include_action_zero=False
        )
    # Reward ECDFs
    fig_name = "reward_ecdfs_" + config_name
    plot_ecdf_rewards(dict_rewards, fig_name=fig_name, format=fig_format)

    # Reward ECDFs for rule-based controller
    dict_rewards = {}
    algorithm = "rulebased"
    labels = [r"$n = 5, m = 2$", r"$n = 11, m = 2$", r"$n = 11, m = 11$"]
    for i, config_file in enumerate(["5containers_2presses_2.json", "11containers_2presses_3.json",
                                     "11containers_11presses_5.json"]):
        print(labels[i])
        dict_rewards[labels[i]] = episodic_reward(log_dir, config_file=config_file,
                                                  overwrite_episode_length=600, overwrite_timestep=120,
                                                  algorithm=algorithm, deterministic_policy=True, n_rollouts=15,
                                                  include_action_zero=False)
    fig_name = 'reward_ecdfs_rule_based_controller'
    plot_ecdf_rewards(dict_rewards, fig_name=fig_name, format=fig_format)


if __name__ == "__main__":
    produce_ecdf_rewards()