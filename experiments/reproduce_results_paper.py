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

os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)

def run_trained_agent(log_dir, config_file='1container_1press.json', overwrite_episode_length=False,
                      overwrite_timestep=False, algorithm='ppo', deterministic_policy=True, fig_name='', format='png',
                      save_fig=False):
    """
    Evaluates a trained agent and creates plots of its performance.

    Parameters
    ----------
    log_dir : str
        The log directory in which the agent was saved.
    config_file : str, optional
        The name of the config file used for the environment. Default is '1container_1press.json'.
    overwrite_episode_length : int or bool, optional
        If a different episode length is desired in inference, set that length here. Default is `False`.
    overwrite_timestep : int or bool, optional
        If a different timestep is desired in inference, set that timestep here. Default is `False`.
    algorithm : str, optional
        The RL algorithm used for training the agent. Valid options are: 'ppo', 'trpo', 'dqn', and 'deterministic'.
        Default is 'ppo'.
    deterministic_policy : bool, optional
        Whether to sample from a deterministic policy. Default is `True`.
    fig_name : str, optional
        The figure name for the generated plots. Default is an empty string.
    format : str, optional
        The file format of the generated plots. Default is 'png'.
    save_fig : bool, optional
        Whether to save the figures to file. Default is `False`.

    Returns
    -------
    None
    """

    env = ContainerEnv.from_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../containergym/configs/" + config_file))
    if overwrite_episode_length:
        env.max_episode_length = overwrite_episode_length
    if overwrite_timestep:
        env.timestep = overwrite_timestep
    env = Monitor(env)
    obs = env.reset()
    if algorithm == 'ppo':
        model = PPO.load(log_dir + "best_model.zip")
    elif algorithm == 'trpo':
        model = TRPO.load(log_dir + "best_model.zip")
    elif algorithm == 'dqn':
        model = DQN.load(log_dir + "best_model.zip")
    elif algorithm == 'deterministic':
        model = RuleBasedAgent(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "../containergym/configs/" + config_file))

    # Keep track of state variables (volumes), actions, and rewards
    volumes = []
    actions = []
    rewards = []
    free_presses = []
    step = 0

    episode_length = 0

    # Run episode
    while True:
        episode_length += 1
        action, _ = model.predict(obs, deterministic=deterministic_policy)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        volumes.append(obs["Volumes"].copy())
        rewards.append(reward)
        free_presses.append(len(np.where(obs["Time presses will be free"] == 0)[0]))
        if done:
            break
        step += 1

    # Plot state variables
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(15, 10))
    fig.subplots_adjust(hspace=0.4)
    # fontsize = 15
    env_unwrapped = env.unwrapped

    if algorithm == 'deterministic':
        fig.suptitle("Rollout of deterministic controller on test environment")
    else:
        fig.suptitle("Rollout of " + algorithm.upper() + " on test environment")

    ax1 = fig.add_subplot(311)
    plt.xlim(left=-10, right=env.max_episode_length + 10)
    ax2 = fig.add_subplot(312)
    plt.xlim(left=-10, right=env.max_episode_length + 10)
    ax3 = fig.add_subplot(313)
    plt.xlim(left=-10, right=env.max_episode_length + 10)

    ax1.set_title("Volume")
    ax2.set_title("Action")
    ax3.set_title("Reward")

    ax1.grid()
    ax2.grid()
    ax3.grid()

    ax1.set_ylim(top=40)
    if env_unwrapped.action_space.n == 6: 
        ax2.set_yticks(list(range(env_unwrapped.action_space.n)))
    if env_unwrapped.action_space.n == 12:
        ax2.set_yticks(np.arange(0, env_unwrapped.action_space.n, 2))
    if env_unwrapped.action_space.n > 2:  # Avoid warning due to "bottom" and "top" flags being equal
        ax2.set_ylim(bottom=1, top=env_unwrapped.action_space.n - 1)
    ax3.set_ylim(bottom=-0.1, top=1) 

    plt.xlabel("Timestep")  # , fontsize=fontsize)

    default_color = "#1f77b4"  # Default Matplotlib blue color
    color_code = get_color_code()
    line_width = 3

    # Plot volumes for each container
    for i in range(env_unwrapped.n_containers):
        ax1.plot(np.array(volumes)[:, i], linewidth=line_width, label=env_unwrapped.enabled_containers[i],
                 color=color_code[env_unwrapped.enabled_containers[i]])
    # ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1.5))

    # Plot actions and rewards
    x_axis = range(episode_length)
    for i in x_axis:
        if actions[i] == 0:  # Action: "do nothing"
            ax2.scatter(i,
                        actions[i],
                        linewidth=line_width,
                        color=default_color,
                        alpha=0)
            ax3.scatter(i,
                        rewards[i],
                        linewidth=line_width,
                        color=default_color,
                        clip_on=False,
                        alpha=0)
        elif actions[i] in range(1, env_unwrapped.n_containers + 1):  # Action: "use Press 1"
            ax2.scatter(i,
                        actions[i],
                        linewidth=line_width,
                        color=color_code[env_unwrapped.enabled_containers[actions[i] - 1]],
                        clip_on=False,
                        marker="^")
            ax3.scatter(i,
                        rewards[i],
                        linewidth=line_width,
                        color=color_code[env_unwrapped.enabled_containers[actions[i] - 1]],
                        clip_on=False)
        elif actions[i] in range(env_unwrapped.n_containers + 1, env_unwrapped.n_containers * 2 + 1):  # Action: "use Press 2"
            ax2.scatter(i,
                        actions[i],
                        linewidth=line_width,
                        color=color_code[env_unwrapped.enabled_containers[actions[i] - env_unwrapped.n_containers - 1]],
                        marker="x")
            ax3.scatter(i,
                        rewards[i],
                        linewidth=line_width,
                        color=color_code[env_unwrapped.enabled_containers[actions[i] - env_unwrapped.n_containers - 1]],
                        clip_on=False)
        else:
            print("Unrecognised action: ", actions[i])
    # ax2.legend(handles=[mlines.Line2D([], [], color='black', marker='^', linestyle='None', label='Press 1'),
    #                     mlines.Line2D([], [], color='black', marker='x', linestyle='None', label='Press 2')])

    # Annotate reward plot with cumulative reward
    ax3.annotate("Cumul. " + '$r:{:.2f}$'.format(sum(rewards)), xy=(0.81, 1.02), xycoords='axes fraction')  # , fontsize=14)

    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    if save_fig:
        # Save plot
        plt.savefig(fig_name + '.%s' % format, dpi='figure', format=format)
        plt.close()

    print("Ratio of rewards < 0:", len([x for x in rewards if x < 0]) / len(rewards))

def average_cumulative_reward(log_dir, config_file='1container_1press.json', overwrite_episode_length=False,
                              overwrite_timestep=False, algorithm='ppo', n_rollouts=15, deterministic_policy=True):
    """
    Perform n_rollouts rollouts of a policy and print the average cumulative reward and the standard deviation.

    Parameters
    ----------
    log_dir : str
        The directory containing the trained model.
    config_file : str, optional
        The name of the configuration file to use, by default '1container_1press.json'.
    overwrite_episode_length : int, optional
        If provided, the maximum episode length of the environment will be set to this value, by default False.
    overwrite_timestep : int, optional
        If provided, the timestep size of the environment will be set to this value, by default False.
    algorithm : str, optional
        The type of reinforcement learning algorithm used for training the model (ppo, trpo, or dqn), by default 'ppo'.
    n_rollouts : int, optional
        The number of rollouts to perform, by default 15.
    deterministic_policy : bool, optional
        Whether to use the deterministic policy or not, by default True.

    Returns
    -------
    None
        The function prints the average cumulative reward and the standard deviation of the rollouts.

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

    # Keep track of cumulative reward of each episode/rollout
    cumulative_rewards = []

    # Perform rollouts
    for _ in range(n_rollouts):
        obs = env.reset()
        step = 0
        cumulative_reward = 0

        while True:
            action, _ = model.predict(obs, deterministic=deterministic_policy)
            obs, reward, done, info = env.step(action)
            if done:
                break

            cumulative_reward += reward
            step += 1

        cumulative_rewards.append(cumulative_reward)

    print("Average cumul. reward: {:.2f} \u00B1 {:.2f}".format(np.mean(cumulative_rewards), np.std(cumulative_rewards)))


def emptying_volumes_ecdfs(log_dirs, config_file='1container_1press.json', overwrite_episode_length=None,
                           overwrite_timestep=None, n_rollouts=15, deterministic_policy=True, fig_name='',
                           format='png'):
    """
    Performs n_rollouts rollouts of a policy and plots the empirical cumulative distribution function (ECDF) of the
    emptying volumes collected over all rollouts for PPO, TRPO, and DQN agents passed as a list in log_dirs.

    Parameters
    ----------
    log_dirs : List[str]
        List of three paths to the saved models for PPO, TRPO, and DQN agents respectively.
    config_file : str, optional
        Path to the JSON configuration file for the container environment (default is '1container_1press.json').
    overwrite_episode_length : int, optional
        If provided, overrides the maximum episode length in the environment.
    overwrite_timestep : float, optional
        If provided, overrides the timestep in the environment.
    n_rollouts : int, optional
        Number of rollouts to perform for each agent (default is 15).
    deterministic_policy : bool, optional
        If True, the policy is deterministic; otherwise, it is stochastic (default is True).
    fig_name : str, optional
        Name to give the generated plot file (default is an empty string).
    format : str, optional
        Format to save the generated plot file (default is 'png').

    Returns
    -------
    None

    Notes
    -----
    This function assumes that log_dirs contains paths to 3 models, namely PPO, TRPO, and DQN in that order.
    """
    env = ContainerEnv.from_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../containergym/configs/" + config_file))
    if overwrite_episode_length:
        env.max_episode_length = overwrite_episode_length
    if overwrite_timestep:
        env.timestep = overwrite_timestep
    env = Monitor(env)
    models = [PPO.load(log_dirs[0] + "best_model.zip"), TRPO.load(log_dirs[1] + "best_model.zip"),
              DQN.load(log_dirs[2] + "best_model.zip")]

    # Keep track of volumes and actions
    volumes = [[], [], []]
    actions = [[], [], []]

    for i, model in enumerate(models):
        # Perform rollouts
        for _ in range(n_rollouts):
            obs = env.reset()
            volumes[i].append(obs["Volumes"].copy())  # Save initial volumes
            step = 0
            episode_length = 0

            while True:
                episode_length += 1
                action, _ = model.predict(obs, deterministic=deterministic_policy)
                actions[i].append(action)
                obs, reward, done, info = env.step(action)
                if done:
                    break
                else:
                    volumes[i].append(obs["Volumes"].copy())  # Save all volumes but the last ones
                step += 1

    env_unwrapped = env.unwrapped

    # Emptying volumes
    emptying_volumes = {k: [[], [], []] for k in env_unwrapped.enabled_containers}

    for j in range(len(models)):
        # Extract volumes corresponding to emptying actions
        for i, a in enumerate(actions[j]):
            if a > 0:
                emptying_volumes[env_unwrapped.enabled_containers[a - 1]][j].append(np.array(volumes[j])[i, a - 1])

    # Plot ECDFs
    plt.rcParams.update({'font.size': 13})
    color_code = get_color_code()
    line_width = 2
    for k, v in emptying_volumes.items():
        ecdf_ppo = ECDF(v[0])
        ecdf_trpo = ECDF(v[1])
        ecdf_dqn = ECDF(v[2])
        plt.figure()
        plt.title(k)
        plt.plot(ecdf_ppo.x, ecdf_ppo.y, color=color_code[k], label="PPO", linewidth=line_width)
        plt.plot(ecdf_trpo.x, ecdf_trpo.y, color=color_code[k], label="TRPO", linewidth=line_width, ls="-.")
        plt.plot(ecdf_dqn.x, ecdf_dqn.y, color=color_code[k], label="DQN", linewidth=line_width, ls="--")
        plt.grid()
        plt.xlim(-1, 40)
        plt.ylim(0, 1)
        plt.xlabel("Emptying volumes")

        # Annotate optimal volumes
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../containergym/configs/" + config_file), 'r') as f:
            # Add optimal volumes
            data = json.load(f)
            optimal_vols = data.get("REWARD_PARAMS")[k]["peaks"]
            plt.vlines(x=optimal_vols[0], ymin=0, ymax=1, label="global opt.", colors="grey", linewidth=line_width)
            if len(optimal_vols) > 1:
                plt.vlines(x=optimal_vols[1:], ymin=0, ymax=1, ls=":", label="local opt.", colors="grey",
                           linewidth=line_width)

            # Add fill rate
            plt.annotate("Fill rate: {:.2e}".format(data.get("RW_MUS")[k]),
                         xy=(0.03, 0.5),
                         xycoords='axes fraction')

        plt.legend(loc=2)

        plt.savefig(fig_name + '_' + k + '.%s' % format, dpi='figure', format=format)
        plt.close()

def plot_ecdf(model, config_file='1container_1press.json', overwrite_episode_length=False,
              overwrite_timestep=False, n_rollouts=15, deterministic_policy=True, fig_name='', format='png'):
    """
    Plot empirical cumulative distribution function (ECDF) of emptying volumes for a given trained model.

    Parameters:
    -----------
    model : object
        Trained machine learning model to use for predicting actions in the environment.
    config_file : str, optional
        Path to the JSON configuration file containing environment settings. Default is '1container_1press.json'.
    overwrite_episode_length : int or bool, optional
        If not False, overwrite the maximum episode length defined in the configuration file. Default is False.
    overwrite_timestep : float or bool, optional
        If not False, overwrite the timestep defined in the configuration file. Default is False.
    n_rollouts : int, optional
        Number of rollouts to perform. Default is 15.
    deterministic_policy : bool, optional
        Whether to use a deterministic policy for predicting actions. Default is True.
    fig_name : str, optional
        Name of the file to save the resulting plot. If empty, the plot is not saved. Default is an empty string.
    format : str, optional
        Format of the saved figure. Default is 'png'.

    Returns:
    --------
    None
    """

    env = ContainerEnv.from_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../experiments/configs/" + config_file))
    if overwrite_episode_length:
        env.max_episode_length = overwrite_episode_length
    if overwrite_timestep:
        env.timestep = overwrite_timestep
    env = Monitor(env)

    # Keep track of volumes and actions
    volumes = []
    actions = []

    # Perform rollouts
    for _ in range(n_rollouts):
        obs = env.reset()
        volumes.append(obs["Volumes"].copy())  # Save initial volumes
        step = 0
        episode_length = 0

        while True:
            episode_length += 1
            action, _ = model.predict(obs, deterministic=deterministic_policy)
            actions.append(action)
            obs, reward, done, info = env.step(action)
            if done:
                break
            else:
                volumes.append(obs["Volumes"].copy())  # Save all volumes but the last ones
            step += 1

    env_unwrapped = env.unwrapped

    # Emptying volumes
    emptying_volumes = {k: [] for k in env_unwrapped.enabled_containers}

    # Extract volumes corresponding to emptying actions
    for i, a in enumerate(actions):
        if a > 0:
            emptying_volumes[env_unwrapped.enabled_containers[a - 1]].append(np.array(volumes)[i, a - 1])

    # Plot ECDFs
    plt.rcParams.update({'font.size': 13})
    color_code = get_color_code()
    line_width = 2
    for k, v in emptying_volumes.items():
        ecdf = ECDF(v)
        plt.figure()
        plt.title(k)
        plt.plot(ecdf.x, ecdf.y, color=color_code[k], linewidth=line_width)
        plt.grid()
        plt.xlim(-1, 40)
        plt.ylim(0, 1)
        plt.xlabel("Emptying volumes")

        # Annotate optimal volumes
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../configs/" + config_file), 'r') as f:
            # Add optimal volumes
            data = json.load(f)
            optimal_vols = data.get("REWARD_PARAMS")[k]["peaks"]
            plt.vlines(x=optimal_vols[0], ymin=0, ymax=1, label="global opt.", colors="grey", linewidth=line_width)
            if len(optimal_vols) > 1:
                plt.vlines(x=optimal_vols[1:], ymin=0, ymax=1, ls=":", label="local opt.", colors="grey",
                           linewidth=line_width)

            # Add fill rate
            plt.annotate("Fill rate: {:.2e}".format(data.get("RW_MUS")[k]),
                         xy=(0.03, 0.5),
                         xycoords='axes fraction')

        plt.legend(loc=2)

        plt.savefig(fig_name + '_' + k + '.%s' % format, dpi='figure', format=format)

        plt.show()


def get_color_code():
    """
    Returns a dictionary containing color codes for each container in the environment.

    Returns:
    ----------
    color_code : dict
        A dictionary containing color codes for each container in the environment. Keys are container names and values are color codes represented as hexadecimal strings.

    Note: The user should make sure to define a color for each container contained in the environment.

    """

    color_code = {"C1-10": "#872657",  # raspberry
                  "C1-20": "#0000FF",  # blue
                  "C1-30": "#FFA500",  # orange
                  "C1-40": "#008000",  # green
                  "C1-50": "#B0E0E6",  # powderblue
                  "C1-60": "#FF00FF",  # fuchsia
                  "C1-70": "#800080",  # purple
                  "C1-80": "#FF4500",  # orangered
                  "C2-10": "#DB7093",  # palevioletred
                  "C2-20": "#FF8C69",  # salmon1
                  "C2-40": "#27408B",  # royalblue4
                  "C2-50": "#54FF9F",  # seagreen1
                  "C2-60": "#FF3E96",  # violetred1
                  "C2-70": "#FFD700",  # gold1
                  "C2-80": "#7FFF00",  # chartreuse1
                  "C2-90": "#D2691E",  # chocolate
                  }
    return color_code

def main():
    """
    Runs the main function of the script which performs the following tasks:

    1. Finds all zip files containing logs in the specified directory and subdirectories.
    2. Extracts run and config information from each path.
    3. Averages cumulative rewards over 15 rollouts.
    4. Creates plots for volume, action, and reward over a single episode.
    5. Creates an empirical cumulative distribution function (ECDF) plot for emptying volumes.
    6. Runs a rule-based agent for specified configurations.
    """
    fig_format = 'svg'
    paths = glob.glob("./logs_best_seeds/**/*.zip", recursive=True)
    print("Found logs:")
    print("\n".join(paths))
    for path in paths:
        # Extract run and config info from path
        run_name = path.split(os.sep)[-2]
        print("Run name: " + run_name)
        config_name = run_name.split("_seed_")[0].split("_", maxsplit=1)[-1]
        print("Config name: " + config_name)
        config_file = config_name + ".json"
        algorithm = run_name.split('_')[0]
        log_dir = os.path.dirname(os.path.abspath(__file__)) + '/logs_best_seeds/' + run_name + '/'
        
        # Average cumul. rewards over 15 rollouts
        average_cumulative_reward(log_dir, config_file=config_file, overwrite_episode_length=600,
                                  overwrite_timestep=120, algorithm=algorithm, n_rollouts=15,
                                  deterministic_policy=True)  
        
        # Volume, Action, and Reward plots over single episode
        fig_name = 'run_trained_agent_deterministic_policy_' + run_name
        run_trained_agent(log_dir=log_dir, config_file=config_file, overwrite_episode_length=600,
                          algorithm=algorithm, deterministic_policy=True, fig_name=fig_name, format=fig_format,
                          save_fig=True)
        print("---------------------------------------")
    
    # ECDF Plotting
    config_for_ecdf = "5containers_2presses_2"
    config_file_for_ecdf = config_for_ecdf + ".json"
    run_names = ['ppo_' + config_for_ecdf + '_seed_3_budget_2000000_ent-coef_0.0_gamma_0.99_steps_6144',
                 'trpo_' + config_for_ecdf + '_seed_1_budget_2000000_gamma_0.99_steps_6144',
                 'dqn_' + config_for_ecdf + '_seed_13_budget_2000000_gamma_0.99']
    log_dirs = [os.path.dirname(os.path.abspath(__file__)) + '/logs_best_seeds/' + run_name + '/' for run_name in run_names]
    print("Creating ECDF Plots...")
    emptying_volumes_ecdfs(log_dirs, config_file=config_file_for_ecdf, overwrite_episode_length=600, overwrite_timestep=120,
                           fig_name='ecdf_emptying_vols_' + config_name + '_best_seeds', format=fig_format)
    print("Done.")
    # Rule Based Agent runs
    rule_based_configs = ["5containers_2presses_2", "5containers_5presses_2",
                          "11containers_2presses_3", "11containers_11presses_5"]
    for config in rule_based_configs:
        algorithm = 'deterministic'
        config_file = config + ".json"
        fig_name = 'run_rule_based_agent_' + config
        print("Running Rule Based Agent for config: " + config + " | algorithm: " + algorithm)
        run_trained_agent(log_dir=None, config_file=config_file, overwrite_episode_length=600,
                          algorithm='deterministic', fig_name=fig_name, format=fig_format,
                          save_fig=True)
if __name__ == "__main__":
    main()