import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import argparse
from distutils.util import strtobool
from multiprocessing import Process
import glob
import re
from statsmodels.distributions.empirical_distribution import ECDF
import json
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO
from sb3_contrib import TRPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor

from containergym.env import ContainerEnv

os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(1)



def get_color_code():
    """Dummy function to avoid defining color code in different places.
    The user should make sure to define a color for each container contained in the environment.
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
                  "C2-60": "#FF3E96",  # violet
                  "C2-70": "#FFD700",  # gold1
                  "C2-80": "#7FFF00",  # chartreuse1
                  "C2-90": "#D2691E",  # chocolate
                  }
    return color_code

def inference(args):
    """
    The inference function is used to evaluate a trained agent and create plots of its performance.

    :param seed: Set the seed for the random number generator
    :param args: Pass arguments to the script
    :return: None
    """


    save_fig = True
    n_steps = args["steps"]
    seed = args["seed"]
    config_file = args["config_file"]+".json"
    fig_format = "pdf"
    budget = args["budget"]
    ent_coef = args["ent_coeff"]
    gamma = args["gamma"]
    RL_agent = args["RL_agent"].upper()
    deterministic_policy = True

    run_name = args["dir"]

    fig_name = "run_trained_agent_deterministic_policy_" + run_name

    log_dir = os.path.dirname(os.path.abspath(__file__)) + "/logs_best_seeds/" + run_name + "/"

    log_dir_results = os.path.dirname(os.path.abspath(__file__)) + "/results_paper/"

    os.makedirs(log_dir, exist_ok=True)

    env = ContainerEnv.from_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../configs/" + config_file))
    env = Monitor(env)
    obs = env.reset()
    if RL_agent == 'PPO':
        model = PPO.load(log_dir + "best_model.zip")
    elif RL_agent == 'TRPO':
        model = TRPO.load(log_dir + "best_model.zip")
    elif RL_agent == 'DQN':
        model = DQN.load(log_dir + "best_model.zip")

    # Keep track of state variables (volumes), actions, and rewards
    volumes = []
    actions = []
    rewards = []
    free_presses = []

    # free_presses = [idx for idx, time in enumerate(self.state.press_times) if time == 0]
    # obs["Time presses will be free"]

    step = 0

    episode_length = 0

    # Run episode
    while True:
        episode_length += 1
        action, _ = model.predict(obs, deterministic=deterministic_policy)
        actions.append(action)
        # print("Step {}".format(step + 1))
        # print("Action: ", action)
        obs, reward, done, info = env.step(action)
        volumes.append(obs["Volumes"].copy())
        rewards.append(reward)
        free_presses.append(len(np.where(obs["Time presses will be free"] == 0)[0]))
        if done:
            break
        step += 1

    # Plot state variables
    fig = plt.figure(figsize=(15, 10))
    fontsize = 15
    env_unwrapped = env.unwrapped
    if deterministic_policy:
        fig.suptitle(RL_agent + ". Trained policy (deterministic) on test environment", fontsize=fontsize)
    else:
        fig.suptitle(RL_agent + ". Trained policy (non-deterministic) on test environment", fontsize=fontsize)

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    # ax4 = fig.add_subplot(414)

    ax1.set_title("Volume")
    ax2.set_title("Action")
    ax3.set_title("Reward")
    # ax4.set_title("Free presses")
    # ax3.set_title("Reward. Cumulative reward: {:.2f}".format(sum(rewards)))

    ax1.grid()
    ax2.grid()
    ax3.grid()
    # ax4.grid()

    ax1.set_ylim(top=40)  # TODO: get param from json config file
    ax2.set_yticks(list(range(env_unwrapped.action_space.n)))
    ax3.set_ylim(bottom=-0.1, top=1)  # TODO: get param from json config file
    # ax4.set_yticks(list(range(env_unwrapped.observation_space["Time presses will be free"].shape[0])))

    plt.xlabel("Steps", fontsize=fontsize)

    default_color = "#1f77b4"  # Default Matplotlib blue color
    color_code = get_color_code()
    line_width = 3

    # Plot volumes for each bunker
    for i in range(env_unwrapped.n_containers):
        ax1.plot(np.array(volumes)[:, i], linewidth=line_width, label=env_unwrapped.enabled_containers[i],
                 color=color_code[env_unwrapped.enabled_containers[i]])
    ax1.legend()

    # Plot actions and rewards
    x_axis = range(episode_length)
    for i in x_axis:
        if actions[i] == 0:  # Action: "do nothing"
            ax2.scatter(i,
                        actions[i],
                        linewidth=line_width,
                        color=default_color)
            ax3.scatter(i,
                        rewards[i],
                        linewidth=line_width,
                        color=default_color,
                        clip_on=False)
        elif actions[i] in range(1, env_unwrapped.n_containers + 1):  # Action: "use Press 1"
            ax2.scatter(i,
                        actions[i],
                        linewidth=line_width,
                        color=color_code[env_unwrapped.enabled_containers[actions[i] - 1]],
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

    # Plot number of free presses
    # ax4.scatter(range(episode_length), free_presses, linewidth=line_width, color=default_color, clip_on=False)

    # Annotate reward plot with cumulative reward
    ax3.annotate("Cumul. reward: {:.2f}".format(sum(rewards)), xy=(0.9, 0.9), xycoords='axes fraction', fontsize=14)

    if save_fig:
        # Save plot
        file_name = fig_name+".png"
        plt.savefig(log_dir_results+file_name, dpi='figure',)

    print("Ratio of rewards equal to rpen: ",
          100 * len([x for x in rewards if x == -1e-1]) / len([x for x in rewards if x <= 0]))

    #plt.show()
def plot_ecdf(args = None , n_rollouts=15):
    """Performs n_rollouts rollouts of a policy and plots the ECDF of the emptying volumes collected over all rollouts
    for a trained model.
    """
    save_fig = True
    n_steps = args["steps"]
    seed = args["seed"]
    config_file = args["config_file"]+".json"
    fig_format = "png"
    budget = args["budget"]
    ent_coef = args["ent_coeff"]
    gamma = args["gamma"]
    RL_agent = args["RL_agent"].upper()
    deterministic_policy = True
    n_rollouts = n_rollouts
    run_name = args["dir"]

    config_name = args["config_file"]

    fig_name = 'ecdf_emptying_vols_ppo_' + config_name
    format = fig_format


    log_dir = os.path.dirname(os.path.abspath(__file__)) + "/logs_best_seeds/" + run_name + "/"

    log_dir_results = os.path.dirname(os.path.abspath(__file__)) + "/results_paper/"

    os.makedirs(log_dir, exist_ok=True)

    if RL_agent == 'PPO':
        model = PPO.load(log_dir + "best_model.zip")
    elif RL_agent == 'TRPO':
        model = TRPO.load(log_dir + "best_model.zip")
    elif RL_agent == 'DQN':
        model = DQN.load(log_dir + "best_model.zip")

    env = ContainerEnv.from_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../configs/" + config_file))
    env = Monitor(env)

    log_dir_results = os.path.dirname(os.path.abspath(__file__)) + "/results_paper/ecdf/"

    os.makedirs(log_dir_results, exist_ok=True)


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
        file_name = fig_name + ".png"

        plt.savefig(log_dir_results+fig_name + '_' + k + '.%s' % format, dpi='figure', format=format)
        plt.close()
        #plt.show()
def average_cumulative_reward(args = None,  n_rollouts=15):
    """Performs n_rollouts rollouts of a policy and prints the average cumulative reward and the standard deviation.
    """

    save_fig = True
    config_file = args["config_file"] + ".json"
    RL_agent = args["RL_agent"].upper()
    deterministic_policy = True
    n_rollouts = n_rollouts
    run_name = args["dir"]
    log_dir = os.path.dirname(os.path.abspath(__file__)) + "/logs_best_seeds/" + run_name + "/"
    env = ContainerEnv.from_json(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../configs/" + config_file
        )
    )

    env = Monitor(env)

    if RL_agent == "PPO":
        model = PPO.load(log_dir + "best_model.zip")
    elif RL_agent == "TRPO":
        model = TRPO.load(log_dir + "best_model.zip")
    elif RL_agent == "DQN":
        model = DQN.load(log_dir + "best_model.zip")


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

if __name__ == "__main__":

    path =  os.path.dirname(os.path.abspath(__file__)) + "/logs_best_seeds/*"
    paths = []
    args = []
    for file in sorted(glob.glob(path)):
        arg = {}
        dir = os.path.basename(file)
        paths.append(dir)
        arg["dir"] = dir
        budget = int(dir.split("budget_")[-1].split("_")[0])  # Parse budget from run name
        arg["budget"] = budget
        agent = str(dir.split("_")[0])
        arg["RL_agent"] = agent
        try:
            ent_coeff = float(dir.split("ent-coef_")[-1].split("_")[0])  # Parse entropy from run name
        except:
            ent_coeff = None
        arg["ent_coeff"] = ent_coeff
        seed = int(dir.split("seed_")[-1].split("_")[0])  # Parse seed from run name
        arg["seed"] = seed
        gamma = float(dir.split("gamma_")[-1].split("_")[0])  # Parse gamma from run name
        arg["gamma"] = gamma
        try:
            steps = int(dir.split("steps_")[-1].split("_")[0])  # Parse steps from run name
        except:
            steps = None
        arg["steps"] = steps
        config_file = re.search(f'{agent}_(.*)_seed', dir)  # Parse config_file from run name
        arg["config_file"] = config_file.group(1)

        args.append(arg)
        #print(arg)

    for arg in args:
        average_cumulative_reward(arg, n_rollouts=15)
        inference(arg)
        plot_ecdf(arg, n_rollouts=15)

