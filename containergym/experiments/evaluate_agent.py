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
from sb3_contrib import TRPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor

from containergym.env import ContainerEnv

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
        "--render-episode",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, render the episodes during evaluation",
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

class rulebased_agent():
        def __init__(self):
            self.peak_rew_vols = {"C1-10": [19.84],
                                  "C1-20": [26.75],
                                  "C1-30": [8.84, 17.70, 26.56],
                                  "C1-40": [8.40, 16.80, 25.19],
                                  "C1-50": [4.51, 9.02, 13.53],
                                  "C1-60": [7.19, 14.38, 21.57],
                                  "C1-70": [8.65, 17.31, 25.96],
                                  "C1-80": [12.37, 24.74],
                                  "C2-10": [27.39],
                                  "C2-20": [37.04],
                                  "C2-40": [8.58, 17.17, 25.77],
                                  "C2-50": [4.60, 9.21, 13.82],
                                  "C2-60": [8.58, 17.17, 25.77],
                                  "C2-70": [13.22, 26.46],
                                  "C2-80": [28.75],
                                  "C2-90": [5.76, 11.50, 17.26]}

        def predict(self, obs=None, env_s=None, peak_vols=None):
            clash_counter = 0
            for i in range(env_s.n_bunkers):
                if peak_vols[i] - 1 < obs[i] < peak_vols[i] + 1:
                    action = i + 1
                    clash_counter += 1
                elif obs[i] > peak_vols[i] + 1:
                    action = i + 1
                    clash_counter += 1
                elif not clash_counter:
                    action = 0
            return action

        def set_env(self, env_s=None):
            peak_vols = []
            for i in range(env_s.n_bunkers):
                print(env_s.enabled_bunkers[i])
                peak_vols.append(self.peak_rew_vols[env_s.enabled_bunkers[i]][-1])
            return peak_vols

def get_color_code():
    """Dummy function to avoid defining color code in different places.
    The user should make sure to define a color for each bunker contained in the environment.
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

def inference(seed, args):
    """
    The inference function is used to evaluate a trained agent and create plots of its performance.

    :param seed: Set the seed for the random number generator
    :param args: Pass arguments to the script
    :return: None
    """


    save_fig = True
    n_steps = args.n_steps
    seed = seed
    config_file = args.config_file
    fig_format = "svg"
    budget = args.budget
    config_file = args.config_file
    budget = args.budget
    ent_coef = args.ent_coeff
    gamma = args.gamma

    if args.RL_agent in ["PPO", "A2C", "TRPO"]:
        run_name = (
                f"{args.RL_agent}_"
                + config_file.replace(".json", "")
                + "_seed_"
                + str(seed)
                + "_budget_"
                + str(budget)
                + "_ent-coef_"
                + str(ent_coef)
                + "_gamma_"
                + str(gamma)
                + "_n_steps_"
                + str(n_steps)
        )
    else:
        run_name = (
                f"{args.RL_agent}_"
                + config_file.replace(".json", "")
                + "_seed_"
                + str(seed)
                + "_budget_"
                + str(budget)
                + "_ent-coef_"
                + str(ent_coef)
                + "_gamma_"
                + str(gamma)
        )


    fig_name = "run_trained_agent_deterministic_policy_" + run_name

    log_dir = os.path.dirname(os.path.abspath(__file__)) + "/logs/" + run_name + "/"

    budget = int(
        run_name.split("budget_")[-1].split("_")[0]
    )  # Parse budget from run name

    # if args.RL_agent in ["PPO", "A2C", "DQN", "TRPO"]:
    #
    #     results_plotter.plot_results(
    #         [log_dir], budget, results_plotter.X_TIMESTEPS, f"{args.RL_agent} ContainerEnv"
    #     )

    log_dir_results = os.path.dirname(os.path.abspath(__file__)) + "/results/"
    os.makedirs(log_dir_results, exist_ok=True)

    #plt.savefig(log_dir_results + "train_reward_" + run_name + ".%s" % fig_format)

    env = ContainerEnv.from_json(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../configs/" + config_file
        )
    )
    if args.inf_eplen:
        env.max_episode_length = args.inf_eplen

    env = Monitor(env)
    obs = env.reset()

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


    # Keep track of state variables (volumes), actions, and rewards
    volumes = []
    actions = []
    rewards = []
    press_indices = []
    step = 0
    episode_length = 0
    plt.figure(figsize=(15, 10))

    # Run episode and render
    while True:
        episode_length += 1
        if args.RL_agent == "rulebased":
            action = model.predict(env_s = env, obs = obs[env.n_presses:].copy(), peak_vols = peak_vols)
        else:
            action, _ = model.predict(obs, deterministic=deterministic_policy)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        volumes.append(obs["Volumes"].copy())
        press_indices.append(info["press_indices"])
        # Toggle to render the episode
        if args.render_episode:
            env.render(volumes)
        rewards.append(reward)
        if done:
            break
        step += 1

    # Plot state variables during episode
    fig = plt.figure(figsize=(15, 10))
    env_unwrapped = env.unwrapped
    if deterministic_policy:
        fig.suptitle(f"Inference using a trained {args.RL_agent} agent", fontsize=16)
    else:
        fig.suptitle(
            f"Inference using a trained {args.RL_agent} agent with stochastic policy",
            fontsize=16,
        )

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.set_title("Volume", fontsize=14)
    ax2.set_title("Action", fontsize=14)
    ax3.set_title("Reward", fontsize=14)

    ax1.grid()
    ax2.grid()
    ax3.grid()

    ax1.set_ylim(top=40)
    ax1.set_xlim(left=0, right=args.inf_eplen)
    ax2.set_yticks(list(range(env_unwrapped.action_space.n)))
    ax2.set_xlim(left=0, right=args.inf_eplen)
    ax3.set_ylim(bottom=-0.1, top=1.05)
    ax3.set_xlim(left=0, right=args.inf_eplen)
    plt.xlabel("Time Steps", fontsize=12)

    default_color = "#1f77b4"  # Default Matplotlib blue color
    color_code = get_color_code()
    line_width = 3

    ## Plot volumes for each bunker
    for i in range(env_unwrapped.n_bunkers):
        ax1.plot(
            np.array(volumes)[:, i],
            linewidth=line_width,
            label=env_unwrapped.enabled_bunkers[i],
            color=color_code[env_unwrapped.enabled_bunkers[i]],
        )
    ax1.legend(bbox_to_anchor=(1.085, 1), loc="upper right", borderaxespad=0.0)

    ## Plot actions and rewards
    x_axis = range(episode_length)
    for i in x_axis:
        if actions[i] == 0:  # Action: "do nothing"
            ax2.scatter(i, actions[i], linewidth=line_width, color=default_color)
            ax3.scatter(
                i, rewards[i], linewidth=line_width, color=default_color, clip_on=False
            )
        else:
            ax2.scatter(
                i,
                actions[i],
                linewidth=line_width,
                color=color_code[env_unwrapped.enabled_bunkers[actions[i] - 1]],
                marker="^",
            )
            ax3.scatter(
                i,
                rewards[i],
                linewidth=line_width,
                color=color_code[env_unwrapped.enabled_bunkers[actions[i] - 1]],
                clip_on=False,
            )

    ax3.annotate(
        "Cum rew: {:.2f}".format(sum(rewards)),
        xy=(0.85, 0.9),
        xytext=(1.005, 0.9),
        xycoords="axes fraction",
        fontsize=13,
    )

    plt.subplots_adjust(hspace=0.5)

    if save_fig:
        # Save plot
        plt.savefig(
            log_dir_results + fig_name + ".%s" % fig_format,
            dpi="figure",
            format=fig_format,
        )

    #plt.show()


if __name__ == "__main__":
    # get the arguments
    args = parse_args()

    seeds = range(1, args.n_seeds+1)
    processes = [Process(target=inference, args=(s, args)) for s in seeds]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
