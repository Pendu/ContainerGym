import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import argparse
import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from statsmodels.distributions.empirical_distribution import ECDF

from containergym.env import ContainerEnv
from containergym.experiments.calculate_avg_cum_rew_n_rollouts import *

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
        "--best-seeds",
        nargs="+",
        type=int,
        default=[1,1,1],
        help="Best seeds to plot ecdf for PPO, TRPO and DQN respectively",
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
                  "C2-60": "#FF3E96",  # violetred1
                  "C2-70": "#FFD700",  # gold1
                  "C2-80": "#7FFF00",  # chartreuse1
                  "C2-90": "#D2691E",  # chocolate
                  }
    return color_code

def emptying_volumes_ecdfs(log_dirs, config_file=None, n_rollouts=15, deterministic_policy=True, fig_name= None,
                           format=None):
    """Performs n_rollouts rollouts of a policy and plots the ECDF of the emptying volumes collected over all rollouts
    for PPO, TRPO and DQN agents passed as a list in log_dirs.
    """
    # TODO: Caveat: assumes log_dirs contains paths to 3 models, namely PPO, TRPO and DQN in that order.

    env = ContainerEnv.from_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../configs/" + config_file))

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
            cumulative_reward = 0

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
    emptying_volumes = {k: [[], [], []] for k in env_unwrapped.enabled_bunkers}

    for j in range(len(models)):
        # Extract volumes corresponding to emptying actions
        for i, a in enumerate(actions[j]):
            if a > 0:
                emptying_volumes[env_unwrapped.enabled_bunkers[a - 1]][j].append(np.array(volumes[j])[i, a - 1])

    # Plot ECDFs
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
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../configs/" + config_file), 'r') as f:
            # Add optimal volumes
            data = json.load(f)
            optimal_vols = data.get("REWARD_PARAMS")[k]["peaks"]
            plt.vlines(x=optimal_vols[0], ymin=0, ymax=1, label=r'$v^*$', colors="grey", linewidth=line_width)
            if len(optimal_vols) > 1:
                plt.vlines(x=optimal_vols[1:], ymin=0, ymax=1, ls=":", label="Local optimum", colors="grey",
                           linewidth=line_width)

            # Add filling rate
            plt.annotate("Filling rate: {:.3f}".format(data.get("RW_MUS")[k]),
                         xy=(0.03, 0.6),
                         xycoords='axes fraction')

        plt.legend(loc=2)

        plt.savefig(fig_name + '_' + k + '.%s' % format, dpi='figure', format=format)

        plt.show()



if __name__ == '__main__':

    # get the arguments
    args = parse_args()

    n_steps = args.n_steps
    config_file = args.config_file
    fig_format = "svg"
    budget = args.budget
    config_file = args.config_file
    budget = args.budget
    ent_coef = args.ent_coeff
    gamma = args.gamma
    config_name = config_file.split('.json')[0]

    run_names = []
    for args.RL_agent in ["PPO", "TRPO","DQN"]:
        if args.RL_agent in ["PPO", "TRPO"]:
           if args.RL_agent == "PPO":
               seed = args.best_seeds[0]
           else:
               seed = args.best_seeds[1]
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
                    + str(n_steps))
        else:
            seed = args.best_seeds[2]
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
                    + str(gamma))
        run_names.append(run_name)

    log_dirs = [os.path.dirname(os.path.abspath(__file__)) + '/logs/' + run_name + '/' for run_name in run_names]

    emptying_volumes_ecdfs(log_dirs, config_file=config_file,n_rollouts=15,
                           fig_name='ecdf_emptying_vols_' + config_name + '_best_seeds', format=fig_format)