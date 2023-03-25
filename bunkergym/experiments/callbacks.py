import os
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# This callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

'''
# ENV LOGGER CURRENTLY UNTESTED
class EnvLogger(BaseCallback):
    """
    Callback for logging episodes of the SutcoEnv during training.

    :param log_frequency: (int) How many episodes to wait before logging an episode. (1 -> log every episode, 5 -> log every 5th episode)
    """
    def __init__(self, log_frequency, log_dir):
        super(EnvLogger, self).__init__()
        self.log_frequency = log_frequency
        self.log_dir = log_dir
        # create dir if not exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.episode_num = 1
    
    def _init_callback(self) -> None:
        bunkers_raw = self.model.env.get_attr("bunker_ids")
        self.bunkers = [x for x, y in bunkers_raw[0]]
        # Create output frame
        self.df = pd.DataFrame(columns=['action', 'reward'] + self.bunkers)

    def _on_step(self):
        if self.episode_num % self.log_frequency == 0:
            # Check done
            if self.locals['dones'][0]:
                self.save_csv()   
                self.episode_num += 1
                return # Stablebaselines calls reset() before the callback, so this step has invalid values

            # Write action and reward
            row_dict = dict()
            row_dict['action'] = self.locals['actions'][0]
            row_dict['reward'] = self.locals['rewards'][0]

            # Write volumes
            obs = self.locals['new_obs']
            volumes = obs['Volumes'][0]
            for i, bunker in enumerate(self.bunkers):
                row_dict[bunker] = volumes[i]
            self.df = self.df.append(row_dict, ignore_index=True)

        # Count episodes
        done = self.locals['dones'][0]
        if self.locals['dones'][0]:
            self.episode_num += 1
        return
    
    def save_csv(self):
        # Save to file
        self.df.to_csv(self.log_dir + f"episode_{self.episode_num}.csv", index=False)
        # Reset logged data
        self.df = self.df[0:0]
        return
'''