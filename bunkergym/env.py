from math import floor
import gym
from gym import spaces
from bunkergym.reward import RewardEvaluator
from models.random_walk_models import VectorizedRandomWalkModel
from models.linear_press_models import PressModel
from typing import Union
import json
import numpy as np
import matplotlib.pyplot as plt


# TODO: Asserts, input validation
class SutcoEnv(gym.Env):
    """
    Simplified Sutco environment for OpenAI Gym.

    Attributes
    ----------
    max_episode_length: int
        maximum number of steps of a simulation episode 
    timestep: float
        length of a simulation step in seconds
    enabled_bunkers: list
        list of enabled bunkers. If dictionaries are provided for bunker parameters, 
        the names in this list must correspond to the keys in the parameter dictionaries.
    n_presses : int
        number of enabled presses
    min_starting_volume: float
        minimum volume with which to initialize a bunker's volume
    max_starting_volume: float
        maximum volume with which to initialize a bunker's volume
    failure_penalty: float
        negative reward to apply if a bunker reaches critical volume
    rw_mus: list
        list of mu values per bunker for the volume increasing random walk
    rw_sigmas: list
        list of sigma values per bunker for the volume increasing random walk
    max_volumes: Union[dict, list, np.ndarray]
        dict of key=bunker_id and value=max_volume for each bunker. 
        Alternatively, a list or array can be given if bunker names are anonymous.
    bale_sizes: Union[dict, list, np.ndarray]
        dict of key=bunker_id and value=bale_size for each bunker. 
        Alternatively, a list or array can be given if bunker names are anonymous.
    press_offsets: Union[dict, list, np.ndarray]
        constant time cost of actuating a press, regardless of the number of bales
    press_slopes: Union[dict, list, np.ndarray]
        slopes of the linear functions that determine pressing durations based on number of bales pressed
    reward_params: Union[dict, list]
        dict of key=bunker_id and a nested dict with keys "peaks", "heights", and "widths" for each bunker,
        which correspond to the ideal emptying volumes of the corresponding bunker and associated rewards.
        Alternatively, a list may be given that contains tuples of these values for each bunker:
        [[(b1_peak1, b1_height1, b1_width1), (b1_peak2, b1_height2, b1_width2)], 
        [(b2_peak1, b2_height1, b2_width1), (b2_peak2, b2_height2, b2_width2)]] 
        for the case of two bunkers and two peaks each.
    min_reward: float
        reward given to an agent, if it takes no action (action=0)
    """

    def __init__(self,
                 max_episode_length: int = 300,
                 timestep: float = 60,
                 enabled_bunkers: list = ['C1-20'],
                 n_presses: int = 1,
                 min_starting_volume: float = 0,
                 max_starting_volume: float = 30,
                 failure_penalty: float = -10,
                 rw_mus: Union[dict, list, np.ndarray] = {'C1-20': 0.005767754387396311},
                 rw_sigmas: Union[dict, list, np.ndarray] = {'C1-20': 0.055559018416836935},
                 max_volumes: Union[dict, list, np.ndarray] = {'C1-20': 32},
                 bale_sizes: Union[dict, list, np.ndarray] = {'C1-20': 27},
                 press_offsets: Union[dict, list, np.ndarray] = {'C1-20': 106.798502},
                 press_slopes: Union[dict, list, np.ndarray] = {'C1-20': 264.9},
                 reward_params: Union[dict, list] = {
                     "C1-20": {
                         "peaks": [26.71],
                         "heights": [1],
                         "widths": [2]
                     }},
                 min_reward: float = -1e-1):
        self.max_episode_length = max_episode_length
        self.enabled_bunkers = enabled_bunkers
        self.n_presses = n_presses
        self.min_starting_volume = min_starting_volume
        self.max_starting_volume = max_starting_volume
        self.failure_penalty = failure_penalty
        self.timestep = timestep
        self.press_model = PressModel(enabled_bunkers=enabled_bunkers, slopes=press_slopes, offsets=press_offsets)
        self.reward_evaluator = RewardEvaluator(bunker_params=reward_params, min_reward=min_reward)

        # Create RW object

        if type(enabled_bunkers) == list:
            mus = [rw_mus[bunker] for bunker in enabled_bunkers]
            sigmas = [rw_sigmas[bunker] for bunker in enabled_bunkers]
        elif (type(rw_mus) == list or type(rw_mus) == np.ndarray) and \
                (type(rw_sigmas) == list or type(rw_sigmas) == np.ndarray) and \
                (type(enabled_bunkers) == int):
            mus = rw_mus
            sigmas = rw_sigmas
        else:
            raise ValueError("""Could not parse the random walk parameters. 
                They must be of type dict, list, or np.ndarray. 
                Mus and sigmas must be provided in the same format.""")

        self.random_walk = VectorizedRandomWalkModel(mus=mus, sigmas=sigmas)
        # Overloading of the constructor based on given types
        # Max Volumes: Vector of critical/maximum volumes for just the enabled bunkers
        if type(max_volumes) == dict:
            self.max_volumes = np.array([max_volumes[bunker] for bunker in enabled_bunkers])
        elif type(max_volumes) == list:
            self.max_volumes = np.array(max_volumes)
        elif type(max_volumes) == np.ndarray:
            self.max_volumes = max_volumes

        # Bale Sizes: Vector of bale sizes for just the enabled bunkers
        if type(bale_sizes) == dict:
            self.bale_sizes = np.array([bale_sizes[bunker] for bunker in enabled_bunkers])
        elif type(bale_sizes) == list:
            self.bale_sizes = np.array(bale_sizes)
        elif type(bale_sizes) == np.ndarray:
            self.bale_sizes = bale_sizes

        # Number of bunkers
        if type(enabled_bunkers) == list and len(enabled_bunkers) > 0:
            self.n_bunkers = len(enabled_bunkers)
        else:  # Anonymous bunkers
            self.n_bunkers = len(max_volumes)
            self.enabled_bunkers = len(max_volumes)

        if n_presses < 1:
            raise ValueError("Must enable at least one press")

        # Create internal state object
        self.state = State(self.n_bunkers, self.n_presses)

        # Create action space based on how many bunkers exist
        self.action_space = spaces.Discrete(self.n_bunkers + 1)

        self.observation_space = spaces.Dict({"Volumes": spaces.Box(low=-100, high=100, shape=(self.n_bunkers,)),
                                              "Time presses will be free": spaces.Box(low=0, high=np.inf,
                                                                                      shape=(self.n_presses,))
                                              })
        # TODO: Move away from Dict observation space
        # self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.n_bunkers + self.n_presses))

    @classmethod
    def from_json(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            obj = cls(
                max_episode_length=data.get("MAX_EPISODE_LENGTH", 300),
                timestep=data.get("TIMESTEP", 60),
                enabled_bunkers=data.get("ENABLED_BUNKERS", ['C1-20']),
                n_presses=data.get("N_PRESSES", 1),
                min_starting_volume=data.get("MIN_STARTING_VOLUME", 0),
                max_starting_volume=data.get("MAX_STARTING_VOLUME", 30),
                failure_penalty=data.get("FAILURE_PENALTY", -10),
                rw_mus=data.get("RW_MUS", {"C1-20": 0.005767754387396311}),
                rw_sigmas=data.get("RW_SIGMAS", {"C1-20": 0.055559018416836935}),
                max_volumes=data.get("MAX_VOLUMES", {"C1-20": 32}),
                bale_sizes=data.get("BALE_SIZES", {"C1-20": 27}),
                press_offsets=data.get("PRESS_OFFSETS", {'C1-20': 106.798502}),
                press_slopes=data.get("PRESS_SLOPES", {'C1-20': 264.9}),
                reward_params=data.get("REWARD_PARAMS", {"C1-20": {
                    "peaks": [26.71],
                    "heights": [1],
                    "widths": [2]
                }}),
                min_reward=data.get("MIN_REWARD", -1e-1)
            )
            return obj

    def step(self, action):
        press_is_free = False  # Used to calculate reward at the end of the step
        emptied_volume = 0  # Current volume of the bunker that should be emptied, also for reward
        bunker_id = ""  # Name of the bunker that should be emptied
        bunker_idx = -1  # Index of the bunker that should be emptied
        press_idx = 0  # Index of the press that should be used

        # Get number of bales to be pressed
        n_bales = 0
        if action != 0:
            bunker_idx = (action - 1) % self.n_bunkers
            bunker_id = self.enabled_bunkers[bunker_idx] if type(self.enabled_bunkers) == list else bunker_idx
            n_bales = floor(self.state.volumes[bunker_idx] / self.bale_sizes[bunker_idx])
            emptied_volume = self.state.volumes[bunker_idx]

        # Fill bunkers now, so that after emptying, all volumes are increased and the emptied bunker set to 0
        self.state.volumes = self.random_walk.future_volume(self.state.volumes, self.timestep)

        if action > 0:
            # Choose a free press, if one exists
            free_presses = [idx for idx, time in enumerate(self.state.press_times) if time == 0]
            if free_presses:
                press_idx = np.random.choice(free_presses)  # Uniform random choice

                # Get pressing time
                t_pressing_ends = self.press_model.get_pressing_time(
                    current_time=0,
                    time_prev_pressing_done=self.state.press_times[press_idx],
                    bunker_idx=bunker_idx,
                    n_bales=n_bales)

                if t_pressing_ends is not None:
                    # Emptying is possible
                    press_is_free = True  # Used to calculate reward later
                    # Update state
                    self.state.volumes[bunker_idx] = 0
                    self.state.press_times[press_idx] = t_pressing_ends
            else:
                # Emptying is not possible 
                press_is_free = False

        # Decrease time counters, clip values below 0 to 0
        self.state.press_times = np.clip(self.state.press_times - self.timestep, a_min=0, a_max=None)

        # Increment episode length
        self.state.episode_length += 1

        # Calculate reward
        current_reward = self.reward_evaluator.reward(action, emptied_volume, press_is_free, bunker_id)

        # Check if episode is done. An episode ends if at least one bunker has exceeded max. vol.
        # or if max. episode length is reached.
        done = False

        if not (self.state.volumes < self.max_volumes).all():
            # At least one bunker has exceeded max. vol.
            done = True
            current_reward = self.failure_penalty  # apply failure penalty

        if self.state.episode_length == self.max_episode_length:
            # Max episode length is reached
            done = True

        info = {'press_indices': press_idx}
        return self.state.to_dict(), current_reward, done, info

    def reset(self):
        # Reset state
        self.state = State(self.n_bunkers, self.n_presses)
        self.state.reset(self.min_starting_volume, self.max_starting_volume)
        return self.state.to_dict()

    def render(self, y_volumes=None):

        for i in range(self.n_bunkers):
            y_values = np.array(y_volumes)[:, i]
            x_values = np.arange(len(y_volumes))
            # plt.plot(x_values,y_values,label=self.enabled_bunkers[i], linestyle='--')
            plt.plot(x_values, y_values, linestyle='--')
            # plt.legend()
            plt.axis([0, self.max_episode_length, 0, 40])
        plt.legend([i for i in self.enabled_bunkers])
        # ax.legend(["Value 1 ","Value 2","Value 3"])
        # plt.set_xlabel("Time")
        # plt.set_ylabel("Volumes")
        plt.title('Dynamic line graphs')
        plt.pause(0.0125)


class State():
    def __init__(self, enabled_bunkers, n_presses):
        self.episode_length = 0
        if type(enabled_bunkers) == list:
            self.volumes = np.zeros(len(enabled_bunkers))
        elif type(enabled_bunkers) == int:
            self.volumes = np.zeros(enabled_bunkers)
        else:
            raise ValueError("enabled_bunkers must be of type list or int")
        self.press_times = np.zeros(n_presses)

    def reset(self, min_volumes, max_volumes):
        self.volumes = np.random.uniform(min_volumes, max_volumes, size=len(self.volumes))

    def to_dict(self):
        return {"Volumes": self.volumes, "Time presses will be free": self.press_times}


if __name__ == "__main__":
    print(
        "Running env.py on its own is only meant for testing purposes. Please create an instance of the environment class and use it in your own code.")
    # Testing examples of the pre-configured environments with randomly chosen actions
    # TODO: Remove

    ### Full Sutco Setup
    # env = SutcoEnv.from_json("./configs/all_sutco_constants.json")
    # for n in range(100):
    #     print(env.step(np.random.choice(a=17, p=[0.84, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])))

    ### One Bunker One Press
    # env = SutcoEnv.from_json("./configs/1bunker_1press.json")
    # env = FlattenObservation(env)

    # for n in range(100):
    #     print(env.step(action=np.random.choice(2, p=[0.98, 0.02])))

    ### Five Bunkers Two Presses
    # env = SutcoEnv.from_json("./configs/5bunkers_2presses.json")
    # for n in range(100):
    #     print(env.step(action=np.random.choice(11, p=[0.90, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])))

    ### Alternative format with anonymous bunkers
    env = SutcoEnv.from_json("configs/list_constants_example.json")
    env = FlattenObservation(env)
    for n in range(100):
        print(env.step(action=np.random.choice(7, p=[0.94, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])))
