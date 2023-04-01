import json
from math import floor
from typing import Union

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

from containergym.reward import RewardEvaluator
from containergym.models.linear_press_models import PressModel
from containergym.models.random_walk_models import VectorizedRandomWalkModel


# TODO: Asserts, input validation
class ContainerEnv(gym.Env):
    """
    Simplified ContainerEnv environment for OpenAI Gym.

    Attributes
    ----------
    max_episode_length: int
        maximum number of steps of a simulation episode
    timestep: float
        length of a simulation step in seconds
    enabled_containers: list
        list of enabled containers. If dictionaries are provided for container parameters,
        the names in this list must correspond to the keys in the parameter dictionaries.
    n_presses : int
        number of enabled presses
    min_starting_volume: float
        minimum volume with which to initialize a container's volume
    max_starting_volume: float
        maximum volume with which to initialize a container's volume
    failure_penalty: float
        negative reward to apply if a container reaches critical volume
    rw_mus: list
        list of mu values per container for the volume increasing random walk
    rw_sigmas: list
        list of sigma values per container for the volume increasing random walk
    max_volumes: Union[dict, list, np.ndarray]
        dict of key=container_id and value=max_volume for each container.
        Alternatively, a list or array can be given if container names are anonymous.
    bale_sizes: Union[dict, list, np.ndarray]
        dict of key=container_id and value=bale_size for each container.
        Alternatively, a list or array can be given if container names are anonymous.
    press_offsets: Union[dict, list, np.ndarray]
        constant time cost of actuating a press, regardless of the number of bales
    press_slopes: Union[dict, list, np.ndarray]
        slopes of the linear functions that determine pressing durations based on number of bales pressed
    reward_params: Union[dict, list]
        dict of key=container_id and a nested dict with keys "peaks", "heights", and "widths" for each container,
        which correspond to the ideal emptying volumes of the corresponding container and associated rewards.
        Alternatively, a list may be given that contains tuples of these values for each container:
        [[(b1_peak1, b1_height1, b1_width1), (b1_peak2, b1_height2, b1_width2)],
        [(b2_peak1, b2_height1, b2_width1), (b2_peak2, b2_height2, b2_width2)]]
        for the case of two containers and two peaks each.
    min_reward: float
        reward given to an agent, if it takes no action (action=0)
    dict_observation: bool
        flag to set true if dict observations are desired. Otherwise the box observation space is chosen.
    """

    def __init__(
        self,
        max_episode_length: int = 300,
        timestep: float = 60,
        enabled_containers: list = ["C1-20"],
        n_presses: int = 1,
        min_starting_volume: float = 0,
        max_starting_volume: float = 30,
        failure_penalty: float = -10,
        rw_mus: Union[dict, list, np.ndarray] = {"C1-20": 0.005767754387396311},
        rw_sigmas: Union[dict, list, np.ndarray] = {"C1-20": 0.055559018416836935},
        max_volumes: Union[dict, list, np.ndarray] = {"C1-20": 32},
        bale_sizes: Union[dict, list, np.ndarray] = {"C1-20": 27},
        press_offsets: Union[dict, list, np.ndarray] = {"C1-20": 106.798502},
        press_slopes: Union[dict, list, np.ndarray] = {"C1-20": 264.9},
        reward_params: Union[dict, list] = {
            "C1-20": {"peaks": [26.71], "heights": [1], "widths": [2]}
        },
        min_reward: float = -1e-1,
        dict_observation: bool = True,
    ):
        self.max_episode_length = max_episode_length
        self.enabled_containers = enabled_containers
        self.n_presses = n_presses
        self.min_starting_volume = min_starting_volume
        self.max_starting_volume = max_starting_volume
        self.failure_penalty = failure_penalty
        self.timestep = timestep
        self.press_model = PressModel(
            enabled_containers=enabled_containers,
            slopes=press_slopes,
            offsets=press_offsets,
        )
        self.reward_evaluator = RewardEvaluator(
            container_params=reward_params, min_reward=min_reward
        )

        # Create RW object

        if type(enabled_containers) == list:
            mus = [rw_mus[container] for container in enabled_containers]
            sigmas = [rw_sigmas[container] for container in enabled_containers]
        elif (
            (type(rw_mus) == list or type(rw_mus) == np.ndarray)
            and (type(rw_sigmas) == list or type(rw_sigmas) == np.ndarray)
            and (type(enabled_containers) == int)
        ):
            mus = rw_mus
            sigmas = rw_sigmas
        else:
            raise ValueError(
                """Could not parse the random walk parameters. 
                They must be of type dict, list, or np.ndarray. 
                Mus and sigmas must be provided in the same format."""
            )

        self.random_walk = VectorizedRandomWalkModel(mus=mus, sigmas=sigmas)
        # Overloading of the constructor based on given types
        # Max Volumes: Vector of critical/maximum volumes for just the enabled containers
        if type(max_volumes) == dict:
            self.max_volumes = np.array(
                [max_volumes[container] for container in enabled_containers]
            )
        elif type(max_volumes) == list:
            self.max_volumes = np.array(max_volumes)
        elif type(max_volumes) == np.ndarray:
            self.max_volumes = max_volumes

        # Bale Sizes: Vector of bale sizes for just the enabled containers
        if type(bale_sizes) == dict:
            self.bale_sizes = np.array(
                [bale_sizes[container] for container in enabled_containers]
            )
        elif type(bale_sizes) == list:
            self.bale_sizes = np.array(bale_sizes)
        elif type(bale_sizes) == np.ndarray:
            self.bale_sizes = bale_sizes

        # Number of containers
        if type(enabled_containers) == list and len(enabled_containers) > 0:
            self.n_containers = len(enabled_containers)
        else:  # Anonymous containers
            self.n_containers = len(max_volumes)
            self.enabled_containers = len(max_volumes)

        if n_presses < 1:
            raise ValueError("Must enable at least one press")

        # Create internal state object
        self.state = State(self.n_containers, self.n_presses)

        # Create action space based on how many containers exist
        self.action_space = spaces.Discrete(self.n_containers + 1)

        # Create observation space
        self.dict_observation = dict_observation
        if dict_observation:
            self.observation_space = spaces.Dict(
                {
                    "Volumes": spaces.Box(
                        low=-100, high=100, shape=(self.n_containers,)
                    ),
                    "Time presses will be free": spaces.Box(
                        low=0, high=np.inf, shape=(self.n_presses,)
                    ),
                }
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=np.inf, shape=(self.n_containers + self.n_presses)
            )

    @classmethod
    def from_json(cls, filepath):
        """
        Constructs a ContainerEnv object from a JSON file.

        Parameters
        ----------
        cls : type
            The class of the object being created.
        filepath : str
            The filepath of the JSON file to be read.

        Returns
        -------
        ContainerEnv
            An object of the class 'ContainerEnv'.
        """

        with open(filepath, "r") as f:
            data = json.load(f)
            obj = cls(
                max_episode_length=data.get("MAX_EPISODE_LENGTH", 300),
                timestep=data.get("TIMESTEP", 60),
                enabled_containers=data.get("ENABLED_CONTAINERS", ["C1-20"]),
                n_presses=data.get("N_PRESSES", 1),
                min_starting_volume=data.get("MIN_STARTING_VOLUME", 0),
                max_starting_volume=data.get("MAX_STARTING_VOLUME", 30),
                failure_penalty=data.get("FAILURE_PENALTY", -10),
                rw_mus=data.get("RW_MUS", {"C1-20": 0.005767754387396311}),
                rw_sigmas=data.get("RW_SIGMAS", {"C1-20": 0.055559018416836935}),
                max_volumes=data.get("MAX_VOLUMES", {"C1-20": 32}),
                bale_sizes=data.get("BALE_SIZES", {"C1-20": 27}),
                press_offsets=data.get("PRESS_OFFSETS", {"C1-20": 106.798502}),
                press_slopes=data.get("PRESS_SLOPES", {"C1-20": 264.9}),
                reward_params=data.get(
                    "REWARD_PARAMS",
                    {"C1-20": {"peaks": [26.71], "heights": [1], "widths": [2]}},
                ),
                min_reward=data.get("MIN_REWARD", -1e-1),
            )
            return obj

    def step(self, action):
        """
        Perform a single step in the environment.

        Parameters
        ----------
        action : int
            The action to be performed.

        Returns
        -------
        tuple
            A tuple of the observation, reward, done flag, and info dictionary.


        Notes
        -----
        This method updates the environment state according to the specified `action`, and returns the resulting observation,
        reward, and done flag, as well as additional information in the `info` dictionary. If `action` is 0, the method does
        nothing and returns the current observation and zero reward. Otherwise, it selects a container to be emptied based on
        the value of `action`, and uses a free press to compress the material in the container. The reward is calculated based on
        whether the emptying was successful, the amount of material that was emptied, and the current state of the press. The
        episode is terminated if any container exceeds its maximum volume or the maximum episode length is reached.

        """

        press_is_free = False  # Used to calculate reward at the end of the step
        emptied_volume = (
            0  # Current volume of the container that should be emptied, also for reward
        )
        container_id = ""  # Name of the container that should be emptied
        container_idx = -1  # Index of the container that should be emptied
        press_idx = 0  # Index of the press that should be used

        # Get number of bales to be pressed
        n_bales = 0
        if action != 0:
            container_idx = (action - 1) % self.n_containers
            container_id = (
                self.enabled_containers[container_idx]
                if type(self.enabled_containers) == list
                else container_idx
            )
            n_bales = floor(
                self.state.volumes[container_idx] / self.bale_sizes[container_idx]
            )
            emptied_volume = self.state.volumes[container_idx]

        # Fill containers now, so that after emptying, all volumes are increased and the emptied container set to 0
        self.state.volumes = self.random_walk.future_volume(
            self.state.volumes, self.timestep
        )

        if action > 0:
            # Choose a free press, if one exists
            free_presses = [
                idx for idx, time in enumerate(self.state.press_times) if time == 0
            ]
            if free_presses:
                press_idx = np.random.choice(free_presses)  # Uniform random choice

                # Get pressing time
                t_pressing_ends = self.press_model.get_pressing_time(
                    current_time=0,
                    time_prev_pressing_done=self.state.press_times[press_idx],
                    container_idx=container_idx,
                    n_bales=n_bales,
                )

                if t_pressing_ends is not None:
                    # Emptying is possible
                    press_is_free = True  # Used to calculate reward later
                    # Update state
                    self.state.volumes[container_idx] = 0
                    self.state.press_times[press_idx] = t_pressing_ends
            else:
                # Emptying is not possible
                press_is_free = False

        # Decrease time counters, clip values below 0 to 0
        self.state.press_times = np.clip(
            self.state.press_times - self.timestep, a_min=0, a_max=None
        )

        # Increment episode length
        self.state.episode_length += 1

        # Calculate reward
        current_reward = self.reward_evaluator.reward(
            action, emptied_volume, press_is_free, container_id
        )

        # Check if episode is done. An episode ends if at least one container has exceeded max. vol.
        # or if max. episode length is reached.
        done = False

        if not (self.state.volumes < self.max_volumes).all():
            # At least one container has exceeded max. vol.
            done = True
            current_reward = self.failure_penalty  # apply failure penalty

        if self.state.episode_length == self.max_episode_length:
            # Max episode length is reached
            done = True

        info = {"press_indices": press_idx}
        if self.dict_observation:
            obs = self.state.to_dict()
        else:
            obs = self.state.to_box()
        return obs, current_reward, done, info

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns
        -------
        dict
            A dictionary containing the initial observation of the environment.
        """
        self.state = State(self.n_containers, self.n_presses)
        self.state.reset(self.min_starting_volume, self.max_starting_volume)
        return self.state.to_dict()

    def render(self, y_volumes=None):
        """
        Plot the live volumes of containers during evaluation.

        Parameters
        ----------
        y_volumes : array-like or None, shape (n_steps, n_containers), optional
            The input volumes of each container for each time step. If not provided, the current state
            volumes will be used instead.

        Returns
        -------
        None

        """

        for i in range(self.n_containers):
            y_values = np.array(y_volumes)[:, i]
            x_values = np.arange(len(y_volumes))
            plt.plot(x_values, y_values, linestyle="--")
            plt.axis([0, self.max_episode_length, 0, 35])
        plt.legend([i for i in self.enabled_containers])
        plt.xlabel("Time", fontsize=16)
        plt.ylabel("Volumes", fontsize=16)
        plt.title("Dynamic volumes of Containers", fontsize=20)
        plt.pause(0.0125)


class State:
    """
    A class representing the state of the Presses environment.

    Parameters
    ----------
    enabled_containers : list or int
        List of enabled containers or number of enabled containers
    n_presses : int
        Number of presses

    Attributes
    ----------
    episode_length : int
        Length of the current episode
    volumes : numpy.ndarray
        Volumes of each container
    press_times : numpy.ndarray
        Press times for each press

    Methods
    -------
    reset(min_volumes, max_volumes)
        Reset volumes to random values between min_volumes and max_volumes
    to_dict()
        Convert the class into a dictionary
    to_box()
        Convert the class into a box-vector

    """

    def __init__(self, enabled_containers, n_presses):
        """ "
        Initializes the State object.

        Parameters
        ----------
        enabled_containers : list or int
            List of enabled containers or number of enabled containers
        n_presses : int
            Number of presses
        """
        self.episode_length = 0
        if type(enabled_containers) == list:
            self.volumes = np.zeros(len(enabled_containers))
        elif type(enabled_containers) == int:
            self.volumes = np.zeros(enabled_containers)
        else:
            raise ValueError("enabled_containers must be of type list or int")
        self.press_times = np.zeros(n_presses)

    def reset(self, min_volumes, max_volumes):
        """
        Resets the volumes of each container to random values within the given range.

        Parameters
        ----------
        min_volumes : float
            Minimum volume of each container
        max_volumes : float
            Maximum volume of each container
        """
        self.volumes = np.random.uniform(
            min_volumes, max_volumes, size=len(self.volumes)
        )

    def to_dict(self):
        """
        Converts the class into a dictionary.

        Returns
        -------
        dict
            A dictionary with the volumes and press times
        """
        return {"Volumes": self.volumes, "Time presses will be free": self.press_times}

    def to_box(self):
        """
        Converts the class into a box-vector.

        Returns
        -------
        numpy.ndarray
            A box-vector with the volumes and press times
        """
        return np.concatenate(self.volumes, self.press_times)
