import json
from typing import Union

import numpy as np


class RewardEvaluator:
    """
    A class to evaluate the reward of a given action in the RL environment.

    Parameters
    ----------
    container_params : Union[dict, list]
        The parameters for the Gaussian reward function. Can be either a dictionary with keys
        'peaks', 'heights', and 'widths', or a list of tuples where each tuple contains the peak,
        height, and width values.
    min_reward : float
        The minimum reward value.

    Methods
    -------
    from_json(filename)
        Returns a RewardEvaluator object with parameters loaded from a JSON file.

    gaussian_reward_dict_params(action, current_volume, press_is_free, container_id)
        Calculates the reward for an action given the current container volume and whether the
        press is free or not. Uses the parameters in container_params (dictionary form).

    gaussian_reward_list_params(action, current_volume, press_is_free, container_id=None)
        Calculates the reward for an action given the current container volume and whether the
        press is free or not. Uses the parameters in container_params (list form).

    reward(action, current_volume, press_is_free, container_id)
        Calculates the reward for an action given the current container volume and whether the
        press is free or not. Depending on the type of container_params, calls either the
        gaussian_reward_dict_params or gaussian_reward_list_params function.

    Attributes
    ----------
    container_params : Union[dict, list]
        The parameters for the Gaussian reward function.
    min_reward : float
        The minimum reward value.
    n_containers : int
        The number of containers in the environment.
    """

    def __init__(self, container_params: Union[dict, list], min_reward: float):
        # "overloading" constructor to allow params in dict or list form
        # if type(container_params) == dict:
        #     self.reward = self.gaussian_reward_dict_params
        # elif type(container_params) == list:
        #     self.reward = self.gaussian_reward_list_params
        self.container_params = container_params
        self.min_reward = min_reward
        self.n_containers = len(container_params)

    @classmethod
    def from_json(cls, filename):
        """
        Returns a RewardEvaluator object with parameters loaded from a JSON file.

        Parameters
        ----------
        filename : str
            The name of the JSON file containing the reward parameters.

        Returns
        -------
        RewardEvaluator
            A RewardEvaluator object with parameters loaded from the JSON file.
        """
        with open(filename) as json_file:
            data = json.load(json_file)

            container_params = data["REWARD_PARAMS"]
            min_reward = data["MIN_REWARD"]

            return cls(container_params, min_reward)

    def gaussian_reward_dict_params(
        self, action, current_volume, press_is_free, container_id
    ):
        """
        Calculates the reward for an action given the current container volume and whether the
        press is free or not. Uses the parameters in container_params (dictionary form).

        Parameters
        ----------
        action : int
            The action taken by the agent at time t.
        current_volume : float
            The volume of the container on which the action is performed at time t.
        press_is_free : bool
            True if the press is free (emptying is possible) at time t. False otherwise.
        container_id : int
            The ID of the container on which the action is performed.

        Returns
        -------
        float
            A value between min_reward and 1, representing the reward for the given action.
        """
        reward = 0

        if action > 0:
            if current_volume == 0.0 or not press_is_free:
                return self.min_reward

            params = self.container_params[container_id]

            for i in range(len(params["peaks"])):
                peak = params["peaks"][i]
                height = params["heights"][i]
                width = params["widths"][i]

                reward += (height - self.min_reward) * np.exp(
                    -(current_volume - peak)
                    * (current_volume - peak)
                    / (2.0 * width * width)
                )

            reward += self.min_reward
        return reward

    def gaussian_reward_list_params(
        self, action, current_volume, press_is_free, container_id=None
    ):
        """
        Calculate reward based on a Gaussian function for a list of container parameters.

        Parameters
        ----------
        action : int
            The action taken by the agent at time t.
        current_volume : float
            The volume of the container at time t.
        press_is_free : bool
            True if the press is free (emptying is possible) at time t. False otherwise.
        container_id : int or None, optional
            The ID of the container on which the action is performed. Default is None.

        Returns
        -------
        float
            A value between min_reward and 1.

        Notes
        -----
        The Gaussian function is defined by the parameters of the container, which are stored in self.container_params.

        """
        reward = 0

        if action > 0:
            if current_volume == 0.0 or not press_is_free:
                return self.min_reward

            container_idx = (action - 1) % self.n_containers
            container_params = self.container_params[container_idx]

            for i in range(len(container_params)):
                peak = container_params[i][0]
                height = container_params[i][1]
                width = container_params[i][2]

                reward += (height - self.min_reward) * np.exp(
                    -(current_volume - peak)
                    * (current_volume - peak)
                    / (2.0 * width * width)
                )

            reward += self.min_reward
        return reward

    def reward(self, action, current_volume, press_is_free, container_id):
        """
        Calculate reward based on a Gaussian function for a container.

        Parameters
        ----------
        action : int
            The action taken by the agent at time t.
        current_volume : float
            The volume of the container at time t.
        press_is_free : bool
            True if the press is free (emptying is possible) at time t. False otherwise.
        container_id : int
            The ID of the container on which the action is performed.

        Returns
        -------
        float
            A value between min_reward and 1.

        Notes
        -----
        The Gaussian function is defined by the parameters of the container with the given ID, which are stored in self.container_params.

        """

        return self.gaussian_reward_dict_params(
            action, current_volume, press_is_free, container_id
        )
