import json
from typing import Union

import numpy as np


class RewardEvaluator:
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
        with open(filename) as json_file:
            data = json.load(json_file)

            container_params = data["REWARD_PARAMS"]
            min_reward = data["MIN_REWARD"]

            return cls(container_params, min_reward)

    def gaussian_reward_dict_params(
        self, action, current_volume, press_is_free, container_id
    ):  # TODO: Remove
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
        The reward function is a Gaussian function. Depending on container_id, calls corresponding Gaussian reward function.
        The reward is 1 if the action taken by the agent leads to an empty container, and min_reward otherwise.

        :param action: Action taken by the agent at time t
        :param current_volume: container volume at time t
        :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
        :param container_id: ID of container on which the action is performed
        :return: A value between min_reward and 1
        """

        return self.gaussian_reward_dict_params(
            action, current_volume, press_is_free, container_id
)