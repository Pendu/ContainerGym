import numpy as np
import json
from typing import Union

class RewardEvaluator():
    def __init__(self, bunker_params: Union[dict, list], min_reward: float):
        # "overloading" constructor to allow params in dict or list form
        # if type(bunker_params) == dict:
        #     self.reward = self.gaussian_reward_dict_params
        # elif type(bunker_params) == list:
        #     self.reward = self.gaussian_reward_list_params
        self.bunker_params = bunker_params
        self.min_reward = min_reward
        self.n_bunkers = len(bunker_params)

    @classmethod
    def from_json(cls, filename):
        with open(filename) as json_file:
            data = json.load(json_file)

            bunker_params = data['REWARD_PARAMS']
            min_reward = data['MIN_REWARD']
            
            return cls(bunker_params, min_reward)

    def gaussian_reward_dict_params(self, action, current_volume, press_is_free, bunker_id): #TODO: Remove
        reward = 0

        if action > 0:
            if current_volume == 0. or not press_is_free:
                return self.min_reward
            
            params = self.bunker_params[bunker_id]

            for i in range(len(params["peaks"])):
                peak = params["peaks"][i]
                height = params["heights"][i]
                width = params["widths"][i]

                reward += (height - self.min_reward) * np.exp(-(current_volume - peak) * (current_volume - peak) / (2. * width * width))

            reward += self.min_reward
        return reward

    def gaussian_reward_list_params(self, action, current_volume, press_is_free, bunker_id=None):
        reward = 0

        if action > 0:
            if current_volume == 0. or not press_is_free:
                return self.min_reward
            
            bunker_idx = (action - 1) % self.n_bunkers
            bunker_params = self.bunker_params[bunker_idx]

            for i in range(len(bunker_params)):
                peak = bunker_params[i][0]
                height = bunker_params[i][1]
                width = bunker_params[i][2]

                reward += (height - self.min_reward) * np.exp(-(current_volume - peak) * (current_volume - peak) / (2. * width * width))

            reward += self.min_reward
        return reward

    def reward(self, action, current_volume, press_is_free, bunker_id):
        """

        General reward function. Depending on bunker_id, calls corresponding Gaussian reward function.

        :param action: Action taken by the agent at time t
        :param current_volume: Bunker volume at time t
        :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
        :param bunker_id: ID of bunker on which the action is performed
        :return: Reward value between min_reward and 1
        """

        return self.gaussian_reward_dict_params(action, current_volume, press_is_free, bunker_id)
    