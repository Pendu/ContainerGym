import json
import numpy as np
import os

def main():
    config_file = '5containers_2presses_2.json'
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../containergym/configs/" + config_file)
    my_rule_based_agent = RuleBasedAgent(filepath)
    print(my_rule_based_agent.enabled_containers)
    print(my_rule_based_agent.best_volumes)

class RuleBasedAgent:
    """
    A rule-based agent that selects a container based on their current volume difference
    with the best volumes defined in the configuration file.

    Parameters
    ----------
    filepath : str
        The path to the JSON configuration file containing the reward parameters
        and enabled containers.
    vol_margin : int, optional
        The minimum difference in volume for a container to be considered for selection, by default 1.

    Attributes
    ----------
    vol_margin : int
        The minimum difference in volume for a container to be considered for selection.
    enabled_containers : list
        A list of enabled containers read from the configuration file.
    best_volumes : list
        A list of best volumes read from the reward parameters in the configuration file.

    Methods
    -------
    predict(obs, deterministic=True)
        Predicts the next action of the agent based on the observation received.

    """
    def __init__(self, filepath, vol_margin=1):
        """
          Initializes a new instance of the RuleBasedAgent.

          Parameters
          ----------
          filepath : str
              The path to the JSON configuration file containing the reward parameters
              and enabled containers.
          vol_margin : int, optional
              The minimum difference in volume for a container to be considered for selection, by default 1.
          """
        self.vol_margin = vol_margin
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.enabled_containers = data.get("ENABLED_CONTAINERS")
            reward_params = data.get("REWARD_PARAMS")
            self.best_volumes = []
            for container in self.enabled_containers:
                self.best_volumes.append(reward_params[container]["peaks"][0])

    def predict(self, obs, deterministic=True):
        """
        Predicts the next action of the agent based on the observation received.

        Parameters
        ----------
        obs : dict
            A dictionary containing the observation data.
        deterministic : bool, optional
            Determines if the agent should select the action deterministically,
            by default True.

        Returns
        -------
        tuple
            A tuple containing the selected container and a placeholder value (0).
        """
        differences = np.array(self.best_volumes) - np.array(obs["Volumes"])
        candidate_containers = np.where(differences <= self.vol_margin)[0]
        if len(candidate_containers) == 0:
            return 0, 0  # 2nd value is returned to be compatible with test code
        return candidate_containers[0] + 1, 0


if __name__ == "__main__":
    main()