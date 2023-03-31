from typing import Optional, Union

import numpy as np

"""
Press models are linear of the form:
	pressing_time ~ slopes_press_(i, container_id) * n_bales
where i is the index of the requested press.
"""


class PressModel:
    def __init__(
        self,
        enabled_containers: Optional[list],
        slopes: Union[dict, list, np.ndarray] = {"C1-20": 264.9},
        offsets: Union[dict, list, np.ndarray] = {"C1-20": 106.798502},
    ):
        # "Overloading" of the constructor to allow dict or list/array input.
        # We assume all parameters are of the same type (either dict, list, or ndarray)
        # List and array parameters must be of length n_containers, even if a press is not connected to a container.
        # In that case, set that corresponding container's value to None
        if type(slopes) == dict and enabled_containers:
            self.slopes = np.array(
                [slopes.get(container, None) for container in enabled_containers]
            )
            self.offsets = np.array(
                [offsets.get(container, None) for container in enabled_containers]
            )
        elif type(slopes) == list:
            self.slopes = np.array(slopes)
            self.offsets = np.array(offsets)
        elif type(slopes) == np.ndarray:
            self.slopes = slopes
            self.offsets = offsets
        else:
            raise ValueError(
                "Parameters must be of types dict, list or ndarray. They must all be of the same type."
            )

    def get_pressing_time(
        self, current_time, time_prev_pressing_done, container_idx, n_bales
    ):
        """
        Calculate how long it takes until press 1 is free.

        Parameters
        ----------
        current_time: float
            the current time since the start of the simulation

        time_prev_pressing_done: float
            the time at which the last pressing process finished

        container_idx: int
            index of the container that will be emptied

        n_bales: int
            how many bales will be pressed

        Returns
        -------
        float
            Time in seconds until pressing will be finished. None if press is not free.
        """
        if current_time > time_prev_pressing_done or not self.slopes[container_idx]:
            # Press is not free or container not compatible with this press
            return None
        return (
            current_time + self.slopes[container_idx] * n_bales + self.offsets[container_idx]
        )