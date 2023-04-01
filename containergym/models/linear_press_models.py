from typing import Optional, Union

import numpy as np

"""
Press models are linear of the form:
	pressing_time ~ slopes_press_(i, container_id) * n_bales
where i is the index of the requested press.
"""


class PressModel:
    """
    A class that models press behavior based on container and pressing parameters.

    Parameters
    ----------
    enabled_containers : list or None
        List of containers that are connected to a press. If None, no containers are connected.

    slopes : dict or list or np.ndarray, optional
        Dictionary, list, or numpy array with the slopes for each press. If a dictionary is provided,
        it must have a key for each container that is connected to a press. If a list or array is provided,
        it must have length n_containers, even if a press is not connected to a container. In that case,
        set that corresponding container's value to None.

    offsets : dict or list or np.ndarray, optional
        Dictionary, list, or numpy array with the offsets for each press. If a dictionary is provided,
        it must have a key for each container that is connected to a press. If a list or array is provided,
        it must have length n_containers, even if a press is not connected to a container. In that case,
        set that corresponding container's value to None.

    Attributes
    ----------
    slopes : np.ndarray
        Array with the slopes for each press.

    offsets : np.ndarray
        Array with the offsets for each press.

    Methods
    -------
    get_pressing_time(current_time, time_prev_pressing_done, container_idx, n_bales)
        Calculate how long it takes until press is free.
    """
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
        Calculate how long it takes until press is free.

        Parameters
        ----------
        current_time : float
            The current time since the start of the simulation.

        time_prev_pressing_done : float
            The time at which the last pressing process finished.

        container_idx : int
            The index of the container that will be emptied.

        n_bales : int
            How many bales will be pressed.

        Returns
        -------
        float or None
            Time in seconds until pressing will be finished. None if press is not free.
        """
        if current_time > time_prev_pressing_done or not self.slopes[container_idx]:
            # Press is not free or container not compatible with this press
            return None
        return (
            current_time + self.slopes[container_idx] * n_bales + self.offsets[container_idx]
        )