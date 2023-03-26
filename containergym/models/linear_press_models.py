from typing import Optional, Union

import numpy as np

"""
Press models are linear of the form:
	pressing_time ~ slopes_press_(i, bunker_id) * n_bales
where i is the index of the requested press.
"""


class PressModel:
    def __init__(
        self,
        enabled_bunkers: Optional[list],
        slopes: Union[dict, list, np.ndarray] = {"C1-20": 264.9},
        offsets: Union[dict, list, np.ndarray] = {"C1-20": 106.798502},
    ):
        # "Overloading" of the constructor to allow dict or list/array input.
        # We assume all parameters are of the same type (either dict, list, or ndarray)
        # List and array parameters must be of length n_bunkers, even if a press is not connected to a bunker.
        # In that case, set that corresponding bunker's value to None
        if type(slopes) == dict and enabled_bunkers:
            self.slopes = np.array(
                [slopes.get(bunker, None) for bunker in enabled_bunkers]
            )
            self.offsets = np.array(
                [offsets.get(bunker, None) for bunker in enabled_bunkers]
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
        self, current_time, time_prev_pressing_done, bunker_idx, n_bales
    ):
        """
        Calculate how long it takes until press 1 is free.

        Parameters
        ----------
        current_time: float
            the current time since the start of the simulation

        time_prev_pressing_done: float
            the time at which the last pressing process finished

        bunker_idx: int
            index of the bunker that will be emptied

        n_bales: int
            how many bales will be pressed

        Returns
        -------
        float
            Time in seconds until pressing will be finished. None if press is not free.
        """
        if current_time > time_prev_pressing_done or not self.slopes[bunker_idx]:
            # Press is not free or bunker not compatible with this press
            return None
        return (
            current_time + self.slopes[bunker_idx] * n_bales + self.offsets[bunker_idx]
        )
