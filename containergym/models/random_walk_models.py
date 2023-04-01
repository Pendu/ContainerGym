import json

import numpy as np


class VectorizedRandomWalkModel:
    """A model for applying a random walk to multiple containers in parallel.

    Parameters
    ----------
    mus : np.array
        Vector of means for the normal distribution.

    sigmas : np.array
        Vector of standard deviations for the normal distribution.

    Attributes
    ----------
    mus : np.array
        Vector of means for the normal distribution.

    sigmas : np.array
        Vector of standard deviations for the normal distribution.

    Methods
    -------
    future_volume(volumes: np.array, timestep: int) -> np.array:
        Applies a random walk to all containers at once with a given timestep duration.
    """

    def __init__(self, mus, sigmas):
        self.mus = mus
        self.sigmas = sigmas

    def future_volume(self, volumes, timestep):
        """Applies a random walk to all containers at once with a given timestep duration.

        Parameters
        ----------
        volumes : np.array
            The current container volumes as a numpy vector.

        timestep : int
            The length of an environment step in seconds.

        Returns
        -------
        np.array
            Vector of the new container volumes after the random walk.
        """
        rw_mat = np.random.normal(
            loc=self.mus, scale=self.sigmas, size=(timestep, len(volumes))
        )

        return np.clip(volumes + rw_mat.sum(axis=0), a_min=0, a_max=None)
