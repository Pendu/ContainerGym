import json

import numpy as np


class VectorizedRandomWalkModel:
    def __init__(self, mus, sigmas):
        self.mus = mus
        self.sigmas = sigmas

    @classmethod
    def from_json(cls, filename):
        with open(filename) as json_file:
            data = json.load(json_file)

            mus = data["BUNKER_MUS"]
            sigmas = data["BUNKER_SIGMAS"]
            if type(mus) == dict:
                mus = list(mus.values())

            if type(sigmas) == dict:
                sigmas = list(sigmas.values())

            return cls(mus, sigmas)

    def future_volume(self, volumes, timestep):
        """Applies a random walk to all bunkers at once with a given timestep duration

        Parameters
        ----------
        volumes : np.array
            the current bunker volumes as a numpy vector
        timestep : int
            the length of an environment step in seconds

        Returns
        -------
        np.array
            Vector of the new bunker volumes after the random walk
        """
        rw_mat = np.random.normal(
            loc=self.mus, scale=self.sigmas, size=(timestep, len(volumes))
        )

        return np.clip(volumes + rw_mat.sum(axis=0), a_min=0, a_max=None)
