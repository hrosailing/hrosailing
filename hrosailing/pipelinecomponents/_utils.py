# pylint: disable=missing-module-docstring

import numpy as np


def _scaled_norm(norm, scal_factors):
    scaled = np.asarray(scal_factors)

    def s_norm(vec):
        return norm(scaled * vec)

    return s_norm


def _euclidean_norm(vec):
    return np.linalg.norm(vec, axis=1)


scaled_euclidean_norm = _scaled_norm(_euclidean_norm, [1 / 40, 1 / 360])
