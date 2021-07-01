"""
Small utility functions used throughout the module
"""

# Author: Valentin F. Dannenberg / Ente


import numpy as np

from typing import Iterable

# TODO: Error checks?


def polar_to_kartesian(arr):
    return np.column_stack(
        (arr[:, 1] * np.cos(arr[:, 0]),
         arr[:, 1] * np.sin(arr[:, 0])))


def euclidean_norm(vec):
    return np.linalg.norm(vec, axis=1)
