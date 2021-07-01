"""
Small utility functions used throughout the module
"""

# Author: Valentin F. Dannenberg / Ente


import numpy as np

from typing import Iterable

# TODO: Error checks?


def euclidean_norm(vec):
    return np.linalg.norm(vec, axis=1)
