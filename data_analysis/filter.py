"""
Collection of various filter functions
to filter weighted points based on their
weight.

Each function will return a boolean array
describing which points are filtered out
and which are not
"""

# Author: Valentin F. Dannenberg / Ente


from abc import ABC, abstractmethod
import numpy as np

from exceptions import ProcessingException


class Filter(ABC):

    @abstractmethod
    def filter(self, weights):
        pass


class QuantileFilter(Filter):

    def __init__(self, percent=25):
        if percent < 0 or percent > 100:
            raise ProcessingException("")

        self._per = percent

    def filter(self, weights):
        weights = np.asarray(weights)
        return weights >= np.percentile(weights, self._per)


class BoundFilter(Filter):

    def __init__(self, upper_bound=1, lower_bound=0.5):
        self._u_b = upper_bound
        self._l_b = lower_bound

    def filter(self, weights):
        weights = np.asarray(weights)
        mask_1 = weights >= self._l_b
        mask_2 = weights <= self._u_b
        return mask_1 & mask_2

