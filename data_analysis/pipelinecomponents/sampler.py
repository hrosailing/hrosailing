"""

"""

import numpy as np
import random

from abc import ABC, abstractmethod
from scipy.spatial import ConvexHull


class Sampler(ABC):

    @abstractmethod
    def sample(self, pts):
        pass


class RandomSampler(Sampler):

    def __init__(self, no_samples):
        self._no_samples = no_samples

    def sample(self, pts):
        random.seed()
        pts = np.asarray(pts)[:, :2]
        ineq = ConvexHull(pts).equations
        samples = []
        ws_bound, wa_bound = _create_bounds(pts)
        while len(samples) < self._no_samples:
            ws = random.uniform(ws_bound[0], ws_bound[1])
            wa = random.uniform(wa_bound[0], wa_bound[1])
            if np.all(ineq[:, :2] @ np.array([ws, wa]) <= -ineq[:, 2]):
                samples.append([ws, wa])

        return np.array(samples)


def _create_bounds(pts):
    ws_bound = (np.amin(pts[:, 0]), np.amax(pts[:, 0]))
    wa_bound = (np.amin(pts[:, 1]), np.amax(pts[:, 1]))
    return ws_bound, wa_bound


class FibonacciSampler(Sampler):

    def __init__(self, no_samples):
        self._no_samples = no_samples

    def sample(self, pts):
        pass


class ArchimedianSampler(Sampler):

    def __init__(self, no_samples):
        self._no_samples = no_samples

    def sample(self, pts):
        pass
