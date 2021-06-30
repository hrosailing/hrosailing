"""
Defines a baseclass for samplers used in the
processing.processing.PolarPipeline class,
that can be used to create custom sampler for use.

Also contains various predefines and usable samplers
"""

# Author: Valentin F. Dannenberg / Ente


import numpy as np

from abc import ABC, abstractmethod
from scipy.spatial import ConvexHull



class Sampler(ABC):
    """Base class for
    all sampler classes

    Abstract Methods
    ----------------
    sample(self, pts)
    """

    @abstractmethod
    def sample(self, pts):
        pass


# TODO: Other random samplers
class UniformRandomSampler(Sampler):
    """A sampler
    that produces a
    number of uniformly
    distributed samples,
    which all lie in the
    convex hull of certain
    given points

    Parameters
    ----------
    n_samples : positive int
        Amount of samples that will
        be produced by the sampler

    Methods
    -------
    sample(self, pts):
        Produces samples
        according to the
        above described
        procedure
    """

    def __init__(self, n_samples):

        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(
                f"The number of samples"
                f"needs to be a positive"
                f"number but {n_samples}"
                f"was passed")

        self._n_samples = n_samples

    def __repr__(self):
        return (f"UniformRandomSampler("
                f"no_samples={self._n_samples}")

    def sample(self, pts):
        """Produces samples
        according to the
        above described
        procedure

        Parameters
        ----------
        pts : array_like of shape (n, 3)
            Points in whose convex
            hull the produced
            samples will lie

        Returns
        -------
        samples : numpy.ndarray of shape (no_samples, 3)
            samples produced by
            the above described
            method
        """
        rng = np.random.default_rng()
        pts = np.asarray(pts)[:, :2]
        ineqs = ConvexHull(pts).equations
        samples = []
        ws_bound, wa_bound = _create_bounds(pts)
        while len(samples) < self._n_samples:
            ws = rng.uniform(ws_bound[0], ws_bound[1],
                             self._n_samples - len(samples))
            wa = rng.uniform(wa_bound[0], wa_bound[1],
                             self._n_samples - len(samples))
            wind = np.column_stack((ws, wa))
            mask = np.all(
                (ineqs[:, :2] @ wind.T).T <= -ineqs[:, 2],
                axis=1)
            samples.extend(wind[mask])

        return np.array(samples)


def _create_bounds(pts):
    ws_bound = (np.amin(pts[:, 0]), np.amax(pts[:, 0]))
    wa_bound = (np.amin(pts[:, 1]), np.amax(pts[:, 1]))
    return ws_bound, wa_bound


# TODO
class FibonacciSampler(Sampler):
    """

    """

    def __init__(self, no_samples):
        self._no_samples = no_samples

    def sample(self, pts):
        pass


# TODO
class ArchimedianSampler(Sampler):
    """

    """

    def __init__(self, no_samples):
        self._no_samples = no_samples

    def sample(self, pts):
        pass
