"""
Defines a baseclass for Interpolators used in the
processing.processing.PolarPipeline class,
that can be used to create custom Interpolators for use.

Also contains various predefined and usable Interpolators
"""

# Author Valentin F. Dannenberg / Ente


import numpy as np

from abc import ABC, abstractmethod

from exceptions import ProcessingException
from utils import euclidean_norm


class Interpolator(ABC):
    """Base class for
    all Interpolator
    classes

    Abstract Methods
    ----------------
    interpolate(self, w_pts)
    """

    @abstractmethod
    def interpolate(self, w_pts, grid_pt):
        pass


class IDWInterpolator(Interpolator):

    def __init__(self, p=2, norm=None):
        if norm is None:
            norm = euclidean_norm
        self._p = p
        self._norm = norm

    def __repr__(self):
        return (f"IDWInterpolator("
                f"p={self._p}, "
                f"norm={self._norm.__name__})")

    def interpolate(self, w_pts, grid_pt):
        if self._p == 0:
            wts = np.ones(w_pts.points.shape[0])
            wts /= wts.shape[0]
        else:
            wts = self._norm(w_pts.points[:, :2] - grid_pt)
            if np.any(wts == 0):
                mask = wts == 0
                return w_pts.points[mask][0, 2]

            wts = 1 / np.power(wts, self._p)
            wts /= np.sum(wts)

            return wts @ w_pts.points[:, 2]


class ArithmeticMeanInterpolator(Interpolator):
    """An Interpolator
    that gets the
    interpolated value
    as follows:

    First the distance
    of the input variables
    of all considered points
    and of the to interpolate
    point is calculated, ie
    || p[:d-1] - inter[d-1] ||.
    Then using a distribution,
    new weights are calculated
    based on the old weights,
    the previously calculated
    distances and other parameters
    depending on the distribution

    The value of the dependend
    variable of the interpolated
    point then equals

    s * (Σ w_p * p) / Σ w_p

    where s is an additional
    scaling factor

    Parameters
    ----------
    s : positive int or float, optional
        Scaling factor for
        the arithmetic mean,

        Defaults to 1
    norm : function or callable, optional
        Norm with which to
        calculate the distances

        If nothing is passed,
        it will default to
        ||.||_2
    distribution : function or callable, optional
        Function with which to
        calculate the updated
        weights.

        Should have the signature
        f(distances, old_weights, *parameters) -> new_weights

        If nothing is passed,
        it will default to
        gauss_potential, which
        calculated weights based
        on the following formular:

        β * exp(-α * old_weights * distances)
    params:
        Parameters to be passed
        to distribution

    Methods
    -------
    interpolate(self, w_pts, grid_pt)
    """

    def __init__(self, *params, s=1, norm=None,
                 distribution=None):
        if not isinstance(s, (int, float)):
            raise ProcessingException("")

        if norm is None:
            norm = euclidean_norm

        if distribution is None:
            distribution = gauss_potential

        self._s = s
        self._norm = norm
        self._distr = distribution
        self._params = params

    def __repr__(self):
        return (f"ArithmeticMeanInterpolator("
                f"*params={self._params}, "
                f"s={self._s}, "
                f"norm={self._norm.__name__},"
                f"distribution={self._distr})")

    def interpolate(self, w_pts, grid_pt):
        distances = self._norm(w_pts.points[:, :2] - grid_pt)
        weights = self._distr(
            distances, w_pts.weights, *self._params)

        return self._s * np.average(
            w_pts.points, axis=0, weights=weights)


def gauss_potential(distances, weights, *params):
    alpha = params[0]
    beta = params[1]
    return beta * np.exp(-alpha * weights * distances)


class ImprovedIDWInterpolator(Interpolator):

    def __init__(self, norm=None):
        if norm is None:
            norm = euclidean_norm

        self._norm = norm

    def __repr__(self):
        pass

    def interpolate(self, w_pts, grid_pt):
        distances = self._norm(w_pts.points[:, :2] - grid_pt)
        r = np.max(distances)
        if np.any(distances == 0):
            mask = distances == 0
            return distances[mask][0, 2]

        wts = np.zeros(w_pts.points.shape[0])
        mask = distances <= r / 3
        wts[mask] = 1 / distances[mask]
        mask = (distances > r/3) & (distances <= r)
        wts[mask] = (27 / (4 * r)) * np.square(distances[mask] / r - 1)
        wts = np.square(wts)
        wts /= np.sum(wts)

        return wts @ w_pts.points[:, 2]


class ShepardInterpolator(Interpolator):

    def __init__(self, tol=np.finfo(float).eps,
                 slope_scal=0.1, norm=None):
        if norm is None:
            norm = euclidean_norm

        self._tol = tol
        self._slope = slope_scal
        self._norm = norm

    def __repr__(self):
        return (f"ShepardInterpolator("
                f"tol={self._tol}, "
                f"slope_scal={self._slope}, "
                f"norm={self._norm.__name__})")

    def interpolate(self, w_pts, grid_pt):
        distances = self._norm(
            w_pts.points[:, :2] - grid_pt)
        wts = _set_weights(w_pts,
                           grid_pt,
                           distances)

        wts /= np.sum(wts)

        mask = distances < self._tol
        n_eps = w_pts.points[mask]
        if not n_eps.size:
            return wts @ w_pts.points[:, 2]

        return np.sum(n_eps[:, 2]) / n_eps.shape[0]


def _set_weights(w_pts, grid_pt, distances):
    wts = np.zeros(w_pts.points.shape[0])
    r = np.max(distances)
    for i, d in enumerate(distances):
        if 0 < d <= r / 3:
            wts[i] = 1 / d
        elif r / 3 < d <= r:
            wts[i] = (27 / (4 * r)
                      * np.square(d / r - 1))

    t = np.zeros(w_pts.points.shape[0])
    for i, w_pt in enumerate(w_pts.points):
        t[i] = _sum_cosines(wts, w_pt, i,
                            w_pts.points,
                            grid_pt,
                            distances)
        t[i] /= np.sum(wts)

    wts = np.square(wts) * (1 + t)

    return wts


def _sum_cosines(wts, w_pt, i, w_pts,
                 grid_pt, distances):

    cosine = (grid_pt[0] - w_pt[0]) \
        * (grid_pt[0] - np.delete(w_pts, i, axis=0)[:, 0]) \
        + (grid_pt[1] - w_pt[1]) \
        * (grid_pt[1] - np.delete(w_pts, i, axis=0)[:, 0])
    cosine /= distances[i] * np.delete(distances, i)
    return np.sum(np.delete(wts, i) * (1 - cosine))
