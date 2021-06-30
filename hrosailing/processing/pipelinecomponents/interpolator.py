"""
Defines a baseclass for Interpolators used in the
processing.processing.PolarPipeline class,
that can be used to create custom Interpolators for use.

Also contains various predefined and usable Interpolators
"""

# Author Valentin F. Dannenberg / Ente


import numpy as np

from abc import ABC, abstractmethod

from hrosailing.utils import euclidean_norm


# TODO: Error checks

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
            wts = 1 / np.power(wts, self._p)
            wts /= np.sum(wts)

        # If interpolated point is
        # a measured point, return
        # the measured value
        if np.any(wts == 0):
            mask = wts == 0
            return w_pts.points[mask][0, 2]

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
        if not isinstance(s, (int, float)) or s <= 0:
            raise ValueError(
                f"The scaling parameter "
                f"needs to be a positive "
                f"number, but {s} was "
                f"passed")

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


# Should be used together with ScalingBall
class ImprovedIDWInterpolator(Interpolator):

    def __init__(self, norm=None):
        if norm is None:
            norm = euclidean_norm

        self._norm = norm

    def __repr__(self):
        pass

    def interpolate(self, w_pts, grid_pt):
        dist = self._norm(w_pts.points[:, :2] - grid_pt)

        # If interpolated point is
        # a measured point, return
        # the measured value
        if np.any(dist == 0):
            mask = dist == 0
            return dist[mask][0, 2]

        wts = _set_weights(w_pts.points, dist)
        wts = np.square(wts)
        wts /= np.sum(wts)

        return wts @ w_pts.points[:, 2]


# Should be used together with ScalingBall
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
        dist = self._norm(
            w_pts.points[:, :2] - grid_pt)

        # If interpolated point is
        # a measured point, return
        # the measured value
        if np.any(dist == 0):
            mask = dist == 0
            return dist[mask][0, 2]

        wts = _set_weights(w_pts, dist)
        wts = _include_direction(w_pts,
                                 grid_pt,
                                 dist,
                                 wts)
        wts /= np.sum(wts)

        mask = dist < self._tol
        n_eps = w_pts.points[mask]
        if n_eps.size:
            return np.sum(n_eps[:, 2]) / n_eps.shape[0]

        return wts @ (w_pts.points[:, 2])


def _set_weights(pts, dist):
    wts = np.zeros(pts.shape[0])
    r = np.max(dist)
    for i, d in enumerate(dist):
        if 0 < d <= r / 3:
            wts[i] = 1 / d
        elif r / 3 < d <= r:
            wts[i] = (27 / (4 * r)
                      * np.square(d / r - 1))

    return wts


def _include_direction(pts, grid_pt, dist, wts):
    t = np.zeros(pts.shape[0])
    for i, pt in enumerate(pts):
        t[i] = _sum_cosines(wts, pt, i,
                            pts, grid_pt,
                            dist)
        t[i] /= np.sum(wts)

    return np.square(wts) * (1 + t)


def _sum_cosines(wts, pt, i, pts,
                 grid_pt, distances):

    cosine = (grid_pt[0] - pt[0])\
              * (grid_pt[0] - np.delete(pts, i, axis=0)[:, 0])\
              + (grid_pt[1] - pt[1])\
              * (grid_pt[1] - np.delete(pts, i, axis=0)[:, 0])
    cosine /= (distances[i] * np.delete(distances, i))

    return np.sum(np.delete(wts, i) * (1 - cosine))
