"""
Classes used for modular modeling of different interpolation methods

Defines the Interpolator Abstract Base Class that can be used to
create custom interpolation methods

Subclasses of Interpolator can be used with

- the TableExtension and PointcloudExtension class in the
hrosailing.pipeline module
- the __call__ method of the PolarDiagramTable and PolarDiagramPointcloud
class in the hrosailing.polardiagram module
"""

# Author Valentin Dannenberg


from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ._utils import scaled_euclidean_norm
from .neighbourhood import Neighbourhood


class InterpolatorInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of an Interpolator
    """


class Interpolator(ABC):
    """Base class for all Interpolator classes


    Abstract Methods
    ----------------
    interpolate(self, w_pts)
    """

    @abstractmethod
    def interpolate(self, w_pts, grid_pt):
        """This method should be used, given a point grid_pt and an
        instances of WeightedPoints, to determine the z-value at grid_pt,
        based on the z-values of the points in the WeightedPoints instance
        """


class IDWInterpolator(Interpolator):
    """Basic inverse distance interpolator, based on the work
    of Shepard, "A two-dimensional interpolation function
    for irregulary-spaced data"

    For a given point grid_pt, that is to be interpolated, we
    calculate the distances d_pt = ||grid-pt - pt[:2]|| for all considered
    measured points. Then we set the weights of a point pt to be
    w_pt = 1 / d_pt^p, for some nonnegative integer p

    The interpolated value on grid_pt then equals (Σ w_pt pt[2]) / Σ w_pt
    or if grid_pt is already a measured point pt*, it will equal pt*[2]

    Parameters
    ----------
    p : nonnegative int, optional
        Defaults to 2

    norm : function or callable, optional
        Norm with which to calculate the distances, ie ||.||

        If nothing is passed, it will default to a scaled
        version of ||.||_2


    Raises an InterpolatorInitializationException if p is negative
    """

    def __init__(self, p=2, norm: Callable = scaled_euclidean_norm):
        if p < 0:
            raise InterpolatorInitializationException("`p` is negative")

        self._p = p
        self._norm = norm

    def __repr__(self):
        return f"IDWInterpolator(p={self._p}, norm={self._norm.__name__})"

    def interpolate(self, w_pts, grid_pt):
        """Interpolates a given grid_pt according to the
        above described method

        Parameters
        ----------
        w_pts : WeightedPoints
            Considered measured points

        grid_pt : numpy.ndarray of shape (2,)
            Point that is to be interpolated

        Returns
        -------
        out : int / float
            Interpolated values at grid_pt
        """
        pts = w_pts.points
        wts = self._norm(pts[:, :2] - grid_pt)
        if np.any(wts == 0):
            mask = wts == 0
            return pts[mask][0, 2]

        wts = 1 / np.power(wts, self._p)
        wts /= np.sum(wts)

        return wts @ pts[:, 2]


def gauss_potential(distances, weights, *params):
    alpha = params[0]
    return np.exp(-alpha * weights * distances)


class ArithmeticMeanInterpolator(Interpolator):
    """An Interpolator that gets the interpolated value according
    to the following procedure

    First the distance of the independent variables of all considered
    points and of the to interpolate point is calculated, ie
    || p[:2] - grid_pt ||
    Then using a distribution, new weights are calculated based on
    the old weights, the previously calculated distances and other
    parameters depending on the distribution

    The value of the dependent variable of the interpolated point then equals
    s * (Σ w_p * p) / Σ w_p
    where s is an additional scaling factor

    In fact it is a more general approach to the inverse distance
    interpolator

    Parameters
    ----------
    s : positive int or float, optional
        Scaling factor for the arithmetic mean,

        Defaults to 1

    norm : function or callable, optional
        Norm with which to calculate the distances, ie ||.||

        If nothing is passed, it will default to a scaled
        version of ||.||_2

    distribution : function or callable, optional
        Function with which to calculate the updated weights.

        Should have the signature
        f(distances, old_weights, *parameters) -> new_weights

        If nothing is passed, it will default to gauss_potential, which
        calculated weights based on the formula
        β * exp(-α * old_weights * distances)

    params:
        Parameters to be passed to distribution


    Raises an InterpolatorInitializationException if s is nonpositive
    """

    def __init__(
        self,
        *params,
        s=1,
        norm: Callable = scaled_euclidean_norm,
        distribution: Callable = gauss_potential,
    ):
        if s <= 0:
            raise InterpolatorInitializationException("`s` is not positive")

        self._s = s
        self._norm = norm
        self._distr = distribution
        self._params = params

    def __repr__(self):
        return (
            f"ArithmeticMeanInterpolator(*params={self._params}, s={self._s}, "
            f"norm={self._norm.__name__}, distribution={self._distr})"
        )

    def interpolate(self, w_pts, grid_pt):
        """Interpolates a given grid_pt according to the
        above described method

        Parameters
        ----------
        w_pts : WeightedPoints
            Considered measured points

        grid_pt : numpy.ndarray of shape (2,)
            Point that is to be interpolated

        Returns
        -------
        out : int / float
            Interpolated values at grid_pt
        """
        pts = w_pts.points
        dist = self._norm(pts[:, :2] - grid_pt)
        if np.any(dist == 0):
            mask = dist == 0
            return pts[mask][0, 2]

        wts = w_pts.weights
        wts = self._distr(dist, wts, *self._params)
        return self._s * np.average(pts[:, 2], weights=wts)


class ImprovedIDWInterpolator(Interpolator):
    """An improved inverse distance interpolator, based
    on the work of Shepard, "A two-dimensional interpolation
    function for irregulary-spaced data"

    Instead of setting the weights as the normal inverse distance
    to some power, we set the weights in the following way:

    Let r be the radius of the ScalingBall with the center being some
    point grid_pt which is to be interpolated.
    For all considered measured points let d_pt be the same as
    in IDWInterpolator. If d_pt <= r/3 we set w_pt = 1 / d_pt.
    Otherwise we set w_pt = 27 / (4 * r) * (d_pt / r - 1)^2

    The resulting value on grid_pt will then be calculated the same
    way as in IDWInterpolator

    Parameters
    ----------
    norm : function or callable, optional
        Norm with which to calculate the distances, ie ||.||

        If nothing is passed, it will default to a scaled
        version of ||.||_2
    """

    def __init__(self, norm: Callable = scaled_euclidean_norm):
        self._norm = norm

    def interpolate(self, w_pts, grid_pt):
        """Interpolates a given grid_pt according to the
        above described method

        Parameters
        ----------
        w_pts : WeightedPoints
            Considered measured points

        grid_pt : numpy.ndarray of shape (2,)
            Point that is to be interpolated

        Returns
        -------
        out : int / float
            Interpolated values at grid_pt
        """
        pts = w_pts.points
        dist = self._norm(pts[:, :2] - grid_pt)
        if np.any(dist == 0):
            mask = dist == 0
            return pts[mask][0, 2]

        wts = _set_weights(pts, dist)
        wts = np.square(wts)
        wts /= np.sum(wts)

        return wts @ pts[:, 2]


# Shouldn't be used just yet
class ShepardInterpolator(Interpolator):
    """A full featured inverse distance interpolator, based
    on the work of Shepard, "A two-dimensional interpolation
    function for irregulary-spaced data"


    Parameters
    ----------
    tol : positive float , optional

        Defautls to numpy.finfo(float).eps

    slope: positive float, optional

        Defaults to 0.1

    norm : function or callable, optional
        Norm with which to calculate the distances, ie ||.||

        If nothing is passed, it will default to a scaled
        version of ||.||_2


    Raises an InterpolatorInitializationException

    - if tol is nonpositive
    - if slope is nonpositive
    """

    def __init__(
        self,
        neighbourhood: Neighbourhood,
        tol=np.finfo(float).eps,
        slope=0.1,
        norm: Callable = scaled_euclidean_norm,
    ):
        if tol <= 0:
            raise InterpolatorInitializationException("`tol` is not positive")
        if slope <= 0:
            raise InterpolatorInitializationException(
                "`slope` is not positive"
            )

        self._norm = norm
        self._tol = tol
        self._slope = slope
        self._neighbourhood = neighbourhood

    def __repr__(self):
        return (
            f"ShepardInterpolator( tol={self._tol}, "
            f"slope_scal={self._slope}, norm={self._norm.__name__})"
        )

    def interpolate(self, w_pts, grid_pt):
        """Interpolates a given grid_pt according to the
        above described method

        Parameters
        ----------
        w_pts : WeightedPoints
            Considered measured points

        grid_pt : numpy.ndarray of shape (2,)
            Point that is to be interpolated

        Returns
        -------
        out : int / float
            Interpolated values at grid_pt
        """
        pts = w_pts.points
        dist = self._norm(pts[:, :2] - grid_pt)
        if np.any(dist == 0):
            mask = dist == 0
            return pts[mask][0, 2]

        wts = _set_weights(pts, dist)
        wts = _include_direction(pts, grid_pt, dist, wts)
        zdelta = _determine_slope(
            pts,
            grid_pt,
            dist,
            wts,
            self._neighbourhood,
            self._norm,
            self._slope,
        )

        # reduce computational errors
        mask = dist < self._tol
        n_eps = pts[mask]
        if n_eps.size:
            return np.sum(n_eps[:, 2]) / n_eps.shape[0]

        wts /= np.sum(wts)
        return wts @ (pts[:, 2] + zdelta)


def _set_weights(pts, dist):
    wts = np.zeros(pts.shape[0])
    r = np.max(dist)

    for i, d in enumerate(dist):
        if 0 < d <= r / 3:
            wts[i] = 1 / d
        elif r / 3 < d <= r:
            wts[i] = 27 / (4 * r) * np.square(d / r - 1)

    return wts


def _include_direction(pts, grid_pt, dist, wts):
    t = np.zeros(pts.shape[0])

    for i, pt in enumerate(pts):
        cosine = (grid_pt[0] - pt[0]) * (
            grid_pt[0] - np.delete(pts, i, axis=0)[:, 0]
        ) + (grid_pt[1] - pt[1]) * (
            grid_pt[1] - np.delete(pts, i, axis=0)[:, 0]
        )
        cosine /= dist[i] * np.delete(dist, i)
        t[i] = np.sum(np.delete(wts, i) * (1 - cosine))
        t[i] /= np.sum(wts)

    return np.square(wts) * (1 + t)


def _determine_slope(pts, grid_pt, dist, wts, nhood, norm, slope):
    xderiv = np.zeros(pts.shape[0])
    yderiv = np.zeros(pts.shape[0])

    for i, pt in enumerate(pts):
        pts_i = np.delete(pts, i, axis=0)
        mask = nhood.is_contained_in(pts_i - pt)
        pts_i = pts_i[mask]
        dist_i = norm(pts_i - pt)
        wts_i = np.delete(wts, i)[mask]
        xderiv[i] = np.sum(
            wts_i
            * (pts_i[:, 2] - pt[:, 2])
            * (pts_i[:, 0] - pt[:, 0])
            / np.square(dist_i)
        )
        xderiv[i] /= np.sum(wts_i)
        yderiv[i] = np.sum(
            wts_i
            * (pts_i[:, 2] - pt[:, 2])
            * (pts_i[:, 1] - pt[:, 1])
            / np.square(dist_i)
        )
        yderiv[i] /= np.sum(wts_i)

    dim_of_dist = (
        slope
        * (np.max(pts[:, 2]) - np.min(pts[:, 2]))
        / np.sqrt(np.max(np.square(xderiv) + np.square(yderiv)))
    )

    return (
        xderiv * (grid_pt[0] - pts[:, 0]) + yderiv * (grid_pt[1] - pts[:, 1])
    ) * (dim_of_dist / (dim_of_dist + dist))
