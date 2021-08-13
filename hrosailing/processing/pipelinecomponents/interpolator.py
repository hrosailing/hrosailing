"""
Defines a baseclass for Interpolators used in the
processing.processing.PolarPipeline class,
that can be used to create custom Interpolators for use.

Also contains various predefined and usable Interpolators
"""

# Author Valentin F. Dannenberg / Ente


from abc import ABC, abstractmethod

import numpy as np


def scaled(norm, scal):
    scal = np.array(list(scal))

    def scaled_norm(vec):
        return norm(scal * vec)

    return scaled_norm


def euclidean_norm(vec):
    return np.linalg.norm(vec, axis=1)


class InterpolatorException(Exception):
    """Custom exception for errors that may appear whilst
    working with the Interpolator class and subclasses
    """

    pass


# TODO Improve docstrings


class Interpolator(ABC):
    """Base class for all Interpolator classes


    Abstract Methods
    ----------------
    interpolate(self, w_pts)
    """

    @abstractmethod
    def interpolate(self, w_pts, grid_pt):
        pass


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

    Raises an InterpolatorException if the inputs are not of the
    specified or functionally equivalent type


    Methods
    -------
    interpolate(w_pts, grid_pt)
        Interpolates a given grid_pt according to the
        above described method
    """

    def __init__(self, p=2, norm=None):
        if p < 0 or not isinstance(p, int):
            raise InterpolatorException(
                f"p needs to be a nonnegative integer, but {p} was passed"
            )
        if norm is None:
            norm = scaled(euclidean_norm, (1 / 40, 1 / 360))
        if not callable(norm):
            raise InterpolatorException(f"{norm.__name__} is not callable")

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

        grid_pt : array_like of shape (2,)
            Point that is to be interpolated

        Returns
        -------
        out : int / float
            Interpolated values at grid_pt


        Raises an InterpolatorException
            - if
            - if
        """
        pts = w_pts.points
        wts = self._norm(pts[:, :2] - grid_pt)
        if np.any(wts == 0):
            mask = wts == 0
            return pts[mask][0, 2]

        wts = 1 / np.power(wts, self._p)
        wts /= np.sum(wts)

        return wts @ pts[:, 2]


class ArithmeticMeanInterpolator(Interpolator):
    """An Interpolator that gets the interpolated value according
    to the following procedure

    First the distance of the independent variables of all considered
    points and of the to interpolate point is calculated, ie
    || p[:d-1] - inter[d-1] ||
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

    Raises an InterpolatorException if the inputs are not of the
    specified or functionally equivalent type


    Methods
    -------
    interpolate(self, w_pts, grid_pt)
        Interpolates a given grid_pt according to the
        above described method
    """

    def __init__(self, *params, s=1, norm=None, distribution=None):
        if not isinstance(s, (int, float)) or s <= 0:
            raise InterpolatorException(
                f"The scaling parameter needs to be a positive "
                f"number, but {s} was passed"
            )

        if norm is None:
            norm = scaled(euclidean_norm, (1 / 40, 1 / 360))
        if not callable(norm):
            raise InterpolatorException(f"{norm.__name__} is not callable")

        if distribution is None:
            distribution = gauss_potential
        if not callable(distribution):
            raise InterpolatorException(
                f"{distribution.__name__} is not callable"
            )

        self._s = s
        self._norm = norm
        self._distr = distribution
        self._params = params

    def __repr__(self):
        return (
            f"ArithmeticMeanInterpolator("
            f"*params={self._params}, "
            f"s={self._s}, "
            f"norm={self._norm.__name__},"
            f"distribution={self._distr})"
        )

    def interpolate(self, w_pts, grid_pt):
        """Interpolates a given grid_pt according to the
        above described method

        Parameters
        ----------
        w_pts : WeightedPoints
            Considered measured points

        grid_pt : array_like of shape (2,)
            Point that is to be interpolated

        Returns
        -------
        out : int / float
            Interpolated values at grid_pt


        Raises an InterpolatorException
            - if
            - if
        """
        pts = w_pts.points
        dist = self._norm(pts[:, :2] - grid_pt)
        if np.any(dist == 0):
            mask = dist == 0
            return pts[mask][0, 2]

        wts = w_pts.weights
        wts = self._distr(dist, wts, *self._params)
        return self._s * np.average(pts, axis=0, weights=wts)


def gauss_potential(distances, weights, *params):
    alpha = params[0]
    beta = params[1]
    return beta * np.exp(-alpha * weights * distances)


class ImprovedIDWInterpolator(Interpolator):
    """An improved inverse distance interpolator, based
    on the work of Shepard, "A two-dimensional interpolation
    function for irregulary-spaced data"

    Should (only) be used together with the ScalingBall neighbourhood

    Instead of setting the weights as the normal inverse distance
    to some power, we set the weights in the following way:

    Let r be the radius of the ScalingBall with the center being some
    point grid_pt which is to be interpolated.
    For all considered measured points let d_pt be the same as
    in IDWInterpolator. If d_pt <= r/3 we set w_pt = 1 / d_pt.
    Otherwise we set w_pt = 27 / (4 * r) * (d / r - 1)^2

    The resulting value on grid_pt will then be calculated the same
    way as in IDWInterpolator

    Parameters
    ----------
    norm : function or callable, optional
        Norm with which to calculate the distances, ie ||.||

        If nothing is passed, it will default to a scaled
        version of ||.||_2

    Raises an InterpolatorException if the input is not of the
    specified or functionally equivalent type


    Methods
    -------
    interpolate(self, w_pts, grid_pt)
        Interpolates a given grid_pt according to the
        above described method
    """

    def __init__(self, norm=None):
        if norm is None:
            norm = scaled(euclidean_norm, (1 / 40, 1 / 360))
        if not callable(norm):
            raise InterpolatorException(f"{norm.__name__} is not callable")

        self._norm = norm

    def __repr__(self):
        pass

    def interpolate(self, w_pts, grid_pt):
        """Interpolates a given grid_pt according to the
        above described method

        Parameters
        ----------
        w_pts : WeightedPoints
            Considered measured points

        grid_pt : array_like of shape (2,)
            Point that is to be interpolated

        Returns
        -------
        out : int / float
            Interpolated values at grid_pt

        Raises an InterpolatorException
            - if
            - if
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


# TODO Finish implementation
class ShepardInterpolator(Interpolator):
    """A full featured inverse distance interpolator, based
    on the work of Shepard, "A two-dimensional interpolation
    function for irregulary-spaced data"

    Should (only) be used together with the ScalingBall neighbourhood



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

    Raises an InterpolatorException
        - if
        - if


    Methods
    -------
    interpolate(self, w_pts, grid_pt)
        Interpolates a given grid_pt according to the
        above described method
    """

    def __init__(self, tol=np.finfo(float).eps, slope=0.1, norm=None):
        if norm is None:
            norm = scaled(euclidean_norm, (1 / 40, 1 / 360))
        if not callable(norm):
            raise InterpolatorException(f"{norm.__name__} is not callable")

        if tol <= 0:
            raise InterpolatorException(
                f"tolarance should be a positive number, but {tol} was passed"
            )
        if slope <= 0:
            raise InterpolatorException(
                f"slope should be a positive number, but {slope} was passed"
            )

        self._tol = tol
        self._slope = slope
        self._norm = norm

    def __repr__(self):
        return (
            f"ShepardInterpolator("
            f"tol={self._tol}, "
            f"slope_scal={self._slope}, "
            f"norm={self._norm.__name__})"
        )

    def interpolate(self, w_pts, grid_pt):
        """Interpolates a given grid_pt according to the
        above described method

        Parameters
        ----------
        w_pts : WeightedPoints
            Considered measured points

        grid_pt : array_like of shape (2,)
            Point that is to be interpolated

        Returns
        -------
        out : int / float
            Interpolated values at grid_pt


        Raises an InterpolatorException
            - if
            - if
        """
        pts = w_pts.points
        dist = self._norm(pts[:, :2] - grid_pt)
        if np.any(dist == 0):
            mask = dist == 0
            return pts[mask][0, 2]

        wts = _set_weights(pts, dist)
        wts = _include_direction(pts, grid_pt, dist, wts)
        wts /= np.sum(wts)

        mask = dist < self._tol
        n_eps = pts[mask]
        if n_eps.size:
            return np.sum(n_eps[:, 2]) / n_eps.shape[0]

        return wts @ (pts[:, 2])


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
        t[i] = _sum_cosines(wts, pt, i, pts, grid_pt, dist)
        t[i] /= np.sum(wts)

    return np.square(wts) * (1 + t)


def _sum_cosines(wts, pt, i, pts, grid_pt, distances):

    cosine = (grid_pt[0] - pt[0]) * (
        grid_pt[0] - np.delete(pts, i, axis=0)[:, 0]
    ) + (grid_pt[1] - pt[1]) * (grid_pt[1] - np.delete(pts, i, axis=0)[:, 0])
    cosine /= distances[i] * np.delete(distances, i)

    return np.sum(np.delete(wts, i) * (1 - cosine))
