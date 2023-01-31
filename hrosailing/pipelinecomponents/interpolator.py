"""
Classes used for modular modelling of different interpolation methods.

Defines the `Interpolator` abstract base class that can be used to
create custom interpolation methods.

Subclasses of `Interpolator` can be used with:

- the `TableExtension` and `PointcloudExtension` classes in the
`hrosailing.pipeline` module,
- the `__call__`-methods of the `PolarDiagramTable` and `PolarDiagramPointcloud`
classes in the `hrosailing.polardiagram` module.
"""


from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ._utils import scaled_euclidean_norm
from .neighbourhood import Neighbourhood


class InterpolatorInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of an `Interpolator`.
    """


class Interpolator(ABC):
    """Base class for all `Interpolator` classes.


    Abstract Methods
    ----------------
    interpolate(self, w_pts, grid_pt)
    """

    @abstractmethod
    def interpolate(self, w_pts, grid_pt):
        """This method should be used, given a point `grid_pt` and an
        instance of `WeightedPoints`, to determine the z-value at `grid_pt`,
        based on the z-values of the points in the `WeightedPoints` instance.

        Parameters
        ----------
        w_pts : WeightedPoints
            Considered measured points.

        grid_pt : numpy.ndarray of shape (2,)
            Point that is to be interpolated.

        Returns
        -------
        out : int / float
            Interpolated value at `grid_pt`.
        """


class IDWInterpolator(Interpolator):
    """Basic inverse distance interpolator, based on the work
    of Shepard, "A two-dimensional interpolation function
    for irregularly-spaced data".

    For a given point :math:`grid\\\\_pt`, that is to be interpolated, we
    calculate the distances :math:`d_{pt} = ||grid\\\\_pt - pt[:2]||` for all considered
    measured points. Then we set the weights of a point :math:`pt` to be
    :math:`w_{pt} = \\cfrac{1}{d_{pt}^p}`, for some non-negative integer :math:`p`.

    The interpolated value at :math:`grid\\\\_pt` then equals
    :math:`\\cfrac{\\sum_{pt} w_{pt} * pt[2]}{\\sum_{pt} w_{pt}}`
    or if :math:`grid\\\\_pt` is already a measured point :math:`pt`, it will equal :math:`pt[2]`.

    Parameters
    ----------
    p : non-negative int, optional

        Defaults to `2`.

    norm : function or callable, optional
        Norm with which to calculate the distances, i.e. :math:`||.||`.

        If nothing is passed, it will default to a scaled
        version of :math:`||.||_2`.

    Raises
    ------
    InterpolatorInitializationException
        If `p` is negative.
    """

    def __init__(self, p=2, norm: Callable = scaled_euclidean_norm):
        super().__init__()
        if p < 0:
            raise InterpolatorInitializationException("`p` is negative")

        self._p = p
        self._norm = norm

    def __repr__(self):
        return f"IDWInterpolator(p={self._p}, norm={self._norm.__name__})"

    def interpolate(self, w_pts, grid_pt):
        """Interpolates a given point `grid_pt` according to the procedure described
        above.

        Parameters
        ----------
        w_pts : WeightedPoints
            Considered measured points.

        grid_pt : numpy.ndarray of shape (2,)
            Point that is to be interpolated.

        Returns
        -------
        out : int / float
            Interpolated value at `grid_pt`.
        """
        pts = w_pts.data
        wts = self._norm(pts[:, :2] - grid_pt)
        if np.any(wts == 0):
            mask = wts == 0
            return pts[mask][0, 2]

        wts = 1 / np.power(wts, self._p)
        wts /= np.sum(wts)

        return wts @ pts[:, 2]


def _gauss_potential(distances, weights, *params):
    alpha = params[0]
    return np.exp(-alpha * weights * distances)


class ArithmeticMeanInterpolator(Interpolator):
    """An interpolator that computes the interpolated value according
    to the following procedure.

    First the distance of the independent variables (wind angle and wind speed)
    of all considered
    points and of the to interpolate point is calculated, i.e.
    :math:`|| p[:2] - grid\\\\_pt ||`.
    Then using a distribution, new weights are calculated based on
    the old weights, the previously calculated distances and other
    parameters depending on the distribution.

    The value of the dependent variable (boat speed) of the interpolated point
    then equals :math:`s * \\cfrac{\\sum_p w_p * p}{\\sum_p w_p}`
    where :math:`s` is an additional scaling factor.

    Note that this is a more general approach to the inverse distance
    interpolator.

    Parameters
    ----------
    s : positive int or float, optional
        Scaling factor for the arithmetic mean.

        Defaults to `1`.

    norm : function or callable, optional
        Norm with which to calculate the distances, i.e. :math:`||.||`.

        If nothing is passed, it will default to a scaled
        version of :math:`||.||_2`.

    distribution : function or callable, optional
        Function with which to calculate the updated weights.

        Should have the signature
        `f(distances, old_weights, *parameters) -> new_weights`.

        If nothing is passed, it will default to `gauss_potential`, which
        calculates weights based on the formula
        :math:`\\mathrm{e}^{\\textstyle -\\alpha * oldweights * distances}\\` with
        parameter :math:`\\alpha`.

    params : tuple
        Parameters to be passed to `distribution`.

        If no `distribution` is passed it defaults to `(100,)`, otherwise to `()`.

    Raises
    ------
    InterpolatorInitializationException
        If `s` is non-positive.
    """

    def __init__(
        self,
        s=1,
        norm: Callable = scaled_euclidean_norm,
        distribution: Callable = None,
        params=(),
    ):
        super().__init__()
        if s <= 0:
            raise InterpolatorInitializationException("`s` is non-positive")

        self._s = s
        self._norm = norm
        self._params = params
        if distribution is None:
            self._distr = _gauss_potential
            if params == ():
                self._params = (100,)
        else:
            self._distr = distribution

    def __repr__(self):
        return (
            f"ArithmeticMeanInterpolator(s={self._s},"
            f" norm={self._norm.__name__}, distribution={self._distr},"
            f" params={self._params})"
        )

    def interpolate(self, w_pts, grid_pt):
        """Interpolates a given `grid_pt` according to the procedure described
        above.

        Parameters
        ----------
        w_pts : WeightedPoints
            Considered measured points.

        grid_pt : numpy.ndarray of shape (2,)
            Point that is to be interpolated.

        Returns
        -------
        out : int / float
            Interpolated value at `grid_pt`.
        """
        pts = w_pts.data
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
    function for irregularly-spaced data".

    Instead of setting the weights as the normal inverse distance
    to some power, we set the weights in the following way:

    Let :math:`r` be the radius of the `neighbourhood.ScalingBall` with the center being some
    point :math:`grid\\\\_pt` which is to be interpolated.
    For all considered measured points let :math:`d_{pt}` be the same as
    in `IDWInterpolator`. If :math:`d_{pt} \\leq \\cfrac{r}{3}` we set :math:`w_{pt} = \\cfrac{1}{d_{pt}}`.
    Otherwise, we set :math:`w_{pt} = \\cfrac{27}{4 * r} * (\\cfrac{d_{pt}}{r - 1})^2`.

    The resulting value at :math:`grid\\\\_pt` will then be calculated the same
    way as in `IDWInterpolator`.

    Parameters
    ----------
    norm : function or callable, optional
        Norm with which to calculate the distances, i.e. :math:`||.||`.

        If nothing is passed, it will default to a scaled
        version of :math:`||.||_2`.
    """

    def __init__(self, norm: Callable = scaled_euclidean_norm):
        super().__init__()
        self._norm = norm

    def interpolate(self, w_pts, grid_pt):
        """Interpolates a given `grid_pt` according to the procedure described
        above.

        Parameters
        ----------
        w_pts : WeightedPoints
            Considered measured points.

        grid_pt : numpy.ndarray of shape (2,)
            Point that is to be interpolated.

        Returns
        -------
        out : int / float
            Interpolated value at `grid_pt`.
        """
        pts = w_pts.data
        dist = self._norm(pts[:, :2] - grid_pt)
        if np.any(dist == 0):
            mask = dist == 0
            return pts[mask][0, 2]

        wts = _set_weights(pts, dist)
        wts = np.square(wts)
        wts /= np.sum(wts)

        return wts @ pts[:, 2]


class ShepardInterpolator(Interpolator):
    """A full-featured inverse distance interpolator, based
    on the work of Shepard, "A two-dimensional interpolation
    function for irregularly-spaced data".


    Parameters
    ----------
    tol : positive float, optional
        The distance around every data point in which the data point is preferred to
        the interpolated data.

        Defaults to `numpy.finfo(float).eps`.

    slope : positive float, optional
        The initial slope used in Shepard`s algorithm.

        Defaults to `0.1`.

    norm : function or callable, optional
        Norm with which to calculate the distances, i.e. :math:`||.||`.

        If nothing is passed, it will default to a scaled
        version of :math:`||.||_2`.


    Raises
    ------
    InterpolatorInitializationException
         If `tol` is non-positive.
    InterpolatorInitializationException
         If `slope` is non-positive.

    """

    def __init__(
        self,
        neighbourhood: Neighbourhood,
        tol=None,
        slope=0.1,
        norm: Callable = scaled_euclidean_norm,
    ):
        if tol is None:
            tol = np.finfo(float).eps
        super().__init__()
        if tol <= 0:
            raise InterpolatorInitializationException("`tol` is non-positive")
        if slope <= 0:
            raise InterpolatorInitializationException(
                "`slope` is non-positive"
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
        """Interpolates a given `grid_pt` according to the procedure described
        above.

        Parameters
        ----------
        w_pts : WeightedPoints
            Considered measured points.

        grid_pt : numpy.ndarray of shape (2,)
            Point that is to be interpolated.

        Returns
        -------
        out : int / float
            Interpolated value at `grid_pt`.
        """
        pts = w_pts.data
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
    """"""
    wts = np.zeros(pts.shape[0])
    r = np.max(dist)

    for i, d in enumerate(dist):
        if 0 < d <= r / 3:
            wts[i] = 1 / d
        elif r / 3 < d <= r:
            wts[i] = 27 / (4 * r) * np.square(d / r - 1)

    return wts


def _include_direction(pts, grid_pt, dist, wts):
    """"""
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
    """"""
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
            * (pts_i[:, 2] - pt[2])
            * (pts_i[:, 0] - pt[0])
            / np.square(dist_i)
        )
        xderiv[i] /= np.sum(wts_i)
        yderiv[i] = np.sum(
            wts_i
            * (pts_i[:, 2] - pt[2])
            * (pts_i[:, 1] - pt[1])
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
