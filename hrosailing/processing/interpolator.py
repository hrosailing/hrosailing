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

from hrosailing.core.computing import scaled_euclidean_norm

from hrosailing.core.exceptions import BilinearInterpolatorOutsideGridException
from hrosailing.core.exceptions import BilinearInterpolatorNoGridException

from .neighbourhood import Neighbourhood

class Interpolator(ABC):
    """Base class for all `Interpolator` classes."""

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

    Supports the `repr` method.

    Parameters
    ----------
    p : non-negative int, optional
        The parameter 'p' used in the formulas above.

        Defaults to `2`.

    norm : function or callable, optional
        Norm with which to calculate the distances, i.e. :math:`||.||`.

        Defaults to a scaled version of :math:`||.||_2`.

    See also
    ----------
    `Interpolator`
    """

    def __init__(self, p=2, norm: Callable = scaled_euclidean_norm):
        super().__init__()
        if p < 0:
            raise ValueError("`p` is negative")

        if not isinstance(p, int):
            raise TypeError(f"`p` has to be of type `int` but is {type(p)}")

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

        See also
        ----------
        `Interpolator.interpolate`
        """
        pts = w_pts.data

        if len(pts) == 0:
            raise ValueError("`pts` has to contain at least one point")

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

    Supports the `repr` method.

    Parameters
    ----------
    s : positive int or float, optional
        Scaling factor for the arithmetic mean.

        Defaults to `1`.

    norm : function or callable, optional
        Norm with which to calculate the distances, i.e. :math:`||.||`.

        Defaults to a scaled version of :math:`||.||_2`.

    distribution : function or callable, optional
        Function with which to calculate the updated weights.

        Needs to have the signature
        `f(distances, old_weights, *parameters) -> new_weights`.

        Defaults to `gauss_potential`, which calculates weights based on the formula
        :math:`\\mathrm{e}^{\\textstyle -\\alpha * oldweights * distances}\\` with
        parameter :math:`\\alpha`.

    params : tuple
        Parameters to be passed to `distribution`.

        If no `distribution` is passed it defaults to `(100,)`, otherwise to `()`.

    See also
    ----------
    `Interpolator`
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
            raise ValueError("`s` is non-positive")

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

        See also
        ----------
        `Interpolator.interpolate`
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

    Let :math:`r` be the maximal distance of the data points and a
    point :math:`grid\\\\_pt` which is to be interpolated.
    For all considered measured points let :math:`d_{pt}` be the same as
    in `IDWInterpolator`. If :math:`d_{pt} \\leq \\cfrac{r}{3}` we set :math:`w_{pt} = \\cfrac{1}{d_{pt}}`.
    Otherwise, we set :math:`w_{pt} = \\cfrac{27}{4 * r} * (\\cfrac{d_{pt}}{r} - 1)^2`.

    The resulting value at :math:`grid\\\\_pt` will then be calculated simliar as in
    `IDWInterpolator` using squared weights instead.

    Parameters
    ----------
    norm : function or callable, optional
        Norm with which to calculate the distances, i.e. :math:`||.||`.

        Defaults to a scaled version of :math:`||.||_2`.

    See also
    ----------
    `Interpolator`, `IDWInterpolator`
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

        See also
        ----------
        `Interpolator.interpolate`
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

    Supports the `repr` method.


    Parameters
    ----------
    neighbourhood: Neighbourhood
        Only points inside the given neighbourhood will be taken into account.

    tol : positive float, optional
        The distance around every data point in which the data point is preferred to
        the interpolated data.

        Defaults to `numpy.finfo(float).eps`.

    slope : positive float, optional
        The initial slope used in Shepard`s algorithm.

        Defaults to `0.1`.

    norm : function or callable, optional
        Norm with which to calculate the distances, i.e. :math:`||.||`.

        Defaults to a scaled version of :math:`||.||_2`.

    See also
    ----------
    `Interpolator`
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
            raise ValueError("`tol` is non-positive")
        if slope <= 0:
            raise ValueError("`slope` is non-positive")

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

        See also
        ----------
        `Interpolator.interpolate`
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


class BilinearGridInterpolator(Interpolator):
    """An interpolator that computes..."""

    def __repr__(self):
        return "BilinearGridInterpolator()"

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

        See also
        ----------
        `Interpolator.interpolate`
        """

        if w_pts.data.shape[1] == 3:

            ws_points = w_pts.data[:, 0]
            wa_points = w_pts.data[:, 1]

            ws_in = grid_pt[0]
            wa_in = grid_pt[1]

            lower_ws = ws_points[ws_points <= ws_in]
            upper_ws = ws_points[ws_points >= ws_in]
            lower_wa = wa_points[wa_points <= wa_in]
            upper_wa = wa_points[wa_points >= wa_in]

            closest_lower_ws = (
                lower_ws[np.argmax(lower_ws)] if len(lower_ws) > 0 else None
            )
            closest_upper_ws = (
                upper_ws[np.argmin(upper_ws)] if len(upper_ws) > 0 else None
            )
            closest_lower_wa = (
                lower_wa[np.argmax(lower_wa)] if len(lower_wa) > 0 else None
            )
            closest_upper_wa = (
                upper_wa[np.argmin(upper_wa)] if len(upper_wa) > 0 else None
            )

            # check if we are inside or on the edge of the grid
            if (
                closest_lower_ws == None
                or closest_upper_ws == None
                or closest_lower_wa == None
                or closest_upper_wa == None
            ):
                raise BilinearInterpolatorOutsideGridException()

            bs_lws_lwa = self._bs_in_grid(
                w_pts.data, closest_lower_ws, closest_lower_wa
            )
            bs_lws_uwa = self._bs_in_grid(
                w_pts.data, closest_lower_ws, closest_upper_wa
            )
            bs_uws_lwa = self._bs_in_grid(
                w_pts.data, closest_upper_ws, closest_lower_wa
            )
            bs_uws_uwa = self._bs_in_grid(
                w_pts.data, closest_upper_ws, closest_upper_wa
            )

            diff_ws = closest_upper_ws - closest_lower_ws
            diff_wa = closest_upper_wa - closest_lower_wa

            if diff_ws == 0 and diff_wa == 0:
                return bs_lws_lwa
            if diff_ws == 0:
                lower_wa_factor = (wa_in - closest_lower_wa) / diff_wa
                bs_lws = (
                    1.0 - lower_wa_factor
                ) * bs_lws_lwa + lower_wa_factor * bs_lws_uwa
                return bs_lws
            if diff_wa == 0:
                lower_ws_factor = (ws_in - closest_lower_ws) / diff_ws
                bs_lwa = (
                    1.0 - lower_ws_factor
                ) * bs_lws_lwa + lower_ws_factor * bs_uws_lwa
                return bs_lwa
            lower_ws_factor = (ws_in - closest_lower_ws) / diff_ws
            lower_wa_factor = (wa_in - closest_lower_wa) / diff_wa
            # interpolation along wa
            bs_lws = (
                1.0 - lower_wa_factor
            ) * bs_lws_lwa + lower_wa_factor * bs_lws_uwa
            bs_uws = (
                1.0 - lower_wa_factor
            ) * bs_uws_lwa + lower_wa_factor * bs_uws_uwa
            # interpolation along ws
            bs = (
                1.0 - lower_ws_factor
            ) * bs_lws + lower_ws_factor * bs_uws
            return bs

        raise BilinearInterpolatorNoGridException()

    def _bs_in_grid(self, w_pts_data, ws, wa):
        ws_points = w_pts_data[:, 0]
        wa_points = w_pts_data[:, 1]
        matching_row = w_pts_data[(ws_points == ws) & (wa_points == wa)]

        if len(matching_row) > 0:
            return matching_row[0, 2]

        raise BilinearInterpolatorNoGridException()
