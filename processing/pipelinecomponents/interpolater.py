"""
Defines a baseclass for interpolaters used in the
processing.processing.PolarPipeline class,
that can be used to create custom interpolaters for use.

Also contains various predefined and usable interpolaters
"""

# Author Valentin F. Dannenberg / Ente


from abc import ABC, abstractmethod
import numpy as np

from scipy.interpolate import bisplrep, bisplev, griddata, \
    SmoothBivariateSpline, Rbf, LSQBivariateSpline

from exceptions import ProcessingException
from utils import euclidean_norm


class Interpolater(ABC):
    """Base class for
    all interpolater
    classes

    Abstract Methods
    ----------------
    interpolate(self, w_pts)
    """

    @abstractmethod
    def interpolate(self, w_pts, grid_pt):
        pass


class ArithmeticMeanInterpolater(Interpolater):
    """An interpolater
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
        return (f"ArithmeticMeanInterpolater("
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


def spline_interpolation(w_pts, w_res):
    ws, wa, bsp = np.hsplit(w_pts.points, 3)
    ws_res, wa_res = w_res
    wa = np.deg2rad(wa)
    wa, bsp = bsp * np.cos(wa), bsp * np.sin(wa)
    spl = SmoothBivariateSpline(ws, wa, bsp, w=w_pts.weights)
    # spl = bisplrep(ws, wa, bsp, kx=1, ky=1)
    # return bisplev(ws_res, wa_res, spl).T
    # d_points, val = np.hsplit(w_points.points, [2])
    ws_res, wa_res = np.meshgrid(ws_res, wa_res)
    ws_res = ws_res.reshape(-1, )
    wa_res = wa_res.reshape(-1, )
    # rbfi = Rbf(ws, wa, bsp, smooth=1)
    # return rbfi(ws_res, wa_res)
    # return griddata(d_points, val, (ws_res, wa_res), 'linear',
    # rescale=True).T
    return spl.ev(ws_res, wa_res)


def gauss_potential(distances, weights, *params):
    alpha = params[0]
    beta = params[1]
    return beta * np.exp(-alpha * weights * distances)
