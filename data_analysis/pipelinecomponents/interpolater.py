"""
"""

# Author Valentin F. Dannenberg / Ente

from abc import ABC, abstractmethod
import numpy as np

from scipy.interpolate import bisplrep, bisplev, griddata, \
    SmoothBivariateSpline, Rbf, LSQBivariateSpline

from exceptions import ProcessingException
from utils import euclidean_norm


class Interpolater(ABC):

    @abstractmethod
    def interpolate(self, w_pts):
        pass


class ArithmeticMeanInterpolater(Interpolater):

    def __init__(self, *params, s=1, norm=None,
                 distribution=None,):
        if not isinstance(s, (int, float)):
            raise ProcessingException("")

        if norm is None:
            norm = euclidean_norm

        if distribution is None:
            distribution = gauss_potential

        self._scal = s
        self._norm = norm
        self._distr = distribution
        self._params = params

    @property
    def scaling_factor(self):
        return self._scal

    @property
    def norm(self):
        return self._norm

    @property
    def distribution(self):
        return self._distr

    @property
    def parameters(self):
        return self._params

    def interpolate(self, w_pts):
        distances = self.norm(w_pts.points)
        updated_weights = self.distribution(
            distances, w_pts.weights, *self.parameters)

        return self.scaling_factor * np.average(
            w_pts.points, axis=0, weights=updated_weights)


def weighted_arithm_mean(points, weights, dist, **kwargs):
    alpha = kwargs.get('alpha', 1)
    beta = kwargs.get('beta', 1)
    weights = gauss_potential(dist, weights, alpha, beta)
    scal_fac = kwargs.get('s', 1)
    return scal_fac * np.average(points, axis=0, weights=weights)


def gauss_potential(distances, weights, *params):
    alpha = params[0]
    beta = params[1]
    return beta * np.exp(-alpha * weights * distances)


def weighted_mean_interpolation(w_pts, norm, neighbourhood,
                                **kwargs):
    points = []
    for w_pt in w_pts.points:
        dist, mask = neighbourhood(w_pts.points - w_pt,
                                   norm, **kwargs)
        points.append(weighted_arithm_mean(
            w_pts.points[mask], w_pts.weights[mask],
            dist, **kwargs))

    return points


def spline_interpolation(w_points, w_res):
    ws, wa, bsp = np.hsplit(w_points.points, 3)
    ws_res, wa_res = w_res
    wa = np.deg2rad(wa)
    wa, bsp = bsp * np.cos(wa), bsp * np.sin(wa)
    spl = SmoothBivariateSpline(ws, wa, bsp, w=w_points.weights)
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
