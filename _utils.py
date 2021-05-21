import numpy as np
from collections import Iterable
from scipy.interpolate import bisplrep, bisplev, griddata, \
    SmoothBivariateSpline, Rbf
from scipy.spatial import ConvexHull
from _exceptions import PolarDiagramException


# V: Soweit in Ordnung
def polar_to_kartesian(rad, ang):
    return np.column_stack((rad * np.cos(ang),
                            rad * np.sin(ang)))


# V: Soweit in Ordnung
def convex_hull_polar(points_rad, points_ang):
    converted_points = polar_to_kartesian(points_rad, points_ang)
    return ConvexHull(converted_points)


# V: In Arbeit
def speed_resolution(ws_res):
    if ws_res is None:
        return np.array(np.arange(2, 42, 2))

    if not isinstance(ws_res, (Iterable, int, float)):
        raise PolarDiagramException(
            "ws_res is neither Iterable, int or float")

    if isinstance(ws_res, Iterable):
        return np.array(list(ws_res))

    return np.array(np.arange(ws_res, 40, ws_res))


# V: In Arbeit
def angle_resolution(wa_res):
    if wa_res is None:
        return np.array(np.arange(0, 360, 5))

    if not isinstance(wa_res, (Iterable, int, float)):
        raise PolarDiagramException(
            "wa_res is neither Iterable, int or float")

    if isinstance(wa_res, Iterable):
        return np.array(list(wa_res))

    return np.array(np.arange(wa_res, 360, wa_res))


# V: In Arbeit
def get_indices(w_list, res_list):
    if w_list is None:
        return list(range(len(res_list)))

    if not isinstance(w_list, Iterable):
        try:
            ind = list(res_list).index(w_list)
            return [ind]
        except ValueError:
            raise PolarDiagramException(
                f"{w_list} is not in resolution")

    if not set(w_list).issubset(set(res_list)):
        raise PolarDiagramException(
            f"{w_list} is not in resolution")

    ind_list = [i for i in range(len(res_list)) if res_list[i] in w_list]
    return ind_list


def bound_filter(weights, upper, lower, strict):
    if strict:
        return (weights > lower) == (weights < upper)

    return (weights >= lower) == (weights <= upper)


def percentile_filter(weights, per):
    per = 1 - per / 100
    num = len(weights) * per
    if int(num) == num:
        return weights >= (weights[int(num)] + weights[int(num) + 1]) / 2

    return weights >= weights[int(np.ceil(num))]


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
    ws_res = ws_res.reshape(-1,)
    wa_res = wa_res.reshape(-1,)
    # rbfi = Rbf(ws, wa, bsp, smooth=1)
    # return rbfi(ws_res, wa_res)
    # return griddata(d_points, val, (ws_res, wa_res), 'linear',
    # rescale=True).T
    return spl.ev(ws_res, wa_res)
