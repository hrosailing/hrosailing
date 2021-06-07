"""
Collection of default functions for the PolarPipeline class
"""

# Author Valentin F. Dannenberg / Ente


import numpy as np

from scipy.interpolate import bisplrep, bisplev, griddata, \
    SmoothBivariateSpline, Rbf, LSQBivariateSpline


def percentile_filter(weights, **kwargs):
    per = kwargs.get("percent", 75)
    return weights >= np.percentile(weights, per)


# def default_w_func(points, **w_func_kw):
#     st_points = w_func_kw.get('st_points', 13)
#     out = w_func_kw.get('out', 5)
#
#     std_list = [[], [], []]
#     weights = []
#
#     for i in range(st_points, len(points)):
#         std_list[0].append(points[i-st_points:i, 0].std())
#         std_list[1].append(points[i-st_points:i, 1].std())
#         std_list[2].append(points[i-st_points:i, 2].std())
#
#     for i in range(3):
#         f_arr_1 = percentile_filter(std_list[i], out)
#         f_arr_2 = percentile_filter(std_list[i], 100 - out)
#         f_arr = (f_arr_1 == f_arr_2)
#         weights.append([1 / std_list[i][j]**2 if f_arr[j] else 0
#                         for j in range(len(std_list[i]))])
#
#     sum_weights = np.array([
#         (ws_w + wa_w + bsp_w)/3 for ws_w, wa_w, bsp_w
#         in zip(weights[0], weights[1], weights[2])])
#     normed_weights = sum_weights / max(sum_weights)
#     return np.concatenate([np.array([1] * st_points), normed_weights])


# def default_w_func(points, **w_func_kw):
#     radius = w_func_kw.get('radius', 1)
#     ws_weight = w_func_kw.get('ws_weight', 1)
#     weights = [0] * len(points)
#
#     for i, point in enumerate(points):
#         mask_WS = np.abs(points[:, 0] - point[0]) <= ws_weight
#         # Hier nicht Degree sondern Radians?
#         # Kartesische Koordinaten?
#         mask_R = np.linalg.norm(
#             polar_to_kartesian(points[:, 1:] - point[1:]),
#             axis=1) <= radius
#         weights[i] = len(points[np.logical_and(mask_R, mask_WS)]) - 1
#
#     weights = np.array(weights)
#     # Andere Normierungen?
#     # weights = weights / max(weights)
#     weights = len(points) * weights / sum(weights)
#     return weights


def default_w_func(points, **w_func_kw):
    weights = [0] * len(points)
    ws_weight = w_func_kw.get('ws_weight', 1)
    wa_weight = w_func_kw.get('wa_weight', 1)

    for i, point in enumerate(points):
        mask_WS = np.abs(points[:, 0] - point[0]) <= ws_weight
        mask_WA = np.abs(points[:, 1] - point[1]) <= wa_weight
        cylinder = points[np.logical_and(mask_WS, mask_WA)][:, 2]
        std = np.std(cylinder) or 1
        mean = np.mean(cylinder) or 0
        weights[i] = np.abs(mean - point[2]) / std

    weights = np.array(weights)
    weights = weights / max(weights)
    # weights = len(points) * weights / sum(weights)
    return weights


def ball(vec, norm, **kwargs):
    radius = kwargs.get('radius', 5)
    distance = norm(vec, axis=1)
    return distance, distance <= radius


def weighted_arithm_mean(points, weights, dist, **kwargs):
    alpha = kwargs.get('alpha', 1)
    beta = kwargs.get('beta', 1)
    weights = gauss_potential(dist, weights, alpha, beta)
    scal_fac = kwargs.get('s', 1)
    return scal_fac * np.average(points, axis=0, weights=weights)


def gauss_potential(dist, weights, alpha, beta):
    return beta * np.exp(-alpha * weights * dist)


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
