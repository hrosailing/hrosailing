import numpy as np
from scipy.interpolate import bisplrep, bisplev, griddata, \
    SmoothBivariateSpline, Rbf

from polar_diagram import PolarDiagramTable, PolarDiagramPointcloud
from _exceptions import ProcessingException
from _utils import convert_wind, polar_to_kartesian


# V: Soweit in Ordnung
def create_polar_diagram(data, p_type=PolarDiagramTable,
                         w_func=None, f_func=None, i_func=None,
                         w_res=None, tw=True,
                         w_func_kw=None, **filter_kw):
    if p_type not in \
            (PolarDiagramTable, PolarDiagramPointcloud):
        raise ProcessingException(
            "functionality for p_type not yet implemented")

    if w_func_kw is None:
        w_func_kw = {}

    w_points = WeightedPoints(
        data, w_func=w_func, tw=tw, **w_func_kw)

    w_points = filter_points(w_points, f_func, **filter_kw)
    if p_type is PolarDiagramPointcloud:
        return PolarDiagramPointcloud(
            points=w_points.points, tw=True)

    data, w_res = interpolate_points(w_points, w_res, i_func)
    ws_res, wa_res = w_res
    data = data.reshape(len(wa_res), len(ws_res))
    return PolarDiagramTable(
        wind_speed_resolution=ws_res,
        wind_angle_resolution=wa_res,
        data=data, tw=True)


# V: Soweit in Ordnung
def interpolate_points(w_points, w_res=None, i_func=None):
    if w_res is None:
        w_res = (np.arange(2, 42, 2), np.arange(0, 360, 5))

    elif w_res == "auto":
        ws_min = round(w_points.points[:, 0].min())
        ws_max = round(w_points.points[:, 0].max())
        wa_min = round(w_points.points[:, 1].min())
        wa_max = round(w_points.points[:, 1].max())
        ws_res = np.around(np.arange(ws_min, ws_max, (ws_max - ws_min)/20))
        wa_res = np.around(np.arange(wa_min, wa_max, (wa_max - wa_min)/72))
        w_res = (ws_res, wa_res)

    if i_func is None:
        return spline_interpolation(w_points, w_res), w_res

    return i_func(w_points, w_res), w_res


# V: Soweit in Ordnung
def filter_points(w_points, f_func=None, f_mode='bound',
                  **filter_kw):
    if f_func is not None:
        return f_func(w_points, **filter_kw)

    if f_mode not in ('bound', 'percentage'):
        raise ProcessingException(
            "functionality for std_mode not yet implemented")

    if f_mode == 'bound':
        f_arr = bound_filter(
            w_points.weights,
            filter_kw.get("u_b", 1),
            filter_kw.get("l_b", 0.5),
            filter_kw.get("strict", True))
    else:
        f_arr = percentile_filter(
            w_points.weights, filter_kw.get("percent", 50))

    return WeightedPoints(
        w_points.points[f_arr],
        weights=w_points.weights[f_arr])


# V: In Arbeit
class WeightedPoints:

    def __init__(self, points, weights=None, w_func=None, tw=True,
                 **w_func_kw):
        points = np.array(points)
        if len(points[0]) != 3:
            try:
                points = points.reshape(-1, 3)
            except ValueError:
                raise ProcessingException(
                    "points could not be broadcasted "
                    "to an array of shape (n,3)")

        w_dict = convert_wind(
            {"wind_speed": points[:, 0],
             "wind_angle": points[:, 1],
             "boat_speed": points[:, 2]},
            tw)
        points = np.column_stack(
            (w_dict["wind_speed"],
             w_dict["wind_angle"],
             points[:, 2]))
        self._points = points

        if w_func is None:
            w_func = yet_another_w_func

        if weights is None:
            self._weights = w_func(points, **w_func_kw)
        else:
            self._weights = np.array(weights)

    @property
    def points(self):
        return self._points.copy()

    @property
    def weights(self):
        return self._weights.copy()


# V: In Arbeit
def default_w_func(points, **w_func_kw):
    st_points = w_func_kw.get('st_points', 13)
    out = w_func_kw.get('out', 5)

    std_list = [[], [], []]
    weights = []

    for i in range(st_points, len(points)):
        std_list[0].append(points[i-st_points:i, 0].std())
        std_list[1].append(points[i-st_points:i, 1].std())
        std_list[2].append(points[i-st_points:i, 2].std())

    for i in range(3):
        f_arr_1 = percentile_filter(std_list[i], out)
        f_arr_2 = percentile_filter(std_list[i], 100 - out)
        f_arr = (f_arr_1 == f_arr_2)
        weights.append([1 / std_list[i][j]**2 if f_arr[j] else 0
                        for j in range(len(std_list[i]))])

    sum_weights = np.array([
        (ws_w + wa_w + bsp_w)/3 for ws_w, wa_w, bsp_w
        in zip(weights[0], weights[1], weights[2])])
    normed_weights = sum_weights / max(sum_weights)
    return np.concatenate([np.array([1] * st_points), normed_weights])


# V: In Arbeit
def better_w_func(points, **w_func_kw):
    radius = w_func_kw.get('radius', 1)
    ws_weight = w_func_kw.get('ws_weight', 1)
    weights = [0] * len(points)

    for i, point in enumerate(points):
        mask_WS = np.abs(points[:, 0] - point[0]) <= ws_weight
        # Hier nicht Degree sondern Radians?
        # Kartesische Koordinaten?
        mask_R = np.linalg.norm(
            polar_to_kartesian(points[:, 1:] - point[1:]),
            axis=1) <= radius
        weights[i] = len(points[np.logical_and(mask_R, mask_WS)]) - 1

    weights = np.array(weights)
    # Andere Normierungen?
    # weights = weights / max(weights)
    weights = len(points) * weights / sum(weights)
    return weights


def yet_another_w_func(points, **w_func_kw):
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
