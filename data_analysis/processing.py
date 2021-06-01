import numpy as np
from scipy.interpolate import bisplrep, bisplev, griddata, \
    SmoothBivariateSpline, Rbf

from polardiagram import PolarDiagram, PolarDiagramTable, \
    PolarDiagramPointcloud, PolarDiagramCurve
from exceptions import ProcessingException
from utils import convert_wind, polar_to_kartesian, \
    speed_resolution, angle_resolution


class WeightedPoints:
    def __init__(self, points, weights=None,
                 w_func=None, tw=True, **w_func_kw):
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
            w_func = default_w_func

        if weights is None:
            self._weights = w_func(points, **w_func_kw)
        else:
            weights = np.array(weights)
            no_pts = len(points)

            if len(weights) != no_pts:
                try:
                    weights = weights.reshape(no_pts, )
                except ValueError:
                    raise ProcessingException(
                        f"weights could not be broadcasted"
                        f"to an array of shape ({no_pts}, )")

            self._weights = weights

    @property
    def points(self):
        return self._points.copy()

    @property
    def weights(self):
        return self._weights.copy()

    def filter(self, f_func, **filter_kw):
        if f_func is None:
            f_func = percentile_filter

        mask = f_func(self.weights, **filter_kw)

        self._points = self._points[mask]
        self._weights = self._weights[mask]


class PolarPipeline:
    def __init__(self, w_func=None, f_func=None,
                 neighbourhood=None, eval_func=None,
                 i_func=None):

        self._w_func = w_func
        self._f_func = f_func
        self._neighbourhood = neighbourhood
        self._eval_func = eval_func
        self._i_func = i_func

    @property
    def weight_function(self):
        return self._w_func

    @property
    def filter_function(self):
        return self._f_func

    @property
    def evaluation_function(self):
        return self._eval_func

    @property
    def interpolating_function(self):
        return self._i_func

    @property
    def neighbourhood(self):
        return self._neighbourhood

    def __repr__(self):
        return f"""PolarPipeline(w_func={self.weight_function.__name__},
        f_func={self.filter_function.__name__},
        neighbourhood={self.neighbourhood.__name__},
        eval_func={self.evaluation_function.__name__},
        i_func={self.interpolating_function.__name__}"""

    def __call__(self, data, p_type: PolarDiagram,
                 w_res=None, filtering=True, tw=True,
                 weight_kw={}, filter_kw={}, **kwargs):

        w_pts = WeightedPoints(data, self.weight_function,
                               tw, **weight_kw)
        if filtering:
            w_pts.filter(self.filter_function, **filter_kw)

        if p_type is PolarDiagramTable:
            return _create_polar_diagram_table(
                w_pts, w_res, self.neighbourhood,
                self.evaluation_function, **kwargs
            )

        if p_type is PolarDiagramCurve:
            return _create_polar_diagram_curve(
                w_pts, self.interpolating_function, **kwargs
            )

        if p_type is PolarDiagramPointcloud:
            return _create_polar_diagram_pointcloud(
                w_pts, **kwargs
            )


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


def _create_polar_diagram_table(w_pts, w_res, neighbourhood,
                                eval_func, **kwargs):
    if neighbourhood is None:
        neighbourhood = ball

    if eval_func is None:
        eval_func = weighted_arithm_mean

    if w_res is None:
        w_res = _set_wind_resolution(w_pts.points,
                                     kwargs.get("auto", False))

    data = _interpolate_grid_points(w_res, w_pts, neighbourhood,
                                    eval_func, **kwargs)

    return PolarDiagramTable(wind_speed_resolution=w_res[0],
                             wind_angle_resolution=w_res[1],
                             data=data)


def _create_polar_diagram_curve():
    return


def _create_polar_diagram_pointcloud():
    return


def _set_wind_resolution(pts, auto=False):
    if auto:
        ws_min = round(pts[:, 0].min())
        ws_max = round(pts[:, 0].max())
        wa_min = round(pts[:, 1].min())
        wa_max = round(pts[:, 1].max())
        ws_res = np.around(np.arange(ws_min, ws_max,
                                     (ws_max - ws_min) / 20))
        wa_res = np.around(np.arange(wa_min, wa_max,
                                     (wa_max - wa_min) / 72))
        return ws_res, wa_res

    return speed_resolution(None), angle_resolution(None)


def _interpolate_grid_points(w_res, w_points,
                             neighbourhood, eval_func,
                             **kwargs):
    ws_res, wa_res = w_res
    data = np.zeros((len(wa_res), len(ws_res)))

    for ws, i in enumerate(ws_res):
        for wa, j in enumerate(wa_res):
            grid_point = np.array([ws, wa])
            dist, mask = neighbourhood(
                w_points.points[:, :2] - grid_point, **kwargs)
            data[j, i] = eval_func(
                w_points.points[mask], dist[mask],
                w_points.weights[mask], **kwargs)

    return data


def ball(vec, **kwargs):
    radius = kwargs.get('radius', 1)
    vec[:, 1] = np.deg2rad(vec[:, 1])
    distance = np.linalg.norm(vec, axis=1)
    return distance, distance <= radius


def weighted_arithm_mean(points, weights, dist, **kwargs):
    alpha = kwargs.get('alpha', 1)
    beta = kwargs.get('beta', 1)
    weights = gauss_potential(dist, weights, alpha, beta)
    scal_fac = kwargs.get('s', 1)
    return scal_fac * weights @ points[:, 2] / len(weights)


def gauss_potential(dist, weights, alpha, beta):
    return beta * np.exp(-alpha * weights * dist)


def bound_filter(weights, upper, lower, strict):
    if strict:
        return np.logical_and(weights > lower, weights < upper)

    return np.logical_and(weights >= lower, weights <= upper)


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
