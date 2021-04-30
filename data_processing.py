import numpy as np
from polar_diagram import PolarDiagramTable, PolarDiagramPointcloud
from _exceptions import ProcessingException
from _utils import convert_wind, bound_filter, percentile_filter, \
    spline_interpolation


# V: Soweit in Ordnung
def create_polar_diagram(data, p_type=PolarDiagramTable,
                         w_func=None, f_func=None, i_func=None,
                         w_res=None, tws=True, twa=True,
                         w_func_kw=None, **filter_kw):
    if p_type not in \
            (PolarDiagramTable, PolarDiagramPointcloud):
        raise ProcessingException(
            "functionality for p_type not yet implemented")

    if w_func_kw is None:
        w_func_kw = {}

    w_points = WeightedPoints(
        data, w_func=w_func, tws=tws, twa=twa, **w_func_kw)

    points = filter_points(w_points, f_func, **filter_kw)
    if p_type is PolarDiagramPointcloud:
        return PolarDiagramPointcloud(
            points=points, tws=True, twa=True)

    data = interpolate_points(points, w_res, i_func)
    ws_res, wa_res = w_res
    return PolarDiagramTable(
        wind_speed_resolution=ws_res,
        wind_angle_resolution=wa_res,
        data=data, tws=True, twa=True)


# V: Soweit in Ordnung
def interpolate_points(points, w_res=None, i_func=None):
    if w_res is None:
        w_res = (np.arange(2, 42, 2), np.arange(0, 360, 5))

    elif w_res == "auto":
        ws_min = int(round(points[:, 0].min()))
        ws_max = int(round(points[:, 0].max()))
        wa_min = int(round(points[:, 1].min()))
        wa_max = int(round(points[:, 1].max()))
        ws_res = np.around(np.arange(ws_min, ws_max, 20))
        wa_res = np.around(np.arange(wa_min, wa_max, 72))
        w_res = (ws_res, wa_res)

    if i_func is None:
        return spline_interpolation(points, w_res)

    return i_func(points, w_res)


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
            filter_kw.get("u_b", np.inf),
            filter_kw.get("l_b", np.NINF),
            filter_kw.get("strict", False))
    else:
        f_arr = percentile_filter(
            w_points.weights, filter_kw.get("percent", 50))
    return w_points.points[f_arr]


# V: In Arbeit
class WeightedPoints:

    def __init__(self, points, w_func=None, tws=True,
                 twa=True, **w_func_kw):
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
             "wind_angle": points[:, 1]},
            tws, twa)
        points = np.column_stack(
            (w_dict["wind_speed"],
             w_dict["wind_angle"],
             points[:, 2]))
        self._points = points

        if w_func is None:
            w_func = default_w_func

        self._weights = w_func(points, **w_func_kw)

    @property
    def points(self):
        return self._points.copy()

    @property
    def weights(self):
        return self._weights.copy()


# V: In Arbeit
def default_w_func(points, **w_func_kw):
    ws_std = []
    wa_std = []
    bsp_std = []

    st_points = w_func_kw.get('st_points', 13)
    outlier = w_func_kw.get('outlier', 5)

    for i in range(st_points, len(points)):
        ws_std.append(points[i-st_points:i, 0].std())
        wa_std.append(points[i-st_points:i, 1].std())
        bsp_std.append(points[i-st_points:i, 2].std())

    f_arr_ws_1 = percentile_filter(ws_std, outlier)
    f_arr_ws_2 = percentile_filter(ws_std, 100 - outlier)
    f_arr_ws = (f_arr_ws_1 == f_arr_ws_2)
    ws_weight = [1 / ws_std[i] ** 2 if f_arr_ws else 0
                 for i in range(len(ws_std))]

    f_arr_wa_1 = percentile_filter(wa_std, outlier)
    f_arr_wa_2 = percentile_filter(wa_std, 100 - outlier)
    f_arr_wa = (f_arr_wa_1 == f_arr_wa_2)
    wa_weight = [1 / wa_std[i] ** 2 if f_arr_wa else 0
                 for i in range(len(wa_std))]

    f_arr_bsp_1 = percentile_filter(bsp_std, outlier)
    f_arr_bsp_2 = percentile_filter(bsp_std, 100 - outlier)
    f_arr_bsp = (f_arr_bsp_1 == f_arr_bsp_2)
    bsp_weight = [1/bsp_std[i]**2 if f_arr_bsp else 0
                  for i in range(len(bsp_std))]

    sum_weights = np.array([
        (ws_w + wa_w + bsp_w)/3 for ws_w, wa_w, bsp_w
        in zip(ws_weight, wa_weight, bsp_weight)])

    normed_weights = sum_weights / sum(sum_weights)
    weights = np.array([1] * st_points)
    return np.concatenate([weights, normed_weights])
