import numpy as np
from polar_diagram import PolarDiagramTable, PolarDiagramPointcloud
from _exceptions import ProcessingException
from _utils import convert_wind, bound_filter, percentile_filter, \
    spline_interpolation


def create_polardiagram(data, p_type=PolarDiagramTable, tws=True, twa=True,
                        **kwargs):

    w_func = kwargs.pop("w_func", None)
    f_func = kwargs.pop("f_func", None)
    i_func = kwargs.pop("i_func", None)
    w_res = kwargs.pop("w_res", None)

    w_points = WeightedPoints(data, w_func=w_func, tws=tws, twa=twa)
    points = filter_points(w_points, f_func, **kwargs)
    if p_type == PolarDiagramTable:
        if w_res is None:
            w_res = np.column_stack(
                (np.arange(2, 42, 2), np.arange(0, 360, 5)))
        elif w_res is "dontknowyet":
            ws_min = int(round(points[:, 0].min()))
            ws_max = int(round(points[:, 0].max()))
            wa_min = int(round(points[:, 1].min()))
            wa_max = int(round(points[:, 1].max()))
            #Hier noch Intervalteilung
        data = interpolate_points(points, w_res, i_func)
        ws_res, wa_res = np.hsplit(w_res, 2)
        ws_res, wa_res = ws_res.reshape(-1,), wa_res.reshape(-1,)
        return PolarDiagramTable(
            wind_speed_resolution=ws_res,
            wind_angle_resolution=wa_res,
            data=data, tws=True, twa=True)
    elif p_type == PolarDiagramPointcloud:
        return PolarDiagramPointcloud(
            points=points, tws=True, twa=True)


def interpolate_points(points, w_res, i_func=None):
    if i_func is not None:
        return i_func(points, w_res)
    else:
        return spline_interpolation(points, w_res)


def filter_points(w_points, f_func=None, std_mode='bound', **filter_kw):
    if f_func is not None:
        return f_func(w_points, **filter_kw)
    else:
        if std_mode == 'bound':
            f_arr = bound_filter(
                w_points.weights,
                filter_kw.get("u_b", np.inf),
                filter_kw.get("l_b", np.NINF),
                filter_kw.get("strict", False))
        elif std_mode == 'percentage':
            f_arr = percentile_filter(
                w_points.weights, filter_kw.get("percent", 50))
        else:
            raise ProcessingException
        return w_points.points[f_arr]


class WeightedPoints:

    def __init__(self, points, w_func=None, tws=True, twa=True):

        if len(points) != 3:
            raise ProcessingException

        points = np.array(points)
        w_dict = convert_wind(
            {"wind_speed": points[:, 0], "wind_angle": points[:, 1]},
            tws, twa)
        points = np.column_stack(
            (w_dict["wind_speed"], w_dict["wind_angle"], points[:, 2]))

        self._points = points
        self._weights = w_func(points)

    def points(self):
        return self._points.copy()

    def weights(self):
        return self._weights.copy()