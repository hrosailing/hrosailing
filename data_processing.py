import numpy as np
from polar_diagram import PolarDiagramTable, PolarDiagramPointcloud
from _exceptions import ProcessingException
from _utils import convert_wind, speed_resolution, angle_resolution, \
    bound_filter, percentile_filter, rbf_interpolation


def create_polardiagram(data, p_type=PolarDiagramTable, **kwargs):

    w_func = kwargs.pop("w_func", None)
    w_points = WeightedPoints(data, w_func=w_func)
    f_func = kwargs.pop("f_func", None)
    i_func = kwargs.pop("i_func", None)
    w_res = kwargs.pop("w_res", None)
    data = filter_points(w_points, f_func, **kwargs)
    data = interpolate_points(data, i_func, w_res)

    if p_type == PolarDiagramTable:
        pass
    elif p_type == PolarDiagramPointcloud:
        pass


def interpolate_points(points, i_func=None, w_res=None):
    if i_func is not None:
        if w_res is None:
            return i_func(points)
        else:
            return i_func(points, w_res)
    else:
        if w_res is None:

        else:
            return rbf_interpolation(points, w_res)






def filter_points(w_points, f_func=None, std_mode='bound', **filter_kw):
    if f_func is not None:
        return f_func(w_points)
    else:
        if std_mode == 'bound':
            f_arr = bound_filter(
                w_points.weights, filter_kw.get("u_b", np.inf),
                filter_kw.get("l_b", np.NINF), filter_kw.get("strict", False))
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