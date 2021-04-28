import numpy as np
from polar_diagram import PolarDiagramTable, PolarDiagramPointcloud
from _exceptions import ProcessingException
from _utils import convert_wind


def create_interpolated_table(points, interpol_func, ws_res=None,
                              wa_res=None):
    pass


def filter_points(w_points, f_mode='bound', **filter_kw):
    if f_mode == 'bound':
        strict = filter_kw.get("strict", False)
        lower = filter_kw.get("l_b", np.NINF)
        upper = filter_kw.get("u_b", np.inf)
        weights = w_points.weights
        if strict:
            f_arr_l = weights > lower
            f_arr_u = weights < upper
        else:
            f_arr_l = weights >= lower
            f_arr_u = weights <= upper

        f_arr = (f_arr_l == f_arr_u)

    elif f_mode == 'percentage':
        points = w_points.points
        per = 1 - filter_kw.get("percent", 50)/100
        num = len(points) * per
        if int(num) == num:
            bound = (points[num]+points[num+1]) / 2
        else:
            bound = points[np.ceil(num)]

        f_arr = w_points.weights >= bound
    else:
        raise ProcessingException

    return w_points.points[f_arr]


class WeightedPoints:

    def __init__(self, points, weights=None, tws=True, twa=True):

        if len(points) != 3:
            raise ProcessingException

        points = np.array(points)
        w_dict = convert_wind(
            {"wind_speed": points[:, 0], "wind_angle": points[:, 1]},
            tws, twa)
        points = np.column_stack(
            (w_dict["wind_speed"], w_dict["wind_angle"], points[:, 2]))

        self._points = points
        self._weights = weights

    def points(self):
        return self._points.copy()

    def weights(self):
        return self._weights.copy()

    def weighing(self, w_func):
        weights = w_func(points)
        self._weights = weights
        return
