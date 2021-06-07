"""
A Pipeline class to automate getting a polar diagram from "raw" dataa
"""

# Author: Valentin F. Dannenberg / Ente


from polardiagram import PolarDiagram, PolarDiagramTable, \
    PolarDiagramPointcloud, PolarDiagramCurve
from exceptions import ProcessingException
from utils import convert_wind, polar_to_kartesian, \
    speed_resolution, angle_resolution
from data_analysis.defaultfunctions import *


class WeightedPoints:
    def __init__(self, points, weights=None,
                 w_func=None, tw=True, **w_func_kw):
        points = np.asarray(points)

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

    def __repr__(self):
        return f"""WeightedPoints(points={self.points},
        weights={self.weights})"""

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.points):
            self.index += 1
            return self.points[self.index], \
                self.weights[self.index]

        raise StopIteration


class PolarPipeline:
    def __init__(self, w_func=None, f_func=None,
                 eval_func=None,
                 i_func=None, s_fit_func=None):

        self._w_func = w_func
        self._f_func = f_func
        self._eval_func = eval_func
        self._i_func = i_func
        self._s_fit_func = s_fit_func

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
    def surface_fitting_function(self):
        return self._s_fit_func

    def __repr__(self):
        return f"""PolarPipeline(w_func={self.weight_function.__name__},
        f_func={self.filter_function.__name__},
        eval_func={self.evaluation_function.__name__},
        i_func={self.interpolating_function.__name__},
        s_fit_func={self.surface_fitting_function.__name__})"""

    def __call__(self, data, p_type: PolarDiagram,
                 w_res=None, neighbourhood=None,
                 norm=None, filtering=True, tw=True,
                 weight_kw=None, filter_kw=None,
                 **kwargs):

        if weight_kw is None:
            weight_kw = {}

        w_pts = WeightedPoints(data, w_func=self.weight_function,
                               tw=tw, **weight_kw)
        if filtering:
            if filter_kw is None:
                filter_kw = {}

            w_pts.filter(self.filter_function, **filter_kw)

        if norm is None:
            norm = np.linalg.norm

        if p_type is PolarDiagramTable:
            return _create_polar_diagram_table(
                w_pts, w_res, norm, neighbourhood,
                self.evaluation_function, **kwargs
            )

        if p_type is PolarDiagramCurve:
            return _create_polar_diagram_curve(
                w_pts, self.surface_fitting_function, **kwargs
            )

        if p_type is PolarDiagramPointcloud:
            return _create_polar_diagram_pointcloud(
                w_pts, norm, neighbourhood,
                self.interpolating_function, **kwargs
            )


def _create_polar_diagram_table(w_pts, w_res, norm,
                                neighbourhood, eval_func,
                                **kwargs):
    if neighbourhood is None:
        neighbourhood = ball

    if eval_func is None:
        eval_func = weighted_arithm_mean

    w_res = _set_wind_resolution(w_res, w_pts.points,
                                 auto=kwargs.get("auto", False))

    data = _interpolate_grid_points(w_res, w_pts, norm,
                                    neighbourhood, eval_func,
                                    **kwargs)

    return PolarDiagramTable(ws_res=w_res[0],
                             wa_res=w_res[1],
                             data=data)


def _create_polar_diagram_curve(w_pts, s_fit_function, **kwargs):

    if s_fit_function is None:
        # TODO
        pass

    return s_fit_function(w_pts, **kwargs)


def _create_polar_diagram_pointcloud(w_pts, norm, neighbourhood,
                                     i_func, **kwargs):
    if i_func is None:
        i_func = weighted_mean_interpolation

    if neighbourhood is None:
        neighbourhood = ball

    points = i_func(w_pts, norm, neighbourhood,
                    **kwargs)

    return PolarDiagramPointcloud(points=points)


def _set_wind_resolution(w_res, pts, auto=False):
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

    if w_res is None:
        ws_res = None
        wa_res = None
    else:
        ws_res, wa_res = w_res

    return speed_resolution(ws_res), angle_resolution(wa_res)


def _interpolate_grid_points(w_res, w_points, norm,
                             neighbourhood, eval_func,
                             **kwargs):
    ws_res, wa_res = w_res
    data = np.zeros((len(wa_res), len(ws_res)))

    for i, ws in enumerate(ws_res):
        for j, wa in enumerate(wa_res):
            grid_point = np.array([ws, wa])
            dist, mask = neighbourhood(
                w_points.points[:, :2] - grid_point,
                norm, **kwargs)
            if not any(mask):
                continue
            data[j, i] = eval_func(
                w_points.points[mask][:, 2],
                w_points.weights[mask],
                dist[mask], **kwargs)

    return data
