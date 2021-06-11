"""
A Pipeline class to automate getting a
polar diagram from "raw" dataa
"""

# Author: Valentin F. Dannenberg / Ente


import logging.handlers

from data_analysis.filter import *
from data_analysis.interpolater import *
from data_analysis.neighbourhood import *
from data_analysis.weigher import *
from filereading import read_csv_file, read_nmea_file
from polardiagram import PolarDiagram, PolarDiagramTable, \
    PolarDiagramPointcloud, PolarDiagramCurve
from utils import speed_resolution, angle_resolution


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='logging/processing.log')

LOG_FILE = 'logging/processing.log'

logger = logging.getLogger(__name__)
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when='midnight')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


class PolarPipeline:
    """

    """
    def __init__(self, weigher=None, filter_=None,
                 eval_func=None, i_func=None,
                 s_fit_func=None):

        if filter_ is None:
            filter_ = ...
        if not isinstance(filter_, Filter):
            raise ProcessingException("")

        self._weigher = weigher
        self._filter = filter_
        self._eval_func = eval_func
        self._i_func = i_func
        self._s_fit_func = s_fit_func

    @property
    def weigher(self):
        return self._weigher

    @property
    def filter(self):
        return self._filter

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
        return f"""PolarPipeline(w_func={self.weigher.__name__},
        f_func={self.filter.__name__},
        eval_func={self.evaluation_function.__name__},
        i_func={self.interpolating_function.__name__},
        s_fit_func={self.surface_fitting_function.__name__})"""

    def __call__(self, p_type: PolarDiagram,
                 data=None, data_file=None,
                 file_format=None, tw=True,
                 w_res=None, neighbourhood=None,
                 norm=None, filtering=True,
                 **kwargs):

        if data is None and data_file is None:
            raise ProcessingException("")
        if data is None:
            data, tw = _read_file(data_file, file_format,
                                  tw, kwargs.get('mode', 'mean'))

        w_pts = WeightedPoints(data, weigher=self.weigher,
                               tw=tw)
        if filtering:
            mask = self.filter.filter(w_pts.weights)
            w_pts = WeightedPoints(w_pts.points[mask],
                                   weights=w_pts.weights[mask])

        if norm is None:
            norm = euclidean_norm

        if p_type is PolarDiagramTable:
            return _create_polar_diagram_table(
                w_pts, w_res, norm, neighbourhood,
                self.evaluation_function, **kwargs)

        if p_type is PolarDiagramCurve:
            return _create_polar_diagram_curve(
                w_pts, self.surface_fitting_function,
                **kwargs)

        if p_type is PolarDiagramPointcloud:
            return _create_polar_diagram_pointcloud(
                w_pts, norm, neighbourhood,
                self.interpolating_function,
                **kwargs)


def _read_file(data_file, file_format, tw, mode):
    if file_format is None:
        raise ProcessingException("")

    if file_format not in ('csv', 'nmea'):
        raise ProcessingException("")

    if file_format == 'csv':
        data = read_csv_file(data_file)
    if file_format == 'nmea':
        data = read_nmea_file(
            data_file, mode)
        if data[0][3] == 'R':
            tw = False
        data = [d[:3] for d in data]

    return data, tw


def _create_polar_diagram_table(w_pts, w_res, norm,
                                neighbourhood, eval_func,
                                **kwargs):
    if neighbourhood is None:
        neighbourhood = Ball()

    if not isinstance(neighbourhood, Neighbourhood):
        raise ProcessingException("")

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
    # Work with regression?
    if s_fit_function is None:
        # TODO
        pass

    return s_fit_function(w_pts, **kwargs)


def _create_polar_diagram_pointcloud(w_pts, norm, neighbourhood,
                                     i_func, **kwargs):
    if i_func is None:
        i_func = weighted_mean_interpolation

    if neighbourhood is None:
        neighbourhood = Ball()

    if not isinstance(neighbourhood, Neighbourhood):
        raise ProcessingException()

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
            mask = neighbourhood.is_contained_in(
                w_points.points[:, :2] - grid_point)
            if not any(mask):
                continue
            dist = norm(w_points.points[mask][:, :2] - grid_point)
            data[j, i] = eval_func(
                w_points.points[mask][:, 2],
                w_points.weights[mask],
                dist, **kwargs)

    return data
