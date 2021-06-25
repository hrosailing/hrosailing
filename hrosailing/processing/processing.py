"""
A Pipeline class to automate getting a
polar diagram from "raw" dataa
"""

# Author: Valentin F. Dannenberg / Ente


import logging.handlers
import numpy as np

import polardiagram as pol
import processing.modelfunctions as mf
import processing.pipelinecomponents as pc

from exceptions import ProcessingException
from filereading import (
    read_csv_file,
    read_nmea_file,
)
from utils import (
    speed_resolution,
    angle_resolution,
)


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='../hrosailing/logging/processing.log')

LOG_FILE = '../hrosailing/logging/processing.log'

logger = logging.getLogger(__name__)
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when='midnight')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


class PolarPipeline:
    """A Pipeline class to create
    polar diagrams from raw data,
    similiar to sklearns Pipeline.

    Parameters
    ----------
    weigher : Weigher, optional

    filter_ : Filter, optional

    sampler : Sampler, optional

    interpolater : Interpolator, optional

    regressor : Regressor, optional

    Methods
    -------
    weigher
        Returns a read only version
        of self._weigher
    filter
        Returns a read only version
        of self._filter
    sampler
        Returns a read only version
        of self._sampler
    interpolater
        Returns a read only version
        of self._interpolater
    regressor
        Returns a read only version
        of self._regressor
    __call__(p_type: PolarDiagram,
             data=None, data_file=None,
             file_format=None, file_mode='mean',
             tw=True, filtering=True, w_res=None,
             neighbourhood=None)

    """

    def __init__(self, weigher=None, filter_=None,
                 sampler=None, interpolater=None,
                 regressor=None):
        if weigher is None:
            weigher = pc.CylindricMeanWeigher()
        if filter_ is None:
            filter_ = pc.QuantileFilter()
        if sampler is None:
            sampler = pc.UniformRandomSampler(no_samples=500)
        if interpolater is None:
            interpolater = pc.ArithmeticMeanInterpolator(1, 1)
        if regressor is None:
            regressor = pc.ODRegressor(
                model_func=mf.tws_s_s_dt_twa_gauss_comb,
                init_values=(0.25, 10, 1.7, 0,
                             1.9, 30, 17.6, 0))

        if not isinstance(weigher, pc.Weigher):
            raise ProcessingException(
                f"{weigher.__name__} is "
                f"not a Weigher")
        if not isinstance(filter_, pc.Filter):
            raise ProcessingException(
                f"{filter_.__name__} is "
                f"not a Filter")
        if not isinstance(sampler, pc.Sampler):
            raise ProcessingException(
                f"{sampler.__name__} is "
                f"not a Sampler")
        if not isinstance(interpolater, pc.Interpolator):
            raise ProcessingException(
                f"{interpolater.__name__} is "
                f"not an Interpolator")
        if not isinstance(regressor, pc.Regressor):
            raise ProcessingException(
                f"{regressor.__name__} is "
                f"not a Regressor")

        self._weigher = weigher
        self._filter = filter_
        self._sampler = sampler
        self._interpolater = interpolater
        self._regressor = regressor

    @property
    def weigher(self):
        """Returns a read only version
        of self._weigher"""
        return self._weigher

    @property
    def filter(self):
        """Returns a read only version
        of self._filter"""
        return self._filter

    @property
    def sampler(self):
        """Returns a read only version
        of self._sampler"""
        return self._sampler

    @property
    def interpolater(self):
        """Returns a read only version
        of self._interpolater"""
        return self._interpolater

    @property
    def regressor(self):
        """Returns a read only version
        of self._regressor"""
        return self._regressor

    def __repr__(self):
        return (f"PolarPipeline("
                f"weigher={self.weigher.__name__}, "
                f"filter={self.filter.__name__}, "
                f"interpolater={self.interpolater.__name__}, "
                f"s_fit_func={self.regressor.__name__})")

    def __call__(self, p_type: pol.PolarDiagram,
                 data=None, data_file=None,
                 file_format=None, file_mode='mean',
                 tw=True, filtering=True, w_res=None,
                 neighbourhood=None):
        """

        Parameters
        ----------
        p_type : PolarDiagram
            Specifies the type of polar
            diagram, that is to be created

            Can either be PolarDiagramTable,
            PolarDiagramCurve or
            PolarDiagramPointcloud
        data : array_like, optional
            Data from which to create
            the polar diagram, given
            as a sequence of points
            consisting of wind speed,
            wind angle and boat speed
        data_file : string, optional
            file containing data from
            which to create a polar
            diagram. Can either be
            a .csv file containing a
            sequence of points consisting
            of wind speed, wind angle and
            boat speed,
            or a file containing
            nmea-sentences from which the
            data will be extracted.
        file_format : string, optional
            Specifies wether data_file is
            a .csv file or a file containing
            nmea sentences
        file_mode : string, optional
            Reading mode to be passed to
            filereading.read_nmea_file
            in the case that data_file is
            a file containing nmea_sentences

            Defaults to 'mean'
        tw : bool, optional
            Specifies if the given
            wind data should be
            viewed as true wind

            If False, wind data
            will be converted to
            true wind

            Defaults to True
        filtering : bool, optional
            Specifies if the data
            should be filtered using
            self.filter after weighing

            Defaults to True
        w_res : tuple of length 2 or string, optional
            Only used if p_type is
            PolarDiagramTable

            Specifies the wind speed
            and wind angle resolution
            for the PolarDiagramTable

            Can either be a tuple of
            length 2 containing the
            wind speed and wind angle
            resolution given in the
            same manner as in
            PolarDiagramTable,
            or the string 'auto',
            in which case the
            function will try to extract
            a good wind resolution based
            on the given data.

            If nothing is passed
            w_res will default to
            (numpy.arange(2, 42, 2),
             numpy.aragen(0, 360, 5))
        neighbourhood : Neighbourhood, optional
            Only used if p_type is
            PolarDiagramTable or
            PolarDiagramPointcloud

            Specifies the neighbourhood
            of the grid and sample points
            to be used for interpolation

            If nothing is passed,
            neighbourhood will default
            to neighbourhood.Ball()

        Returns
        -------
        polar_diagram : PolarDiagram
            An instance of the given
            p_type based on the input
            data
        """

        if data is None and data_file is None:
            raise ProcessingException("")
        if data is None:
            data, tw = _read_file(data_file,
                                  file_format,
                                  file_mode, tw)

        w_pts = pc.WeightedPoints(data,
                                  weigher=self.weigher,
                                  tw=tw)
        if filtering:
            mask = self.filter.filter(w_pts.weights)
            w_pts = w_pts[mask]

        if p_type is pol.PolarDiagramTable:
            return _create_polar_diagram_table(
                w_pts, w_res, neighbourhood,
                self.interpolater)

        if p_type is pol.PolarDiagramCurve:
            return _create_polar_diagram_curve(
                w_pts, self.regressor)

        if p_type is pol.PolarDiagramPointcloud:
            return _create_polar_diagram_pointcloud(
                w_pts, neighbourhood, self.interpolater,
                self.sampler)


def _read_file(data_file, file_format,
               mode, tw):

    if file_format is None:
        raise ProcessingException(
            "No file-format was specified")
    if file_format not in ('csv', 'nmea'):
        raise ProcessingException(
            f"No functionality for the"
            f"specified file-format"
            f"{file_format} implemented")

    if file_format == 'csv':
        data = read_csv_file(data_file)
    if file_format == 'nmea':
        data = read_nmea_file(
            data_file,
            mode=mode,
            convert_wind=True)
        tw = True

    return data, tw


def _create_polar_diagram_table(w_pts, w_res,
                                neighbourhood,
                                interpolater):
    if neighbourhood is None:
        neighbourhood = pc.Ball()
    if not isinstance(neighbourhood, pc.Neighbourhood):
        raise ProcessingException(
            f"{neighbourhood.__name__} is "
            f"not a Neighbourhood")

    w_res = _set_wind_resolution(w_res, w_pts.points)

    bsps = _interpolate_grid_points(w_res, w_pts,
                                    neighbourhood,
                                    interpolater)

    return pol.PolarDiagramTable(ws_res=w_res[0],
                                 wa_res=w_res[1],
                                 bsps=bsps)


def _create_polar_diagram_curve(w_pts, regressor):
    # regressor.set_weights(w_pts.weights)
    regressor.fit(w_pts.points)

    return pol.PolarDiagramCurve(
        regressor.model_func,
        *regressor.optimal_params)


def _create_polar_diagram_pointcloud(w_pts,
                                     neighbourhood,
                                     interpolater,
                                     sampler):
    if neighbourhood is None:
        neighbourhood = pc.Ball()
    if not isinstance(neighbourhood, pc.Neighbourhood):
        raise ProcessingException(
            f"{neighbourhood.__name__} is "
            f"not a Neighbourhood")

    sample_pts = sampler.sample(w_pts.points)
    pts = []

    for s_pt in sample_pts:
        mask = neighbourhood.is_contained_in(
            w_pts.points[:, :2] - s_pt)

        pts.append(interpolater.interpolate(w_pts[mask], s_pt))

    return pol.PolarDiagramPointcloud(pts=pts)


def _set_wind_resolution(w_res, pts):
    if w_res == 'auto':
        ws_res = _extract_wind(pts[:, 0], 2, 100)
        wa_res = _extract_wind(pts[:, 1], 5, 30)
        return ws_res, wa_res

    if w_res is None:
        w_res = (None, None)

    ws_res, wa_res = w_res
    return speed_resolution(ws_res), angle_resolution(wa_res)


def _extract_wind(pts, n, threshhold):
    w_max = round(pts.max())
    w_min = round(pts.min())
    w_start = (w_min // n + 1) * n
    w_end = (w_max // n) * n
    res = [w_max, w_min]
    for w in range(w_start, w_end + n, n):
        if w == w_start:
            mask = pts >= w_min & pts <= w
        elif w == w_end:
            mask = pts >= w & pts <= w_max
        else:
            mask = pts >= w - n & pts <= w

        if len(pts[mask]) >= threshhold:
            res.append(w)

    return res


def _interpolate_grid_points(w_res, w_pts,
                             neighbourhood,
                             interpolater):
    ws_res, wa_res = w_res
    bsps = np.zeros((len(wa_res), len(ws_res)))

    for i, ws in enumerate(ws_res):
        for j, wa in enumerate(wa_res):
            grid_point = np.array([ws, wa])
            mask = neighbourhood.is_contained_in(
                w_pts.points[:, :2] - grid_point)
            if not any(mask):
                continue
            bsps[j, i] = interpolater.interpolate(w_pts[mask],
                                                  grid_point)
    return bsps
