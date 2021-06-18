"""
A Pipeline class to automate getting a
polar diagram from "raw" dataa
"""

# Author: Valentin F. Dannenberg / Ente


import logging.handlers

from processing.modelfunctions.models3d import *
from processing.pipelinecomponents.filter import *
from processing.pipelinecomponents.regressor import *
from processing.pipelinecomponents.interpolater import *
from processing.pipelinecomponents.neighbourhood import *
from processing.pipelinecomponents.sampler import *
from processing.pipelinecomponents.weigher import *
from filereading import read_csv_file, read_nmea_file
from polardiagram.polardiagram import PolarDiagram, PolarDiagramTable, \
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
    """A Pipeline class to create
    polar diagrams from raw data,
    similiar to sklearns Pipeline.

    Parameters
    ----------
    weigher : Weigher, optional

    filter_ : Filter, optional

    sampler : Sampler, optional

    interpolater : Interpolater, optional

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
            weigher = CylindricMeanWeigher()
        if filter_ is None:
            filter_ = QuantileFilter()
        if sampler is None:
            sampler = UniformRandomSampler(no_samples=500)
        if interpolater is None:
            interpolater = ArithmeticMeanInterpolater(1, 1)
        if regressor is None:
            regressor = ODRegressor(model_func=odr_tws_s_dt_gauss_comb,
                                    init_values=(0.25, 10, 1.7, 0,
                                                 1.9, 30, 17.6, 0))

        if not isinstance(weigher, Weigher):
            raise ProcessingException("weigher is not a Weigher")
        if not isinstance(filter_, Filter):
            raise ProcessingException("filter_ is not a Filter")
        if not isinstance(sampler, Sampler):
            raise ProcessingException("sampler is not a Sampler")
        if not isinstance(interpolater, Interpolater):
            raise ProcessingException("interpolater is not an"
                                      "Interpolater")
        if not isinstance(regressor, Regressor):
            raise ProcessingException("regressor is not a"
                                      "Regressor")

        self._weigher = weigher
        self._filter = filter_
        self._sampler = sampler
        self._interpolater = interpolater
        self._regressor = regressor

    @property
    def weigher(self):
        """Returns a read only version of self._weigher"""
        return self._weigher

    @property
    def filter(self):
        """Returns a read only version of self._filter"""
        return self._filter

    @property
    def sampler(self):
        """Returns a read only version of self._sampler"""
        return self._sampler

    @property
    def interpolater(self):
        """Returns a read only version of self._interpolater"""
        return self._interpolater

    @property
    def regressor(self):
        """Returns a read only version of self._regressor"""
        return self._regressor

    def __repr__(self):
        return f"""PolarPipeline(weigher={self.weigher.__name__},
        filter={self.filter.__name__}, 
        interpolater={self.interpolater.__name__},
        s_fit_func={self.regressor.__name__})"""

    def __call__(self, p_type: PolarDiagram,
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
            data, tw = _read_file(data_file, file_format,
                                  tw, file_mode)

        w_pts = WeightedPoints(data, weigher=self.weigher,
                               tw=tw)
        if filtering:
            mask = self.filter.filter(w_pts.weights)
            w_pts = w_pts[mask]

        if p_type is PolarDiagramTable:
            return _create_polar_diagram_table(
                w_pts, w_res, neighbourhood,
                self.interpolater)

        if p_type is PolarDiagramCurve:
            return _create_polar_diagram_curve(
                w_pts, self.regressor)

        if p_type is PolarDiagramPointcloud:
            return _create_polar_diagram_pointcloud(
                w_pts, neighbourhood, self.interpolater,
                self.sampler)


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


def _create_polar_diagram_table(w_pts, w_res, neighbourhood,
                                interpolater):
    if neighbourhood is None:
        neighbourhood = Ball()
    if not isinstance(neighbourhood, Neighbourhood):
        raise ProcessingException("")

    w_res = _set_wind_resolution(w_res, w_pts.points)

    bsps = _interpolate_grid_points(w_res, w_pts, neighbourhood,
                                    interpolater)

    return PolarDiagramTable(ws_res=w_res[0],
                             wa_res=w_res[1],
                             bsps=bsps)


def _create_polar_diagram_curve(w_pts, regressor):
    # regressor.set_weights(w_pts.weights)
    regressor.fit(w_pts.points)

    return PolarDiagramCurve(regressor.model_func,
                             *regressor.optimal_params)


def _create_polar_diagram_pointcloud(w_pts, neighbourhood,
                                     interpolater, sampler):
    if neighbourhood is None:
        neighbourhood = Ball()
    if not isinstance(neighbourhood, Neighbourhood):
        raise ProcessingException("")

    sample_pts = sampler.sample(w_pts.points)
    pts = []

    for s_pt in sample_pts:
        mask = neighbourhood.is_contained_in(
            w_pts.points[:, :2] - s_pt)

        pts.append(interpolater.interpolate(w_pts[mask], s_pt))

    return PolarDiagramPointcloud(pts=pts)


def _set_wind_resolution(w_res, pts):
    # TODO: Better extraction of a good
    #       resolution based on the data
    if w_res == 'auto':
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
        w_res = (None, None)
    ws_res, wa_res = w_res
    return speed_resolution(ws_res), angle_resolution(wa_res)


def _interpolate_grid_points(w_res, w_pts, neighbourhood,
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
