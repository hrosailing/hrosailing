"""
A Pipeline class to automate getting a
polar diagram from "raw" data
"""

# Author: Valentin F. Dannenberg / Ente

import csv
import logging.handlers
import numpy as np
import pynmea2

from abc import ABC, abstractmethod

import hrosailing.polardiagram as pol
import hrosailing.processing.modelfunctions as mf
import hrosailing.processing.pipelinecomponents as pc

from hrosailing.polardiagram import (
    FileReadingException,
    PolarDiagramException,
)
from hrosailing.wind import apparent_wind_to_true, set_resolution

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    filename="hrosailing/logging/processing.log",
)

LOG_FILE = "hrosailing/logging/processing.log"

logger = logging.getLogger(__name__)
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when="midnight"
)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


class PipelineException(Exception):
    pass


class PipelineExtension(ABC):
    @abstractmethod
    def process(self, w_pts: pc.WeightedPoints):
        pass


class PolarPipeline:
    """A Pipeline class to create polar diagrams from raw data
    Parameters
    ----------
    extension: PipelineExtension
    weigher : Weigher, optional
    filter_ : Filter, optional
    Methods
    -------
    weigher
        Returns a read only version of self._weigher
    filter
        Returns a read only version of self._filter
    sampler
        Returns a read only version of self._sampler
    interpolater
        Returns a read only version of self._interpolater
    regressor
        Returns a read only version of self._regressor
    __call__(p_type: PolarDiagram,
             data=None, data_file=None,
             file_format=None, file_mode='mean',
             tw=True, filtering=True, w_res=None,
             neighbourhood=None)
    """

    def __init__(
        self,
        extension: PipelineExtension,
        weigher: pc.Weigher = None,
        filter_: pc.Filter = None,
    ):
        if weigher is None:
            weigher = pc.CylindricMeanWeigher()
        if not isinstance(weigher, pc.Weigher):
            raise PipelineException(f"{weigher.__name__} is not a Weigher")
        self._weigher = weigher

        if filter_ is None:
            filter_ = pc.QuantileFilter()
        if not isinstance(filter_, pc.Filter):
            raise PipelineException(f"{filter_.__name__} is not a Filter")
        self._filter = filter_

        if not isinstance(extension, PipelineExtension):
            raise PipelineException(
                f"{extension.__name__} is not a PipelineExtension"
            )
        self._extension = extension

    @property
    def weigher(self):
        """Returns a read only version of self._weigher"""
        return self._weigher

    @property
    def filter(self):
        """Returns a read only version of self._filter"""
        return self._filter

    @property
    def extension(self):
        return self._extension

    def __repr__(self):
        pass

    def __call__(
        self,
        data=None,
        data_file=None,
        file_format=None,
        file_mode="mean",
        tw=True,
        filtering=True,
    ):
        """
        Parameters
        ----------
        data : array_like, optional
            Data from which to create the polar diagram, given
            as a sequence of points, consisting of wind speed,
            wind angle and boat speed
        data_file : string, optional
            file containing data from which to create a polar
            diagram. Can either be
                - a .csv file containing a sequence of points consisting
                of wind speed, wind angle and boat speed
                - a file containing nmea-sentences from which the data
                will be extracted.
        file_format : string, optional
            Specifies wether data_file is a .csv file or a file containing
            nmea sentences
        file_mode : string, optional
            Reading mode to be passed to the read_nmea_file-function
            in the case that data_file is a file containing nmea_sentences
            Defaults to "mean"
        tw : bool, optional
            Specifies if the given wind data should be viewed as true wind
            If False, wind data will be converted to true wind
            Defaults to True
        filtering : bool, optional
            Specifies if the data should be filtered after weighing
            Defaults to True
        Returns
        -------
        out : PolarDiagram
            An instance of the given p_type based on the input data
        """
        if data is None and data_file is None:
            raise PipelineException("No data was specified")
        if data is None:
            data, tw = _read_file(data_file, file_format, file_mode, tw)

        w_pts = pc.WeightedPoints(data, weigher=self.weigher, tw=tw)
        if filtering:
            mask = self.filter.filter(w_pts.weights)
            w_pts = w_pts[mask]

        self.extension.process(w_pts)


def _read_file(data_file, file_format, mode, tw):
    if file_format not in {"csv", "nmea"}:
        raise PipelineException(
            f"No functionality for the specified file-format "
            f"{file_format} implemented"
        )

    if file_format == "csv":
        data = read_csv_file(data_file)
    else:
        data = read_nmea_file(data_file, mode=mode, tw=True)
        tw = True

    return data, tw


def read_csv_file(csv_path, delimiter=None):
    """Reads a .csv file of data points and returns a numpy.ndarray
    of those data points
    Parameters
    ----------
    csv_path : string
        Path to a .csv file which will be read
    delimiter : string, optional
        Delimiter used in the .csv file
        If nothing is passed, the python parsing engine will try to
        autodetect the used delimiter
    Returns
    -------
    out : numpy.ndarray
        Array of the data points contained in the .csv file
    """

    try:
        with open(csv_path, "r", newline="") as file:
            csv_reader = csv.reader(file, delimiter=delimiter)
            return np.array([[eval(pt) for pt in row] for row in csv_reader])
    except OSError:
        raise FileReadingException(f"Can't find/open/read {csv_path}")


# TODO: Also look for rmc sentences, if there are no vhws?
def read_nmea_file(nmea_path, mode="interpolate", tw=True):
    """Reads a text file containing nmea-sentences and extracts
    data points based on recorded wind speed, wind angle, and speed
    over water
    Function looks for sentences of type:
        - MWV for wind data
        - VHW for speed over water
    Parameters
    ----------
    nmea_path : string
        Path to a text file, containing nmea-0183 sentences, which will
        be read
    mode : string, optional
        In the case where there is more recorded wind data than speed data,
        specifies how to handle the surplus
            - "interpolate": handles the surplus by taking
            convex combinations of two recorded speed datas
            together with the recorded wind data "between" those
            two points to create multiple data points
            - "mean": handles the surplus by taking the mean
            of the wind data "belonging" to a given speed data
            to create a singe data point
        Defaults to "interpolate"
    tw : bool, optional
        Specifies if occuring apparent wind should be automatically
        converted to true wind.
        If False, each point will have an extra component,
        specifying if it is true or apparent wind
        Defaults to True
    Returns
    -------
    nmea_data : list
        If tw:
            A list of points consisting of wind speed, wind angle,
            and boat speed,with the wind being true wind
        else:
            A list of points consisting of wind speed, wind angle,
            boat speed, and a reference, specifying wether the wind
            is true or appearent wind
    Raises a FileReadingException
        - if file can't be found, opened, or read
        - if file isn't "sorted", meaning there has to be at least
        one recorded wind data "between" two recorded speed datas
        - if file is empty or doesn't contain any relevant sentences
        - if file contains invalid relevant nmea sentences
    """
    if mode not in ("mean", "interpolate"):
        raise FileReadingException(f"Mode {mode} not implemented")

    try:
        with open(nmea_path, "r") as nmea_file:
            nmea_data = []
            nmea_stcs = filter(
                lambda line: "VHW" in line or "MWV" in line, nmea_file
            )

            stc = next(nmea_stcs, None)
            if stc is None:
                raise FileReadingException(
                    "File didn't contain any relevant nmea sentences"
                )

            while True:
                try:
                    bsp = pynmea2.parse(stc).data[4]
                except pynmea2.ParseError as pe:
                    raise FileReadingException(
                        f"During parsing of {stc}, the error {pe} occured"
                    )

                stc = next(nmea_stcs, None)

                if stc is None:
                    # eof
                    break

                # check if nmea-file is in a
                # way "sorted"
                if "VHW" in stc:
                    raise FileReadingException(
                        "No recorded wind data in between recorded speed "
                        "data. Parsing not possible"
                    )

                wind_data = []
                while "VHW" not in stc and stc is not None:
                    _get_wind_data(wind_data, stc)
                    stc = next(nmea_stcs, None)

                _process_data(nmea_data, wind_data, stc, bsp, mode)

            if tw:
                aw = [data[:3] for data in nmea_data if data[3] == "R"]
                tw = [data[:3] for data in nmea_data if data[3] != "R"]
                if not aw:
                    return tw

                aw = apparent_wind_to_true(aw)
                return tw.extend(aw)

            return nmea_data
    except OSError:
        raise FileReadingException(f"Can't find/open/read {nmea_path}")


def _get_wind_data(wind_data, stc):
    try:
        wind = pynmea2.parse(stc)
    except pynmea2.ParseError:
        raise FileReadingException(
            f"Invalid nmea-sentences encountered: {stc}"
        )

    wind_data.append(
        [float(wind.wind_speed), float(wind.wind_angle), wind.reference]
    )


def _process_data(nmea_data, wind_data, stc, bsp, mode):
    if mode == "mean":
        wind_arr = np.array([w[:2] for w in wind_data])
        wind_arr = np.mean(wind_arr, axis=0)
        nmea_data.append([wind_arr[0], wind_arr[1], bsp, wind_data[0][2]])

    if mode == "interpolate":
        try:
            bsp2 = pynmea2.parse(stc).data[4]
        except pynmea2.ParseError as pe:
            raise FileReadingException(
                f"During parsing of {stc}, the error {pe} occured"
            )

        inter = len(wind_data)
        for i in range(inter):
            inter_bsp = ((inter - i) / inter) * bsp + (i / inter) * bsp2
            nmea_data.append(
                [wind_data[i][0], wind_data[i][1], inter_bsp, wind_data[i][2]]
            )


class TableExtension(PipelineExtension):
    def __init__(
        self,
        w_res=None,
        neighbourhood: pc.Neighbourhood = None,
        interpolator: pc.Interpolator = None,
    ):
        self.w_res = w_res

        if neighbourhood is None:
            neighbourhood = pc.Ball()
        if not isinstance(neighbourhood, pc.Neighbourhood):
            raise PipelineException(
                f"{neighbourhood.__name__} is not a Neighbourhood"
            )
        self.neighbourhood = neighbourhood

        if interpolator is None:
            interpolator = pc.ArithmeticMeanInterpolator(1, 1)
        if not isinstance(interpolator, pc.Interpolator):
            raise PipelineException(
                f"{interpolator.__name__} is not an Interpolator"
            )
        self.interpolator = interpolator

    def process(self, w_pts: pc.WeightedPoints):
        w_res = _set_wind_resolution(self.w_res, w_pts.points)
        bsp = _interpolate_grid_points(
            w_res, w_pts, self.neighbourhood, self.interpolator
        )

        try:
            return pol.PolarDiagramTable(
                ws_res=w_res[0], wa_res=w_res[1], bsps=bsp
            )
        except PolarDiagramException as pe:
            raise PipelineException(
                f"During creation of the polar diagram, exception {pe} occured"
            )


# TODO Add options for radians
class CurveExtension(PipelineExtension):
    def __init__(self, regressor: pc.Regressor = None):
        if regressor is None:
            regressor = pc.ODRegressor(
                model_func=mf.tws_s_s_dt_twa_gauss_comb,
                init_values=(0.25, 10, 1.7, 0, 1.9, 30, 17.6, 0),
            )
        if not isinstance(regressor, pc.Regressor):
            raise PipelineException(f"{regressor.__name__} is not a Regressor")
        self.regressor = regressor

    def process(self, w_pts: pc.WeightedPoints):
        self.regressor.fit(w_pts.points)

        try:
            return pol.PolarDiagramCurve(
                self.regressor.model_func, *self.regressor.optimal_params
            )
        except PolarDiagramException as pe:
            raise PipelineException(
                f"During creation of the polar diagram, exception {pe} occured"
            )


class PointcloudExtension(PipelineExtension):
    def __init__(
        self,
        sampler: pc.Sampler = None,
        neighbourhood: pc.Neighbourhood = None,
        interpolator: pc.Interpolator = None,
    ):
        if sampler is None:
            sampler = pc.UniformRandomSampler(500)
        if not isinstance(sampler, pc.Sampler):
            raise PipelineException(f"{sampler.__name__} is not a Sampler")
        self.sampler = sampler

        if neighbourhood is None:
            neighbourhood = pc.Ball()
        if not isinstance(neighbourhood, pc.Neighbourhood):
            raise PipelineException(
                f"{neighbourhood.__name__} is not a Neighbourhood"
            )
        self.neighbourhood = neighbourhood

        if interpolator is None:
            interpolator = pc.ArithmeticMeanInterpolator(1, 1)
        if not isinstance(interpolator, pc.Interpolator):
            raise PipelineException(
                f"{interpolator.__name__} is not an Interpolator"
            )
        self.interpolator = interpolator

    def process(self, w_pts: pc.WeightedPoints):
        sample_pts = self.sampler.sample(w_pts.points)
        pts = []
        logger.info(
            f"Beginning to interpolate sample_pts with "
            f"{self.interpolator.__name__}"
        )
        for s_pt in sample_pts:
            mask = self.neighbourhood.is_contained_in(
                w_pts.points[:, :2] - s_pt
            )
            if not np.any(mask):
                raise PipelineException(
                    f"No points where contained in the neighbourhood of "
                    f"{s_pt}. Interpolation not possible"
                )
            pts.append(self.interpolator.interpolate(w_pts[mask], s_pt))

        try:
            return pol.PolarDiagramPointcloud(pts=pts)
        except PolarDiagramException as pe:
            raise PipelineException(
                f"During creation of the polar diagram, exception {pe} occured"
            )


def _set_wind_resolution(w_res, pts):
    if w_res == "auto":
        ws_res = _extract_wind(pts[:, 0], 2, 100)
        wa_res = _extract_wind(pts[:, 1], 5, 30)
        return ws_res, wa_res

    if w_res is None:
        w_res = (None, None)

    ws_res, wa_res = w_res
    return set_resolution(ws_res, "speed"), set_resolution(wa_res, "angle")


# TODO Better approach?
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


def _interpolate_grid_points(w_res, w_pts, nhood, ipol):
    ws_res, wa_res = w_res
    bsp = np.zeros((len(wa_res), len(ws_res)))

    logger.info(f"Beginning to interpolate w_res with {ipol.__name__}")
    for i, ws in enumerate(ws_res):
        for j, wa in enumerate(wa_res):
            grid_point = np.array([ws, wa])
            mask = nhood.is_contained_in(w_pts.points[:, :2] - grid_point)
            if not np.any(mask):
                raise PipelineException(
                    f"No points were contained in the neighbourhood of "
                    f"{grid_point}. Interpolation not possible"
                )
            bsp[j, i] = ipol.interpolate(w_pts[mask], grid_point)

    return bsp
