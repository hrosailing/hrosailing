"""
A Pipeline class to automate getting a
polar diagram from "raw" data
"""

# Author: Valentin F. Dannenberg / Ente

import csv
import logging.handlers
import numpy as np
import pynmea2

import hrosailing.polardiagram as pol
import hrosailing.processing.modelfunctions as mf
import hrosailing.processing.pipelinecomponents as pc

from hrosailing.polardiagram.polardiagram import (
    FileReadingException,
    PolarDiagramException,
)
from hrosailing.wind import (
    apparent_wind_to_true,
    speed_resolution,
    angle_resolution,
)

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


class PolarPipeline:
    """A Pipeline class to create polar diagrams from raw data

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

    # TODO Make it better
    def __init__(
        self,
        weigher=None,
        filter_=None,
        sampler=None,
        interpolater=None,
        regressor=None,
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

        if sampler is None:
            sampler = pc.UniformRandomSampler(500)
        if not isinstance(sampler, pc.Sampler):
            raise PipelineException(f"{sampler.__name__} is not a Sampler")
        self._sampler = sampler

        if interpolater is None:
            interpolater = pc.ArithmeticMeanInterpolator(1, 1)
        if not isinstance(interpolater, pc.Interpolator):
            raise PipelineException(
                f"{interpolater.__name__} is not an Interpolator"
            )
        self._interpolater = interpolater

        if regressor is None:
            regressor = pc.ODRegressor(
                model_func=mf.tws_s_s_dt_twa_gauss_comb,
                init_values=(0.25, 10, 1.7, 0, 1.9, 30, 17.6, 0),
            )
        if not isinstance(regressor, pc.Regressor):
            raise PipelineException(f"{regressor.__name__} is not a Regressor")
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
        pass

    def __call__(
        self,
        p_type: pol.PolarDiagram,
        data=None,
        data_file=None,
        file_format=None,
        file_mode="mean",
        tw=True,
        filtering=True,
        w_res=None,
        neighbourhood=None,
    ):
        """

        Parameters
        ----------
        p_type : PolarDiagram
            Specifies the type of polar diagram, that is to be created

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

        w_res : tuple of length 2 or string, optional
            Only used if p_type is PolarDiagramTable

            Specifies the wind speed and wind angle resolution
            for the PolarDiagramTable. Can either be
                - a tuple of length 2 containing the wind speed
                and wind angle resolution given in the same manner
                as for pol.PolarDiagramTable
                - the string "auto", in which case the function will try
                to extract a good wind resolution based on the given data

            If nothing is passed w_res will default to
            (numpy.arange(2, 42, 2), numpy.aragen(0, 360, 5))

        neighbourhood : Neighbourhood, optional
            Only used if p_type is PolarDiagramTable or PolarDiagramPointcloud

            Specifies the neighbourhood of the grid and sample points
            to be used for interpolation

            If nothing is passed, neighbourhood will default to pc.Ball()

        Returns
        -------
        out : PolarDiagram
            An instance of the given p_type based on the input data
        """
        # TODO: Really necessarry? Different approach?
        if p_type not in {
            pol.PolarDiagramTable,
            pol.PolarDiagramCurve,
            pol.PolarDiagramPointcloud,
        }:
            raise PipelineException(f"An invalid PolarDiagram-type {p_type} was specified")

        if data is None and data_file is None:
            raise PipelineException("No data was specified")
        if data is None:
            data, tw = _read_file(data_file, file_format, file_mode, tw)

        w_pts = pc.WeightedPoints(data, weigher=self.weigher, tw=tw)
        if filtering:
            mask = self.filter.filter(w_pts.weights)
            w_pts = w_pts[mask]

        if p_type is pol.PolarDiagramTable:
            return _create_polar_diagram_table(
                w_pts, w_res, neighbourhood, self.interpolater
            )

        if p_type is pol.PolarDiagramCurve:
            return _create_polar_diagram_curve(w_pts, self.regressor)

        return _create_polar_diagram_pointcloud(
            w_pts, neighbourhood, self.interpolater, self.sampler
        )


def _read_file(data_file, file_format, mode, tw):
    if file_format not in {"csv", "nmea"}:
        raise PipelineException(
            f"No functionality for the"
            f"specified file-format"
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


# TODO: Also look for rmc sentences, if
#       there are no vhws?
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
                bsp = pynmea2.parse(stc).spd_over_water
            except pynmea2.ParseError:
                raise FileReadingException(
                    f"Invalid nmea-sentences encountered: {stc}"
                )

            stc = next(nmea_stcs, None)

            if stc is None:
                # eof
                break

            # check if nmea-file is in a
            # way "sorted"
            if "VHW" in stc:
                raise FileReadingException("No recorded wind data in between recorded speed data. Parsing not possible")

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
            bsp2 = pynmea2.parse(stc).spd_over_grnd
        except pynmea2.ParseError:
            raise FileReadingException(
                f"Invalid nmea-sentences encountered: {stc}"
            )

        inter = len(wind_data)
        for i in range(inter):
            inter_bsp = (inter - i) / inter * bsp + i / inter * bsp2
            nmea_data.append(
                [wind_data[i][0], wind_data[i][1], inter_bsp, wind_data[i][2]]
            )


def _create_polar_diagram_table(w_pts, w_res, neighbourhood, interpolater):
    if neighbourhood is None:
        neighbourhood = pc.Ball()
    if not isinstance(neighbourhood, pc.Neighbourhood):
        raise PipelineException(
            f"{neighbourhood.__name__} is not a Neighbourhood"
        )

    w_res = _set_wind_resolution(w_res, w_pts.points)
    bsps = _interpolate_grid_points(w_res, w_pts, neighbourhood, interpolater)

    try:
        return pol.PolarDiagramTable(
            ws_res=w_res[0], wa_res=w_res[1], bsps=bsps
        )
    except PolarDiagramException as pe:
        raise PipelineException(
            f"During creation of the polar diagram, exception {pe} occured"
        )


def _create_polar_diagram_curve(w_pts, regressor):
    # regressor.set_weights(w_pts.weights)
    regressor.fit(w_pts.points)

    try:
        return pol.PolarDiagramCurve(
            regressor.model_func, *regressor.optimal_params
        )
    except PolarDiagramException as pe:
        raise PipelineException(
            f"During creation of the polar diagram, exception {pe} occured"
        )


def _create_polar_diagram_pointcloud(
    w_pts, neighbourhood, interpolater, sampler
):
    if neighbourhood is None:
        neighbourhood = pc.Ball()
    if not isinstance(neighbourhood, pc.Neighbourhood):
        raise PipelineException(
            f"{neighbourhood.__name__} is not a Neighbourhood"
        )

    sample_pts = sampler.sample(w_pts.points)
    pts = []
    for s_pt in sample_pts:
        mask = neighbourhood.is_contained_in(w_pts.points[:, :2] - s_pt)
        pts.append(interpolater.interpolate(w_pts[mask], s_pt))

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
    return speed_resolution(ws_res), angle_resolution(wa_res)


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
    bsps = np.zeros((len(wa_res), len(ws_res)))

    for i, ws in enumerate(ws_res):
        for j, wa in enumerate(wa_res):
            grid_point = np.array([ws, wa])
            mask = nhood.is_contained_in(w_pts.points[:, :2] - grid_point)
            bsps[j, i] = ipol.interpolate(w_pts[mask], grid_point)
    return bsps
