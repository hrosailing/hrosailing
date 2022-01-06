"""
Contains the baseclass for Weighers used in the PolarPipeline class,
that can also be used to create custom Weighers.

Also contains two predefined and useable weighers, the CylindricMeanWeigher
and the CylindricMemberWeigher, aswell as the WeightedPoints class, used to
represent data points together with their respective weights
"""


import logging.handlers
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable

import numpy as np

import hrosailing._logfolder as log
from hrosailing.wind import convert_apparent_wind_to_true

from ._utils import scaled_euclidean_norm

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.handlers.TimedRotatingFileHandler(
            log.log_folder + "/pipeline.log", when="midnight"
        )
    ],
)
logger = logging.getLogger(__name__)
del log


class WeightedPointsInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of WeightedPoints
    """


class WeightedPoints:
    """A class to weigh data points and represent them together
    with their respective weights

    Parameters
    ----------
    data : dict or numpy.ndarray
        Points that will be weight or paired with given weights

    weights : scalar or array_like of shape (n,), optional
        If the weights of the points are known beforehand,
        they can be given as an argument. If weights are
        passed, they will be assigned to the points
        and no further weighing will take place

        If a scalar is passed, the points will all be assigned
        the same weight

        Defaults to `None`

    weigher : Weigher, optional
        Instance of a Weigher class, which will weigh the points
        Will only be used if weights is `None`

        If nothing is passed, it will default to `CylindricMeanWeigher()`

    apparent_wind : bool, optional
        Specifies if wind data is given in apparent wind 

        If `True`, data will be converted to true wind

        Defaults to `False`

    Raises
    ------
    WeightedPointsInitializationException

    WeighingException
    """

        def __init__(
        self, 
        data, 
        weights=None, 
        weigher: Weigher = CylindricMeanWeigher(),
        apparent_wind=False,
        _enable_logging=False,
    ):
        points = _extract_points_from_data(data)
        if apparent_wind:
            self.points = convert_apparent_wind_to_true(points)
        else:
            self.points = np.asarray_chkfinite(points)

        if weights is None:
                weights = _determine_weights(weigher, points, data, _enable_logging)
        elif _weights_is_scalar(weights):
            weights = np.array([weights] * pts.shape[0])

        self.weights = weights 

    def __getitem__(self, mask):
        return WeightedPoints(points=self.points[mask], weights=self.weights[mask])


def _extract_points_from_data(data):
    if isinstance(data, np.ndarray):
        return data

    ws = _get_wind_speeds_from_data(data)    
    wa = _get_wind_angles_from_data(data)
    bsp = _get_boat_speeds_from_data(data) 

    points = np.column_stack((ws, wa, bsp))

    if points.dtype is object:
        raise WeightedPointsException("`points` is not array_like or has some non-scalar values")

    return np.column_stack((ws, wa, bsp))


def _get_wind_speeds_from_data(data):
    WIND_SPEED_KEYS = {"Wind Speed", "Wind speed", "wind speed", "wind_speed", "WS", "ws"}

    return _get_entries(data, WIND_SPEED_KEYS)


def _get_entries(data, keys):
    for key in keys:
        try:
            entry = data.pop(key)
            return entry
        except KeyError:
            continue
    else:
        raise WeightedPointsException("Essential data is missing, can't proceed")


def _get_wind_angles_from_data(data):
    WIND_ANGLE_KEYS = {"Wind Angle", "Wind angle", "wind angle", "wind_angle", "WA", "wa"}

    return _get_entries(data, WIND_ANGLE_KEYS)


def _get_boat_speeds_from_data(data):
    BOAT_SPEED_KEYS = {"Boat Speed", "Boat speed", "boat speed", "boat_speed", "BSPS", "BSP", "bsps", "bsp", "Speed Over Ground", "Speed over ground", "Speed over Ground", "speed over ground", "speed_over_ground", "speed over ground knots", "speed_over_ground_knots", "SOG", "sog", "Water Speed", "Water speed", "water speed", "water_speed", "Water Speed knots", "Water speed knots", "water speed knots", "water_speed_knots" "WSP", "wsp"}

    return _get_entries(data, BOAT_SPEED_KEYS)


def _determine_weights(weigher, points, data, _enable_logging):
    weights = weigher.weigh(points, data, _enable_logging)
    weights = np.asarray(weights)

    if weights.dtype is object:
        raise WeighingException("`weights` is not array_like or has some non-scalar values")
    if weights.shape != (pts.shape[0],):
        raise Weighingexception("`weights` has incorrect shape")

    return weights

def _wts_is_scalar(weights):
    return np.isscalar(weights)


class WeigherInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of a Weigher
    """


class WeighingException(Exception):
    """Exception raised if an error occurs during the calling
    of the .weigh() method"""


class Weigher(ABC):
    """Base class for all weigher classes


    Abstract Methods
    ----------------
    weight(self, pts)
    """

    @abstractmethod
    def weigh(self, points, extra_data):
        """This method should be used, given certain point
        to determine their weights according to a weighing method, which
        can also use some extra data, depending on the points
        """


class CylindricMeanWeigher(Weigher):
    """A weigher that weighs given points according to the
    following procedure:

    For a given point p and points pts we look at all the points
    pt in pts such that ||pt[:d-1] - p[:d-1]|| <= r

    Then we take the mean m_p and standard deviation std_p
    of the dth component of all those points and set
    w_p = | m_p - p[d-1] | / std_p

    Parameters
    ----------
    radius : positive int or float, optional
        The radius of the considered cylinder, with infinite height, ie r

        Defaults to `0.05`

    norm : function or callable, optional
        Norm with which to evaluate the distances, ie ||.||

        If nothing is passed, it will default to ||.||_2

    Raises
    ------
    WeigherInitializationException
        If radius is nonpositive
    """

    def __init__(
        self,
        radius=0.05,
        norm: Callable = scaled_euclidean_norm,
    ):
        if radius <= 0:
            raise WeigherInitializationException("`radius` is not positive")

        self._radius = radius
        self._norm = norm

    def __repr__(self):
        return (
            f"CylindricMeanWeigher(radius={self._radius}, "
            f"norm={self._norm.__name__})"
        )

    def weigh(self, points, extra_data, _enable_logging):
        """Weigh given points according to the method described above

        Parameters
        ----------
        points : numpy.ndarray of shape (n, 3)
            Points to be weight

        Returns
        -------
        weights : numpy.ndarray of shape (n,)
            Normalized weights of the input points
        """
        weights = np.zeros(len(points))

        for i, point in enumerate(pts):
            weights[i] = self._calculate_weight(point, points)
                        
        if _enable_logging:
            _log_unnormalized_weights(weights)

        weights = 1 - _normalize_weights(weights)

        if _enable_logging:
            _log_normalized_weights(weights)

        return weights

    def _calculate_weight(self, point, points):
        points_in_cylinder = self._determine_points_in_cylinder(point, points)

        std = _standard_deviation_in_cylinder(points_in_cylinder)            
        mean = _mean_in_cylinder(points_in_cylinder)

        return np.abs(mean - point[2]) / std

    def _determine_points_in_cylinder(self, point, points):
        in_cylinder = self._norm(points[:, :2] - point[:2]) <= self._radius
        return points[in_cylinder][:, 2]


def _standard_deviation_in_cylinder(points_in_cylinder):
    return np.std(points_in_cylinder) or 1 # in case that there are no points


def _mean_in_cylinder(points_in_cylinder):
    return np.mean(points_in_cylinder)


def _normalize_weights(weights):
    return weights / np.max(weights)


def _log_unnormalized_weights(weights):
    logger.info(f"Unnormalized weights: {weights}")
    logger.info(f"Unnormalized mean weight: {np.mean(weights)}")
    logger.info(f"Unnormalized maximum weight: {np.max(weights)}")
    logger.info(f"Unnormalized minimum weight: {np.min(weights)}")


def _log_normalized_weights(weights):
    logger.info(f"Normalized weights: {weights}")
    logger.info(f"Normalized mean weight: {np.mean(weights)}")
    logger.info(f"Normalized maximum weight: {np.max(weights)}")
    logger.info(f"Normalized minimum weight: {np.min(weights)}")


class CylindricMemberWeigher(Weigher):
    """A weigher that weighs given points according to the
    following procedure:

    For a given point p and points pts
    we look at all the points pt in pts such that
    |pt[0] - p[0]| <= h and ||pt[1:] - p[1:]|| <= r

    Call the set of all such points P, then w_p = #P - 1

    Parameters
    ----------
    radius : positive int or float, optional
        The radius of the considered cylinder, ie r

        Defaults to `0.05`

    length : nonnegative int of float, optional
        The height of the considered cylinder, ie h

        If length is 0, the cylinder is a d-1 dimensional ball

        Defaults to `0.05`

    norm : function or callable, optional
        Norm with which to evaluate the distances, ie ||.||

        If nothing is passed, it will default to ||.||_2

    Raises
    ------
    WeigherInitializationException

        - If radius is nonpositive
        - If length is negative
    """

    def __init__(
        self,
        radius=0.05,
        length=0.05,
        norm: Callable = scaled_euclidean_norm,
    ):
        if radius <= 0:
            raise WeigherInitializationException("`radius´ is not positive")

        if length < 0:
            raise WeigherInitializationException("`length` is not nonnegative")

        self._radius = radius
        self._length = length
        self._norm = norm

    def __repr__(self):
        return (
            f"CylindricMemberWeigher(radius={self._radius},"
            f"length={self._length}, norm={self._norm.__name__})"
        )

    def weigh(self, points, extra_data, _enable_logging):
        """Weigh given points according to the method described above

        Parameters
        ----------
        points : numpy.ndarray of shape (n, 3)
            Points to be weight

        Returns
        -------
        weights : numpy.ndarray of shape (n,)
            Normalized weights of the input points
        """
        weights = np.zeros(len(points))

        for i, point in enumerate(points):
            weights[i] = self._calculate_weight(point, points)

        if _enable_logging:
            _log_unnormalized_weights(weights)

        weights = _normalize_weights(weights)

        if _enable_logging:
            _log_normalized_weights(weights)

        return weights

    def _calculate_weight(self, point, points):
        no_points_in_cylinder = self._count_points_in_cylinder(point, points)
        return len(points_in_cylinder) - 1

    def _count_points_in_cylinder(self, point, points):
        height = np.abs(points[:, 0] - point[0]) <= self._length
        radius = self._norm(points[:, 1:] - point[1:]) <= self._radius

        cylinder = height & radius
        points_in_cylinder = cylinder[cylinder]
        return points_in_cylinder


class PastFluctuationWeigher(Weigher):
    """"""

    def __init__(self, timespan=13):
        self.timespan = timespan


    def weigh(self, points, extra_data, _enable_logging):
        """"""
        weights = np.zeros(len(points))

        for i, point in enumerate(points):
            weights[i] = self._calculate_weight(i, point, points, extra_data)

        if _enable_logging:
            _log_unnormalized_weights(weights)

        weights = len(weights) * _normalize_weights(weights)

        if _enable_logging:
            _log_normalized_weights(weights)

        return weights

    def _calculate_weight(self, index, point, points, data):
        recording_times = _get_recording_times(data)
        in_time_interval = self._get_points_in_time_interval(index, recording_times)
        considered_points = points[in_time_interval]
        
        return 1 / np.std(considered_points) ** 2

    def _get_points_in_time_interval(index, recording_times):
        in_time_interval = [index]

        reference_time = recording_times[index]
        times_up_to_reference_time = recording_times[:index]

        for i, time in enumerate(times_up_to_reference_time):
            if self._in_time_interval(time, reference_time):
                in_time_interval.append(i)

        return in_time_interval

    def _in_time_interval(self, time, reference_time):
        time_difference = reference_time - time
        return time_difference.total_seconds() <= self.timespan

        
def _get_recording_times(data):
    time_stamps = _get_time_stamps(extra_data)
    date_stamps = _get_date_stamps(extra_data)

    date_and_time_stamps = zip(date_stamps, time_stamps)
    recording_times = [datetime.combine(date_stamp, time_stamp) for (date_stamp, time_stamp) in date_and_time_stamps]
    return recording_times


def _get_time_stamps(data):
    TIME_STAMP_KEYS = {"Timestamp", "timestamp", "Time", "time", "time_stamp"}

    return _get_entries(data, TIME_STAMP_KEYS)


def _get_date_stamps(data):
    DATE_STAMP_KEYS = {"Datestamp", "datestamp", "Date", "date", "date_stamp"}

    return _get_entries(data, DATE_STAMP_KEYS)


class PastFutureFluctuationWeigher(Weigher):
    """"""

    def __init___(self, timespan=6):
        self.timespan = timespan

    def weigh(self, points, extra_data, _enable_logging):
        """"""
        weights = np.zeros(len(points))

        for i, point in enumerate(points):
            weights[i] = self._calculate_weight(i, point, points, extra_data)

        if _enable_logging:
            _log_unnormalized_weights(weights)

        weights = len(weights) * _normalize_weights(weights)

        if _enable_logging:
            _log_normalized_weights(weights)

        return weights

    def _calculate_weight(self, index, point, points, data):
        recording_times = _get_recording_times(data)
        in_time_interval = self._get_points_in_time_interval(index, recording_times)
        considered_points = points[in_time_interval]
        
        return 1 / np.std(considered_points) ** 2

    def _get_points_in_time_interval(self, index, recording_times):
        in_time_interval = []

        reference_time = recording_times[index]

        for i, time in enumerate(recording_times):
            if self._in_time_interval(time, reference_time):
                in_time_interval.append(i)

        return in_time_interval

    def _in_time_interval(self, time, reference_time):
        if time <= reference_time:
            time_difference = reference_time - time
        else:
            time_difference = time - reference_time

        return time_difference.total_seconds() <= self.timespan

