"""
Contains the baseclass for Weighers used in the PolarPipeline class,
that can also be used to create custom Weighers.

Also contains two predefined and useable weighers, the CylindricMeanWeigher
and the CylindricMemberWeigher, aswell as the WeightedPoints class, used to
represent data points together with their respective weights
"""


from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ._utils import euclidean_norm, scaled_norm, data_dict_to_numpy, scaled_euclidean_norm

from hrosailing.pipelinecomponents.data import Data
from hrosailing.pipelinecomponents.constants import NORM_SCALES

class WeightedPointsInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of WeightedPoints
    """


class WeigherInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of a Weigher
    """


class WeighingException(Exception):
    """Exception raised if an error occurs during the calling
    of the .weigh() method"""


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

    Raises
    ------
    WeightedPointsInitializationException

    WeighingException
    """

    def __init__(
        self,
        data,
        weights=None
    ):
        self.data = data
        if isinstance(weights, (float, int)):
            if isinstance(data, dict):
                length = len(list(data.values())[0])
            else:
                length = len(data)
            self.weights = weights*np.ones(length)
        else:
            self.weights = weights

    def __getitem__(self, mask):
        if isinstance(self.data, dict):
            return WeightedPoints(
                data={
                    key: list(np.array(value)[mask])
                    for key, value in self.data.items()
                },
                weights=self.weights[mask]
            )
        return WeightedPoints(
            data=self.data[mask], weights=self.weights[mask]
        )

    def extend(self, other):
        """
        Extends this weighted points by other weighted points.
        If both data is given as a dictionary of lists, the respective lists
        will be extended.
        Keys that are not present in both dictionaries are discarded.

        Parameter
        ----------
        other: WeightedPoints
            points to be appended
        """

        if isinstance(self.data, dict):
            self.data = {
                key: value + other.data[key]
                for key, value in self.data.items()
                if key in other.data
            }
        else:
            self.data = np.concatenate([self.data, other.data])

        self.weights = np.concatenate([self.weights, other.weights])


def get_weight_statistics(weights):
    """
    produces hrosailing standard statistics for weighers, namely the average,
    minimal and maximal weight as well as a list of percentages how many
    weights are contained in each 10th of the span between the minimal and
    maximal weight.
    The respective keys are 'average_weight', 'min_weight', 'max_weight' and
    'quantiles'.
    """
    minw = np.min(weights)
    span = np.max(weights) - minw
    return {
        "average_weight": round(np.mean(weights), 4),
        "minimal_weight": round(minw, 4),
        "maximal_weight": round(np.max(weights), 4),
        "quantiles": [
            round(
                100 * len(
                    [w for w in weights if
                     (w >= minw + span*i / 10)
                     and (w <= minw + span*(i + 1) / 10)]
                ) / len(weights),
                2
            ) for i in range(10)
        ]
    }


class Weigher(ABC):
    """Base class for all weigher classes


    Abstract Methods
    ----------------
    weight(self, pts)
    """

    @abstractmethod
    def weigh(self, points) -> (WeightedPoints, dict):
        """This method should be used, to determine a weight for each point
        where the points might be given as data_dicts or as np.arrays
        and return the result as WeightedPoints as well as a dictionary
        with statistics
        """


class AllOneWeigher(Weigher):
    """A weigher that functions mainly as a stand in if the weighing options
    of the pipeline are set to False. Weighs everything as 1"""

    def weigh(self, points) -> (np.ndarray, dict):
        return np.ones(len(points)), {}


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

        If nothing is passed, it will automatically detect a scaled euclidean norm with respect to the used dimensions

    Raises
    ------
    WeigherInitializationException
        If radius is nonpositive
    """

    def __init__(
        self,
        radius=0.05,
        norm=None,
        dimensions=None
    ):
        if radius <= 0:
            raise WeigherInitializationException("`radius` is not positive")

        self._radius = radius
        self._norm = norm
        self._dimensions = dimensions

    def __repr__(self):
        return (
            f"CylindricMeanWeigher(radius={self._radius}, "
            f"norm={self._norm.__name__})"
        )

    def weigh(self, points):
        """Weigh given points according to the method described above

        Parameters
        ----------
        points : numpy.ndarray of shape (n, d) with d>=3 or dict
            Points to be weight

        Returns
        -------
        WeightedPoints : numpy.ndarray of shape (n,)
            Normalized weights of the input points
        """
        self._dimensions, points = _set_points_from_data(points, self._dimensions)
        weights = [self._calculate_weight(point, points) for point in points]
        weights = np.array(weights)
        weights = 1 - _normalize(weights, np.max)

        statistics = get_weight_statistics(weights)

        return weights, statistics

    def _calculate_weight(self, point, points):
        points_in_cylinder = self._determine_points_in_cylinder(point, points)

        std = _standard_deviation_of(points_in_cylinder)
        mean = _mean_of(points_in_cylinder)

        return np.abs(mean - point[-1]) / std

    def _determine_points_in_cylinder(self, point, points):
        if self._norm is None:
            self._norm = hrosailing_standard_scaled_euclidean_norm(self._dimensions)
        in_cylinder = self._norm(points - point) <= self._radius
        return points[in_cylinder][:, -1]


def _standard_deviation_of(points_in_cylinder):
    return np.std(points_in_cylinder) or 1  # in case that there are no points


def _mean_of(points_in_cylinder):
    return np.mean(points_in_cylinder)


def _normalize(weights, normalizer):
    return weights / normalizer(weights)


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

    dimensions : [str] or None, optional
        If the data is given as dict, 'dimensions' contains the keys
        which should be used in conjunction in order to create the
        data space. If `None` all keys of the given `dict` are used.

        Defaults to None

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
        dimensions=None
    ):
        if radius <= 0:
            raise WeigherInitializationException("`radius´ is not positive")

        if length < 0:
            raise WeigherInitializationException("`length` is not nonnegative")

        self._radius = radius
        self._length = length
        self._norm = norm
        self._dimensions = dimensions

    def __repr__(self):
        return (
            f"CylindricMemberWeigher(radius={self._radius},"
            f"length={self._length}, norm={self._norm.__name__})"
        )

    def weigh(self, points):
        """Weigh given points according to the method described above

        Parameters
        ----------
        points : numpy.ndarray of shape (n, d) or dict
            Points to be weight

        Returns
        -------
        weights : numpy.ndarray of shape (n,)
            Normalized weights of the input points
        """

        self._dimensions, points = _set_points_from_data(points, self._dimensions)

        weights = [self._calculate_weight(point, points) for point in points]
        weights = np.array(weights)
        weights = _normalize(weights, np.max)

        statistics = get_weight_statistics(weights)
        return weights, statistics

    def _calculate_weight(self, point, points):
        points_in_cylinder = self._count_points_in_cylinder(point, points)
        return len(points_in_cylinder) - 1

    def _count_points_in_cylinder(self, point, points):
        height = np.abs(points[:, 0] - point[0]) <= self._length
        radius = self._norm(points[:, 1:] - point[1:]) <= self._radius

        cylinder = height & radius
        points_in_cylinder = cylinder[cylinder]
        return points_in_cylinder


class PastFluctuationWeigher(Weigher):
    """STILL A WIP"""

    def __init__(self, timespan=13, dimensions=None):
        self.timespan = timespan
        self._dimensions = dimensions

    def weigh(self, points):
        """WIP"""
        recording_times = np.array(points["datetime"])
        self._dimensions, points = _set_points_from_data(points, self._dimensions)
        weights = [
            self._calculate_weight(i, points, recording_times)
            for i in range(len(points))
        ]
        weights = np.array(weights)
        weights = len(weights) * _normalize(
            weights, np.sum
        )
        statistics = get_weight_statistics(weights)
        return weights, statistics

    def _calculate_weight(self, index, points, recording_times):
        in_time_interval = self._get_points_in_time_interval(
            index, recording_times
        )
        s2 = np.std(points[in_time_interval]) ** 2
        return 1/s2 if s2>0 else 0

    def _get_points_in_time_interval(self, index, recording_times):
        reference_time = recording_times[index]
        times_up_to_reference_time = recording_times[:index]
        return [
            i
            for i, time in enumerate(times_up_to_reference_time)
            if self._in_time_interval(time, reference_time)
        ]

    def _in_time_interval(self, time, reference_time):
        time_difference = reference_time - time
        return time_difference.total_seconds() <= self.timespan


class PastFutureFluctuationWeigher(Weigher):
    """STILL A WIP"""

    def __init__(self, timespan=6, dimensions=None):
        self.timespan = timespan
        self._dimensions = dimensions

    def weigh(self, points):
        """WIP"""
        if not isinstance(points, dict):
            raise WeighingException(
                "PastFluctuationWeigher can only be used as a Pre Weigher"
            )
        recording_times = np.array(points["datetime"])
        self._dimensions, points = _set_points_from_data(points, self._dimensions)
        weights = [
            self._calculate_weight(time, points, recording_times)
            for time in recording_times
        ]
        weights = np.array(weights)
        weights = len(weights) * _normalize(
            weights, np.sum
        )

        statistics = get_weight_statistics(weights)
        return weights, statistics

    def _calculate_weight(self, reference_time, points, recording_times):
        in_time_interval = self._get_points_in_time_interval(
            reference_time, recording_times
        )
        return 1 / np.std(points[in_time_interval]) ** 2

    def _get_points_in_time_interval(self, reference_time, recording_times):
        return [
            i
            for i, time in enumerate(recording_times)
            if self._in_time_interval(time, reference_time)
        ]

    def _in_time_interval(self, time, reference_time):
        time_difference = (
            reference_time - time
            if time <= reference_time
            else time - reference_time
        )
        return time_difference.total_seconds() <= self.timespan


def _set_points_from_data(data, dimensions):
    if isinstance(data, np.ndarray):
        return dimensions, data[:, :-1]

    if dimensions is None:
        dimensions = list(data.keys()) or list(data.keys)
    else:
        dimensions = dimensions

    if "BSP" in dimensions:
        dimensions.remove("BSP")
    if "SOG" in dimensions:
        dimensions.remove("SOG")

    if isinstance(data, dict):
        return dimensions, data_dict_to_numpy(data, dimensions)

    if isinstance(data, Data):
        return data[dimensions].numerical


def hrosailing_standard_scaled_euclidean_norm(dimensions):
    if dimensions is None:
        return scaled_euclidean_norm
    scales = [NORM_SCALES[key] if key in NORM_SCALES else 1 for key in dimensions]
    return scaled_norm(euclidean_norm, scales)