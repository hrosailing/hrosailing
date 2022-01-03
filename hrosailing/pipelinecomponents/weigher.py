"""
Contains the baseclass for Weighers used in the PolarPipeline class,
that can also be used to create custom Weighers.

Also contains two predefined and useable weighers, the CylindricMeanWeigher
and the CylindricMemberWeigher, aswell as the WeightedPoints class, used to
represent data points together with their respective weights
"""


import logging.handlers
from abc import ABC, abstractmethod
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
    def weigh(self, pts):
        """This method should be used, given certain points,
        to determine their weights according to a weighing method
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

    def weigh(self, pts, _enable_logging=False):
        """Weigh given points according to the method described above

        Parameters
        ----------
        pts : numpy.ndarray of shape (n, 3)
            Points to be weight

        Returns
        -------
        wts : numpy.ndarray of shape (n,)
            Normalized weights of the input points
        """
        wts = np.zeros(pts.shape[0])

        for i, point in enumerate(pts):
            wts[i] = self._calculate_weight(point, pts)
                        
        if _enable_logging:
            _log_non_normalized_weights(wts)

        wts = 1 - _normalize_weights(wts)

        if _enable_logging:
            _log_normalized_weights(wts)

        return wts

    def _calculate_weight(self, point, pts):
        points_in_cylinder = self._determine_points_in_cylinder(point, pts)

        std = _standard_deviation_in_cylinder(points_in_cylinder)            
        mean = _mean_in_cylinder(points_in_cylinder)

        return np.abs(mean - point[2]) / 2

    def _determine_points_in_cylinder(self, point, pts):
        cylinder = self._norm(pts[:, :2] - points[:2]) <= self._radius
        return pts[cylinder][:, 2]


def _standard_deviation_in_cylinder(points_in_cylinder):
    return np.std(points_in_cylinder) or 1 # in case that there are no points


def _mean_in_cylinder(points_in_cylinder):
    return np.mean(points_in_cylinder)


def _normalize_weights(wts):
    return wts / np.max(wts)


def _log_non_normalized_weights(wts):
    logger.info(f"Mean (non-normalized) weight: {np.mean(wts)}")
    logger.info(f"Maximum (non-normalized) weight: {np.max(wts)}")
    logger.info(f"Minimum (non-normalized) weight: {np.min(wts)}")


def _log_normalized_weights(wts):
    logger.info(f"Mean (normalized) weight: {np.mean(wts)}")
    logger.info(f"Final (normalized) weights calculated for {pts}: {wts}")


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

    def weigh(self, pts, _enable_logging=False):
        """Weigh given points according to the method described above

        Parameters
        ----------
        pts : numpy.ndarray of shape (n, 3)
            Points to be weight

        Returns
        -------
        wts : numpy.ndarray of shape (n,)
            Normalized weights of the input points
        """
        wts = np.zeros(pts.shape[0])

        for i, point in enumerate(pts):
            wts[i] = _calculate_weight(point, pts)

        if _enable_logging:
            _log_non_normalized_weights(wts)

        wts = _normalize_weights(wts)

        if _enable_logging:
            _log_normalized_weights(wts)

        return wts

    def _calculate_weight(self, point, pts):
        points_in_cylinder = self._count_points_in_cylinder(point, pts)
        return len(points_in_cylinder) - 1

    def _count_points_in_cylinder(self, point, pts):
        height = np.abs(pts[:, 0] - point[0]) <= self._length
        radius = self._norm(pts[:, 1:] - point[1:]) <= self._radius

        cylinder = height & radius
        points_in_cylinder = cylinder[cylinder]
        return points_in_cylinder


class FluctuationWeigher(Weigher):
    """"""

    def __init__():
        return


    def weigh(self, pts, _enable_logging):
        """"""
        return


class WeightedPoints:
    """A class to weigh data points and represent them together
    with their respective weights

    Parameters
    ----------
    pts : array_like of shape (n, 3)
        Points that will be weight or paired with given weights

    wts : int, float or array_like of shape (n,), optional
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
    """

    def __init__(
        self, 
        pts, 
        wts=None, 
        weigher: Weigher = CylindricMeanWeigher(),
        apparent_wind=False
    ):
        if apparent_wind:
            self._pts = apparent_wind_to_true(pts)
        else:
            self._pts = np.asarray_chkfinite(pts)

        if wts is None:
            self._wts = _determine_weights_with_weigher(pts)
        elif _wts_is_scalar(wts):
            self._wts = np.array([wts] * pts.shape[0])
        else:
            self._wts = wts

    def __getitem__(self, mask):
        return WeightedPoints(pts=self.points[mask], wts=self.weights[mask])

    @property
    def points(self):
        """Returns a read-only version of self._pts"""
        return self._pts

    @property
    def weights(self):
        """Returns a read-only version of self._wts"""
        return self._wts


def _determine_weights(weigher, pts):
    wts = weigher.weigh(pts)

    wts = np.asarray(wts)
    if wts.dtype is object:
        raise WeighingException("`wts` is not array_like")
    if wts.shape != (pts.shape[0],):
        raise Weighingexception("`wts` has incorrect shape")

    return wts

def _wts_is_scalar(wts):
    return isinstance(wts, (int, float))

