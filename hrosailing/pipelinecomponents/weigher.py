"""
Contains the baseclass for Weighers used in the PolarPipeline class,
that can also be used to create custom Weighers.

Also contains two predefined and useable weighers, the CylindricMeanWeigher
and the CylindricMemberWeigher, aswell as the WeightedPoints class, used to
represent data points together with their respective weights
"""

# Author: Valentin Dannenberg


import logging.handlers
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

import hrosailing._logfolder as log
from hrosailing.wind import _convert_wind

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

        Defaults to 0.05

    norm : function or callable, optional
        Norm with which to evaluate the distances, ie ||.||

        If nothing is passed, it will default to ||.||_2


    Raises a WeigherInitializationException if radius is nonpositive
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

    def weigh(self, pts):
        """Weigh given points according to the method described above

        Parameters
        ----------
        pts : numpy.ndarray of shape (n, 3)
            Points to be weight

        Returns
        -------
        wts : numpy.ndarray of shape (n, )
            Normalized weights of the input points
        """
        shape = pts.shape
        wts = np.zeros(shape[0])

        for i, pt in enumerate(pts):
            mask = self._norm(pts[:, :2] - pt[:2]) <= self._radius
            cylinder = pts[mask][:, 2]

            # in case there are on points in cylinder
            std = np.std(cylinder) or 1
            mean = np.mean(cylinder)

            wts[i] = np.abs(mean - pt[2]) / std

        logger.info(f"Mean (non-normalized) weight: {np.mean(wts)}")
        logger.info(f"Maximum (non-normalized) weight: {np.max(wts)}")
        logger.info(f"Minimum (non-normalized) weight: {np.min(wts)}")

        wts = 1 - wts / max(wts)

        logger.info(f"Mean (normalized) weight: {np.mean(wts)}")
        logger.info(f"Final (normalized) weights calculated for {pts}: {wts}")
        return wts


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

        Defaults to 0.05

    length : nonnegative int of float, optional
        The height of the considered cylinder, ie h

        If length is 0, the cylinder is a d-1 dimensional ball

        Defaults to 0.05

    norm : function or callable, optional
        Norm with which to evaluate the distances, ie ||.||

        If nothing is passed, it will default to ||.||_2


    Raises a WeigherInitializationException

    - if radius is nonpositive
    - if length is negative
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

    def weigh(self, pts):
        """Weigh given points according to the method described above

        Parameters
        ----------
        pts : numpy.ndarray of shape (n, 3)
            Points to be weight

        Returns
        -------
        wts : numpy.ndarray of shape (n, )
            Normalized weights of the input points
        """
        wts = np.zeros(pts.shape[0])
        for i, pt in enumerate(pts):
            mask_l = np.abs(pts[:, 0] - pt[0]) <= self._length
            mask_r = self._norm(pts[:, 1:] - pt[1:]) <= self._radius
            wts[i] = len(pts[mask_l & mask_r]) - 1

        logger.info(f"Mean (non-normalized) weight: {np.mean(wts)}")
        logger.info(f"Maximum (non-normalized) weight: {np.max(wts)}")
        logger.info(f"Minimum (non-normalized) weight: {np.min(wts)}")

        wts = wts / max(wts)

        logger.info(f"Mean (normalized) weight: {np.mean(wts)}")
        logger.info(f"Final (normalized) weights calculated for {pts}: {wts}")
        return wts


class WeightedPoints:
    """A class to weigh data points and represent them together
    with their respective weights

    Parameters
    ----------
    pts : array_like of shape (n, 3)
        Points that will be weight or paired with given weights

    wts : int, float or array_like of shape (n, ), optional
        If the weights of the points are known beforehand,
        they can be given as an argument. If weights are
        passed, they will be assigned to the points
        and no further weighing will take place

        If a scalar is passed, the points will all be assigned
        the same weight

        Defaults to None

    weigher : Weigher, optional
        Instance of a Weigher class, which will weigh the points

        Will only be used if weights is None

        If nothing is passed, it will default to CylindricMeanWeigher()

    tw : bool, optional
        Specifies if the given wind data should be viewed as true wind

        If False, wind data will be converted to true wind

        Defaults to True

    """

    def __init__(
        self,
        pts,
        wts=None,
        weigher: Weigher = CylindricMeanWeigher,
        tw=True,
        _checks=True,
    ):
        if _checks:
            pts = _convert_wind(pts, -1, tw=tw, _check_finite=True)

        self._pts = pts

        shape = pts.shape[0]

        if wts is None:
            self._wts = _sanity_checks(weigher.weigh(pts), shape)
        elif isinstance(wts, (int, float)):
            self._wts = np.array([wts] * shape)
        elif _checks:
            self._wts = _sanity_checks(wts, shape)
        else:
            self._wts = wts

    def __getitem__(self, mask):
        return WeightedPoints(
            pts=self.points[mask], wts=self.weights[mask], _checks=False
        )

    @property
    def points(self):
        """Returns a read-only version of self._pts"""
        return self._pts.copy()

    @property
    def weights(self):
        """Returns a read-only version of self._wts"""
        return self._wts.copy()


def _sanity_checks(wts, shape):
    wts = np.asarray(wts)

    if wts.dtype is object:
        raise WeighingException("`wts` is not array_like")

    if wts.shape != (shape,):
        raise WeighingException("`wts` has incorrect shape")

    return wts
