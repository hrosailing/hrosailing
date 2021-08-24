"""
Contains the baseclass for Weighers used in the PolarPipeline class,
that can also be used to create custom Weighers.

Also contains two predefined and useable weighers, the CylindricMeanWeigher
and the CylindricMemberWeigher, aswell as the WeightedPoints class, used to
represent data points together with their respective weights
"""

# Author: Valentin F. Dannenberg / Ente


import logging.handlers
from abc import ABC, abstractmethod

import numpy as np

from hrosailing.wind import convert_wind, WindException


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


def scaled(norm, scal):
    scal = np.array(list(scal))

    def scaled_norm(vec):
        return norm(scal * vec)

    return scaled_norm


def euclidean_norm(vec):
    return np.linalg.norm(vec, axis=1)


class WeightedPointsException(Exception):
    """Custom exception for errors that may appear whilst
    working with the WeightedPoints class
    """

    pass


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

    Raises a WeightedPointsException
        -


    Methods
    -------
    points
    weights
    """

    def __init__(self, pts, wts=None, weigher=None, tw=True, _checks=True):
        if _checks:
            try:
                pts = convert_wind(pts, -1, tw=tw, check_finite=False)
            except WindException as we:
                raise WeightedPointsException("") from we

        self._pts = pts

        if weigher is None:
            weigher = CylindricMeanWeigher()
        elif not isinstance(weigher, Weigher):
            raise WeightedPointsException("`weigher` is not a Weigher")

        try:
            self._wts = _set_weights(self.points, weigher, wts, _checks)
        except WeigherException as we:
            raise WeightedPointsException(
                "During weighing an error occured"
            ) from we

    def __getitem__(self, mask):
        return WeightedPoints(
            pts=self.points[mask], wts=self.weights[mask], _checks=False
        )

    @property
    def points(self):
        return self._pts.copy()

    @property
    def weights(self):
        return self._wts.copy()


def _set_weights(pts, weigher, wts, _checks):
    shape = pts.shape[0]

    if wts is None:
        return _sanity_checks(weigher.weigh(pts), shape)

    if isinstance(wts, (int, float)):
        return np.array([wts] * shape)

    if _checks:
        return _sanity_checks(wts, shape)

    return wts


def _sanity_checks(wts, shape):
    wts = np.asarray(wts)

    if wts.dtype is object:
        raise WeigherException("`wts` is not array_like")

    if wts.shape != (shape,):
        raise WeigherException("`wts` has incorrect shape")

    return wts


class WeigherException(Exception):
    """Custom exception for errors that may appear whilst
    working with the Weigher class and subclasses
    """

    pass


class Weigher(ABC):
    """Base class for all weigher classes


    Abstract Methods
    ----------------
    weight(self, pts)
    """

    @abstractmethod
    def weigh(self, pts):
        pass


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

        Defaults to 1

    norm : function or callable, optional
        Norm with which to evaluate the distances, ie ||.||

        If nothing is passed, it will default to ||.||_2

    Raises a WeigherException if inputs are not of the specified types


    Methods
    -------
    weigh(self, pts)
        Weigh given points according to the method described above
    """

    def __init__(self, radius=1, norm=None):
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise WeigherException("`radius` is not a positive number")

        if norm is None:
            norm = scaled(euclidean_norm, (1 / 40, 1 / 360))
        elif not callable(norm):
            raise WeigherException("`norm` is not callable")

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
            mean = np.mean(cylinder) or pt[2]

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

        Defaults to 1

    length : nonnegative int of float, optional
        The height of the considered cylinder, ie h

        If length is 0, the cylinder is a d-1 dimensional ball

        Defaults to 1

    norm : function or callable, optional
        Norm with which to evaluate the distances, ie ||.||

        If nothing is passed, it will default to ||.||_2

    Raises a WeigherException if inputs are not of the specified types

    Methods
    -------
    weigh(self, pts)
        Weigh given points according to the method described above
    """

    def __init__(self, radius=1, length=1, norm=None):
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise WeigherException("`radius´ is not a positive number")

        if not isinstance(length, (int, float)) or length < 0:
            raise WeigherException("`length` is not a nonnegative number")

        if norm is None:
            norm = scaled(euclidean_norm, (1 / 40, 1 / 360))
        elif not callable(norm):
            raise WeigherException("`norm` is not callable")

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
