"""
Defines a baseclass for weighers used in the
processing.processing.PolarPipeline class,
that can be used to create custom weighers for use.

Also contains various predefined and useable weighers,
aswell as the WeightedPoints class, used to
represent data points together with their
respective weights
"""

# Author: Valentin F. Dannenberg / Ente


import logging.handlers
import numpy as np

from abc import ABC, abstractmethod

from exceptions import ProcessingException
from utils import (
    euclidean_norm,
)
from windconversion import apparent_wind_to_true

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='../hrosailing/logging/processing.log')

LOG_FILE = "../hrosailing/logging/processing.log"

logger = logging.getLogger(__name__)
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when='midnight')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


def _convert_wind(wind_arr, tw):
    if tw:
        return wind_arr

    return apparent_wind_to_true(wind_arr)


class WeightedPoints:
    """A class to weigh data points
    and represent them together with
    their respective weights

    Parameters
    ----------
    pts : array_like of shape (n, d)
        Points that will
        be weight or paired
        with given weights
    wts : int, float or array_like of shape (n, ), optional
        If the weights of the
        points are known beforehand,
        they can be given as an
        argument. If weights are
        passed, they will be
        assigned to the points
        and no further weighing
        will take place

        If a scalar is passed,
        the points will all be
        assigned the same weight

        Defaults to None
    weigher : Weigher, optional
        Instance of a Weigher class,
        which will weigh the points

        Will only be used if
        weights is None

        If nothing is passed, it
        will default to
        CylindricMeanWeigher()
    tw : bool, optional
        Specifies if the
        given wind data should
        be viewed as true wind

        If False, wind data
        will be converted
        to true wind

        Defaults to True

    Methods
    -------
    points
    weights
    """

    def __init__(self, pts, wts=None,
                 weigher=None, tw=True):
        pts = np.asarray(pts)
        shape = pts.shape
        if not pts.size:
            raise ProcessingException("")
        self._points = _convert_wind(pts, tw)

        if weigher is None:
            weigher = CylindricMeanWeigher()
        if not isinstance(weigher, Weigher):
            raise ProcessingException(
                f"{weigher.__name__} is "
                f"not a Weigher")
        if wts is None:
            self._weights = weigher.weigh(pts)
            return

        if isinstance(wts, (int, float)):
            self._weights = np.array([wts] * shape[0])
            return
        wts = np.asarray(wts)
        try:
            wts = wts.reshape(shape[0], )
        except ValueError:
            raise ProcessingException(
                f"weights could not be broadcasted "
                f"to an array of shape ({shape[0]}, )")
        self._weights = wts

    def __getitem__(self, mask):
        return WeightedPoints(
            pts=self.points[mask],
            wts=self.weights[mask])

    @property
    def points(self):
        return self._points.copy()

    @property
    def weights(self):
        return self._weights.copy()


class Weigher(ABC):
    """Base class for all
    weigher classes

    Abstract Methods
    ----------------
    weight(self, pts)
    """

    @abstractmethod
    def weigh(self, pts):
        pass


class CylindricMeanWeigher(Weigher):
    """A weigher that
    weighs given points
    according to the
    following procedure:

    For a given point p
    and points pts
    we look at all the
    points pt in pts such that
    ||pt[:d-1] - p[:d-1]|| <= r.
    Then we take the mean m_p
    and standard deviation std_p of
    the dth component of all those
    points and set
    w_p = | m_p - p[d-1] | / std_p

    Parameters
    ----------
    radius : positive int or float, optional
        The radius of the
        considered cylinder,
        with infinite height,
        ie r

        Defaults to 1
    norm : function or callable, optional
        Norm with which to
        evaluate the distances,
        ie ||.||

        If nothing is passed, it
        will default to ||.||_2

    Methods
    -------
    weigh(self, pts)
        Weigh given points
        according to the method
        described above
    """

    def __init__(self, radius=1, norm=None):

        # Sanity checks
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ProcessingException(
                f"The radius needs to be "
                f"positive number, but "
                f"{radius} was passed")
        if norm is None:
            norm = euclidean_norm
        if not callable(norm):
            raise ProcessingException(
                f"{norm.__name__} is not "
                f"callable")

        self._radius = radius
        self._norm = norm

    def __repr__(self):
        return (f"CylindricMeanWeigher("
                f"radius={self._radius}, "
                f"norm={self._norm.__name__})")

    def weigh(self, pts):
        """Weigh given points
        according to the method
        described above

        Parameters
        ----------
        pts : array_like of shape (n, d)
            Points to be weight

        Returns
        -------
        wts : numpy.ndarray of shape (n, )
            Normalized weights of
            the input points
        """
        pts = np.asarray(pts)
        shape = pts.shape
        if not pts.size:
            raise ProcessingException(
                "No points were passed")

        d = shape[1]
        wts = np.zeros(shape[0])

        for i, pt in enumerate(pts):
            mask = self._norm(pts[:, :d - 1] - pt[:d - 1])\
                   <= self._radius
            cylinder = pts[mask][:, d - 1]
            std = np.std(cylinder) or 1
            mean = np.mean(cylinder) or 0
            wts[i] = np.abs(mean - pt[d - 1]) / std

        logger.info(f"Mean (non-normalized) "
                    f"weight: {np.mean(wts)}")
        logger.info(f"Maximum (non-normalized) "
                    f"weight: {np.max(wts)}")
        logger.info(f"Minimum (non-normalized) "
                    f"weight: {np.min(wts)}")

        wts = wts / max(wts)

        logger.info(f"Mean (normalized) weight: "
                    f"{np.mean(wts)}")
        logger.info(f"Final (normalized) weights "
                    f"calculated for {pts}: {wts}")
        return wts


class CylindricMemberWeigher(Weigher):
    """A weigher that
    weighs given points
    according to the
    following procedure:

    For a given point p
    and points pts
    we look at all the
    points pt in pts such that
    |pt[0] - p[0]| <= l
    ||pt[1:] - p[1:]|| <= r
    Call the set of all such
    points P, then
    w_p = #P - 1, where #
    is the cardinality of
    a set

    Parameters
    ----------
    radius : positive int or float, optional
        The radius of the
        considered cylinder,
        ie r

        Defaults to 1
    length : nonnegative int of float, optional
        The height of the
        considered cylinder,
        ie l

        If length is 0,
        the cylinder is a
        d-1 dimensional ball

        Defaults to 1
    norm : function or callable, optional
        Norm with which to
        evaluate the distances,
        ie ||.||

        If nothing is passed, it
        will default to ||.||_2

    Methods
    -------
    weigh(self, pts)
        Weigh given points
        according to the method
        described above
    """

    def __init__(self, radius=1, length=1, norm=None):
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ProcessingException(
                f"The radius needs to be "
                f"positive number, but "
                f"{radius} was passed")
        if not isinstance(length, (int, float)) or length < 0:
            raise ProcessingException(
                f"The length needs to be "
                f"a nonnegative number, "
                f"but {length} was passed")
        if norm is None:
            norm = euclidean_norm
        if not callable(norm):
            raise ProcessingException(
                f"{norm.__name__} is not "
                f"callable")

        self._radius = radius
        self._length = length
        self._norm = norm

    def __repr__(self):
        return (f"CylindricMemberWeigher("
                f"radius={self._radius}, "
                f"length={self._length}, "
                f"norm={self._norm.__name__})")

    def weigh(self, pts):
        """Weigh given points
        according to the method
        described above

        Parameters
        ----------
        pts : array_like of shape (n, d)
            Points to be weight

        Returns
        -------
        wts : numpy.ndarray of shape (n, )
            Normalized weights of
            the input points
        """
        pts = np.asarray(pts)
        shape = pts.shape
        if not pts.size:
            raise ProcessingException(
                "No points were passed")

        wts = np.zeros(shape[0])
        for i, pt in enumerate(pts):
            mask_l = np.abs(pts[:, 0] - pt[0])\
                     <= self._length
            mask_r = self._norm(pts[:, 1:] - pt[1:])\
                     <= self._radius
            wts[i] = len(pts[mask_l & mask_r]) - 1

        logger.info(f"Mean (non-normalized) "
                    f"weight: {np.mean(wts)}")
        logger.info(f"Maximum (non-normalized) "
                    f"weight: {np.max(wts)}")
        logger.info(f"Minimum (non-normalized) "
                    f"weight: {np.min(wts)}")

        wts = wts / max(wts)

        logger.info(f"Mean (normalized) weight: "
                    f"{np.mean(wts)}")
        logger.info(f"Final (normalized) weights "
                    f"calculated for {pts}: {wts}")
        return wts
