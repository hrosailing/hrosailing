"""
Defines a baseclass for filters used in the
processing.processing.PolarPipeline class,
that can be used to create custom filters for use.

Also contains various predefined and usable filter
"""

# Author: Valentin F. Dannenberg / Ente

import logging.handlers
import numpy as np

from abc import ABC, abstractmethod

from hrosailing.exceptions import ProcessingException

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='hrosailing/logging/processing.log')

LOG_FILE = "hrosailing/logging/processing.log"

logger = logging.getLogger(__name__)
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when='midnight')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


class Filter(ABC):
    """Base class for
    all filter classes

    Abstract Methods
    ----------------
    filter(self, weights)
    """

    @abstractmethod
    def filter(self, weights):
        pass


class QuantileFilter(Filter):
    """A filter that
    filteres all points
    based on if their
    resp. weight lies
    above a certain
    quantile

    Parameters
    ----------
    percent: int or float, optional
        The quantile to be
        calculated

        Should be larger than
        0 and smaller than 100

        Defaults to 25

    Methods
    -------
    filter(self, wts)
        Filters a set
        of points given by
        their resp. weights
    """

    def __init__(self, percent=50):
        if percent < 0 or percent > 100:
            raise ProcessingException(
                f"The percentage needs to"
                f"be between 0 and 100, but"
                f"{percent} was passed")

        self._percent = percent

    def __repr__(self):
        return f"QuantileFilter(percent={self._percent})"

    def filter(self, wts):
        """Filters a set
        of points given by
        their resp. weights

        Parameters
        ----------
        wts : array_like of shape (n, )
            Weights of the points
            that are to be filtered,
            given as a sequence of
            scalars

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing
            with points are filtered
            depending on their resp.
            weight
        """

        wts = np.asarray(wts)
        if not wts.size:
            raise ProcessingException(
                "No weights were passed")
        if not all(np.isfinite(wts)):
            raise ProcessingException(
                "Weights need to be "
                "finite and can't "
                "be NaNs")

        mask = wts >= np.percentile(wts, self._percent)

        logger.info(f"Total amount of filtered"
                    f"points: {wts[mask].shape[0]}")
        logger.info(f"Percentage of filtered"
                    f"points: {wts[mask].shape[0]/ wts.shape[0]}")

        return mask


class BoundFilter(Filter):
    """A filter that
    filters all points
    based on if their
    weight is outside
    a lower and upper
    bound

    Parameters
    ----------
    upper_bound : int or float, optional
        The upper bound for
        the filter

        Defaults to 1
    lower_bound : int or float, optional
        The lower bound for
        the filter

        Defaults to 0.5

    Methods
    -------
    filter(self, wts)
        Filters a set
        of points given by
        their resp. weights
    """

    def __init__(self, upper_bound=1, lower_bound=0.5):
        if upper_bound < lower_bound:
            raise ProcessingException(
                "The upper bound can't be"
                "lower than the lower bound")

        self._u_b = upper_bound
        self._l_b = lower_bound

    def __repr__(self):
        return (f"BoundFilter("
                f"upper_bound={self._u_b}, "
                f"lower_bound={self._l_b})")

    def filter(self, wts):
        """Filters a set
        of points given by
        their resp. weights

        Parameters
        ----------
        wts : array_like of shape (n, )
            Weights of the points
            that are to be filtered,
            given as a sequence of
            scalars

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing
            with points are filtered
            depending on their resp.
            weight
        """

        wts = np.asarray(wts)
        if not wts.size:
            raise ProcessingException(
                "No weights were passed")
        if not all(np.isfinite(wts)):
            raise ProcessingException(
                "Weights need to be "
                "finite and can't "
                "be NaNs")

        mask_1 = wts >= self._l_b
        mask_2 = wts <= self._u_b
        mask = mask_1 & mask_2

        logger.info(f"Total amount of filtered"
                    f"points: {wts[mask].shape[0]}")
        logger.info(f"Percentage of filtered"
                    f"points: {wts[mask].shape[0] / wts.shape[0]}")

        return mask
