"""
Classes used for modular modeling of different filtering methods based
on weights

Defines the Filter Abstract Base Class that can be used to create
custom filtering methods

Subclasses of Filter can be used with the PolarPipeline class
in the hrosailing.pipeline module
"""

# Author: Valentin Dannenberg


import logging.handlers
from abc import ABC, abstractmethod

import numpy as np

import hrosailing._logfolder as log

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


class FilterInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of a Filter
    """


class Filter(ABC):
    """Base class for all filter classes


    Abstract Methods
    ----------------
    filter(self, weights)
    """

    @abstractmethod
    def filter(self, wts):
        """This method should be used, given an array of weights,
        to filter out points based on their weights, and produce a
        boolean array of the same size as wts
        """


class QuantileFilter(Filter):
    """A filter that filteres all points based on if their
    resp. weight lies above a certain quantile

    Parameters
    ----------
    percent: int or float, optional
        The quantile to be calculated

        Defaults to 25


    Raises a FilterInitializationException, if percent
    is not in the interval [0, 100]
    """

    def __init__(self, percent=50):
        if percent < 0 or percent > 100:
            raise FilterInitializationException(
                "`percent` is not between 0 and 100"
            )

        self._percent = percent

    def __repr__(self):
        return f"QuantileFilter(percent={self._percent})"

    def filter(self, wts):
        """Filters a set of points given by their resp. weights
        according to the above described method

        Parameters
        ----------
        wts : numpy.ndarray of shape (n, )
            Weights of the points that are to be filtered, given
            as a sequence of scalars

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing with points are filtered
            depending on their resp. weight
        """
        mask = wts >= np.percentile(wts, self._percent)

        logger.info(f"Total amount of filtered points: {wts[mask].shape[0]}")
        logger.info(
            f"Percentage of filtered "
            f"points: {wts[mask].shape[0]/ wts.shape[0]}"
        )

        return mask


class BoundFilter(Filter):
    """A filter that filters all points based on if their
    weight is outside an interval given by a lower and upper bound

    Parameters
    ----------
    upper_bound : int or float, optional
        The upper bound for the filter

        Defaults to 1

    lower_bound : int or float, optional
        The lower bound for the filter

        Defaults to 0.5


    Raises a FilterInitializationException if
    lower_bound is greater than upper_bound
    """

    def __init__(self, upper_bound=1, lower_bound=0.5):
        if upper_bound < lower_bound:
            raise FilterInitializationException(
                "`upper_bound` is smaller than `lower_bound`"
            )

        self._u_b = upper_bound
        self._l_b = lower_bound

    def __repr__(self):
        return f"BoundFilter(upper_bound={self._u_b}, lower_bound={self._l_b})"

    def filter(self, wts):
        """Filters a set of points given by their resp. weights
        according to the above described method

        Parameters
        ----------
        wts : numpy.ndarray of shape (n, )
            Weights of the points that are to be filtered, given
            as a sequence of scalars

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing with points are filtered
            depending on their resp. weight
        """
        mask_1 = wts >= self._l_b
        mask_2 = wts <= self._u_b
        mask = mask_1 & mask_2

        logger.info(f"Total amount of filtered points: {wts[mask].shape[0]}")
        logger.info(
            f"Percentage of filtered "
            f"points: {wts[mask].shape[0] / wts.shape[0]}"
        )

        return mask
