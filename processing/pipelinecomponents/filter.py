"""
Defines a baseclass for filters used in the
processing.processing.PolarPipeline class,
that can be used to create custom filters for use.

Also contains various predefined and usable filter
"""

# Author: Valentin F. Dannenberg / Ente


import numpy as np

from abc import ABC, abstractmethod

from exceptions import ProcessingException


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

    def __init__(self, percent=25):
        if percent < 0 or percent > 100:
            raise ProcessingException("")

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
        masks : numpy.ndarray of shape (n, )
            Boolean array describing
            with points are filtered
            depending on their resp.
            weight
        """

        wts = np.asarray(wts)
        if not wts.size:
            raise ProcessingException("")
        if not all(np.isfinite(wts)):
            raise ProcessingException("")

        return wts >= np.percentile(wts, self._percent)


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
            raise ProcessingException("")

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
        masks : numpy.ndarray of shape (n, )
            Boolean array describing
            with points are filtered
            depending on their resp.
            weight
        """

        wts = np.asarray(wts)
        if not wts.size:
            raise ProcessingException("")
        if not all(np.isfinite(wts)):
            raise ProcessingException("")

        mask_1 = wts >= self._l_b
        mask_2 = wts <= self._u_b
        return mask_1 & mask_2

