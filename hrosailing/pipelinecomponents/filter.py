"""
Classes used for modular modelling of different filtering methods based
on weights.

Defines the `Filter` abstract base class that can be used to create
custom filtering methods.

Subclasses of `Filter` can be used with the `PolarPipeline` class
in the `hrosailing.pipeline` module.
"""


from abc import ABC, abstractmethod

import numpy as np

from hrosailing.pipelinecomponents._utils import ComponentWithStatistics


class FilterInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of a `Filter`.
    """


class Filter(ComponentWithStatistics, ABC):
    """Base class for all filter classes.


    Abstract Methods
    ----------------
    filter(self, wts)
    """

    @abstractmethod
    def filter(self, wts):
        """This method should be used, given an array of weights,
        to filter out points based on their weights, and produce a
        boolean array of the same size as `wts` and a dictionary containing
        statistics.

        Parameters
        ----------
        wts : numpy.ndarray of shape (n,)
            Weights of the points that are to be filtered, given
            as a sequence of scalars.

        Returns
        -------
        filtered_points : numpy.ndarray of shape (n,)
            Boolean array describing which points are filtered.
        """

    def set_statistics(self, filtered_points):
        """
        Computes standard statistics for the output of a filter.

        Parameters
        ----------
        filtered_points : boolean array
            Boolean array describing which points are filtered.
        """
        n_filtered_points = len([f for f in filtered_points if not f])

        super().set_statistics(
            n_filtered_points=n_filtered_points,
            n_rows=len(filtered_points) - n_filtered_points,
            error=None,
        )


class QuantileFilter(Filter):
    """A filter that removes the points with the lowest weights until a given percentage of points have been removed.

    Parameters
    ----------
    percent : int or float, optional

        Defaults to `50`.

    Raises
    ------
    FilterInitializationException
        If `percent` is not in the interval :math:`[0, 100]`.
    """

    def __init__(self, percent=50):
        super().__init__()
        if percent < 0 or percent > 100:
            raise FilterInitializationException(
                "`percent` is not between 0 and 100"
            )

        self._percent = percent

    def __repr__(self):
        return f"QuantileFilter(percent={self._percent})"

    def filter(self, wts):
        """Filters a set of points given by their resp. weights
        according to the above described method.

        See also
        --------
        `Filter.filter`
        """
        filtered_points = self._calculate_quantile(wts)

        self.set_statistics(filtered_points)

        return filtered_points

    def _calculate_quantile(self, wts):
        return wts >= np.percentile(wts, self._percent)


class BoundFilter(Filter):
    """A filter that filters all points based on if their
    weight is outside an interval given by a lower and an upper bound.

    Parameters
    ----------
    lower_bound : int or float, optional
        The lower bound for the filter.

        Defaults to `0.5`.

    upper_bound : int or float, optional
        The upper bound for the filter.

        Defaults to `1`.

    Raises
    ------
    FilterInitializationException
        If `lower_bound` is greater than `upper_bound`.
    """

    def __init__(self, lower_bound=0.5, upper_bound=1):
        super().__init__()
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
        according to the above described procedure.

        See also
        --------
        `Filter.filter`
        """
        filtered_points = self._determine_points_within_bound(wts)

        self.set_statistics(filtered_points)

        return filtered_points

    def _determine_points_within_bound(self, wts):
        points_above_lower_bound = wts >= self._l_b
        points_below_upper_bound = wts <= self._u_b

        return points_above_lower_bound & points_below_upper_bound
