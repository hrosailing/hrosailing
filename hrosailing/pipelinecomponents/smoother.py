"""
Contains the baseclass `Smoother` used in the `PolarPipeline` class
that can also be used to create custom smoothers.

Also contains predefined and usable smoothers:

- `LazySmoother`,
- `AffineSmoother`.
"""

from abc import ABC, abstractmethod


class Smoother(ABC):
    """
    Base class for all smoothers aimed to smoothen measurement errors in given data.
    """
    @abstractmethod
    def smooth(self, data):
        """
        Should be used to smoothen the measurement errors in `data`, where `data`
        is interpreted as a time series.

        Parameters
        ----------
        data : Data
            The data that should be smoothened.

        Returns
        -------
        data : Data
            The processed data.
        statistics : dict
            Dictionary containing relevant statistics.
        """
        return data, {}


class LazySmoother(Smoother):
    """
    Smoother that doesn't do anything.
    """
    def smooth(self, data):
        """
        Does not change the data and provides an empty statistics dictionary.

        See also
        --------
        `Smoother.smooth`
        """
        return data, {}


class AffineSmoother(Smoother):
    """
    Locates intervals in which the data, interpreted as a time series,
    is constant.
    Between two intervals, the arithmetic mean of the values of the respective
    intervals is used to replace the measured data.
    Then, data points in such an interval are replaced using affine
    interpolation.
    """

    def smooth(self, data):
        """
        Smooths data using the procedure described above.

        See also
        --------
        `Smoother.smooth`
        """
        for key in data.keys():
            if data.type(key) == float:
                data = self._smooth_field(key, data)
        return data, {}

    def _smooth_field(self, key, data):
        ys = data[key].copy()
        start_time = data["datetime"][0]
        xs = [(time - start_time).total_seconds() for time in data["datetime"]]

        i_start = 0
        x_lb = xs[0]
        y_lb = ys[0]
        for i, (x, x_after, y, y_after) in enumerate(
                zip(xs, xs[1:], ys, ys[1:])):
            if y != y_after or i == len(xs) - 2:
                y_ub = 1 / 2 * (y + y_after)
                x_ub = 1 / 2 * (x + x_after)
                mid_pt = 1 / 2 * (x_lb + x_ub)

                # affine approximation of sample points
                for j, x in enumerate(xs[i_start: i + 1]):
                    if x <= mid_pt:
                        try:
                            lamb = (x - x_lb) / (mid_pt - x_lb)
                        except ZeroDivisionError:
                            lamb = 0
                        data[key][i_start + j] = lamb * y + (1 - lamb) * y_lb
                    else:
                        try:
                            lamb = (x - mid_pt) / (x_ub - mid_pt)
                        except ZeroDivisionError:
                            lamb = 0
                        data[key][i_start + j] = lamb * y_ub + (1 - lamb) * y
                i_start = i + 1
                x_lb = x_ub
                y_lb = y_ub
        return data
