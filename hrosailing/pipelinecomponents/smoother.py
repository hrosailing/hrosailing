"""
Contains the baseclass for Smoothers used in the `PolarPipeline` class,
that can also be used to create custom Smoothers.

Also contains predefined and useable smoothers:
"""

from abc import ABC, abstractmethod


class Smoother(ABC):
    @abstractmethod
    def smooth(self, data):
        """
        Should be used to smoothen the measurement errors in data interpreted
        as a time series.

        Parameters
        ----------
        data: dict
            The data that should be smoothened

        Returns
        -----------
        data: dict,
            The processed data
        statistics: dict,
            Dictionary containing relevant statistics
        """
        return data, {}


class LazySmoother(Smoother):
    """
    Smoother that doesn't do anything
    """
    def smooth(self, data):
        return data, {}

class AffineSmoother(Smoother):
    """
    Locates intervals with same data in the numeric data and fills the rest
    with affine splines.
    This is done under the assumption, that between intervals the arithmetic
    mean of the respective values is taken and at the center of the interval the
    actual value of the interval is taken.
    """

    def smooth(self, data):
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
                    print(f"x = {x}, mid_pt = {mid_pt}")
                    if x <= mid_pt:
                        lamb = (x - x_lb) / (mid_pt - x_lb)
                        data[key][i_start + j] = lamb * y + (1 - lamb) * y_lb
                        # out["SOG"][i_start] = 0
                    else:
                        lamb = (x - mid_pt) / (x_ub - mid_pt)
                        data[key][i_start + j] = lamb * y_ub + (1 - lamb) * y
                        # out["SOG"][i_start + j] = 5
                i_start = i + 1
                x_lb = x_ub
                y_lb = y_ub
        return data