"""
Contains the baseclass `Smoother` used in the `PolarPipeline` class
that can also be used to create custom smoothers.

Also contains predefined and usable smoothers:

- `LazySmoother`,
- `AffineSmoother`.
"""
import datetime
from abc import ABC, abstractmethod

from hrosailing.pipelinecomponents._utils import ComponentWithStatistics


class Smoother(ComponentWithStatistics, ABC):
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
        """


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
        return data


class AffineSmoother(Smoother):
    """
    Locates intervals in which the data, interpreted as a time series,
    is constant.
    Between two intervals, the arithmetic mean of the values of the respective
    intervals is used to replace the measured data.
    Then, data points in such an interval are replaced using affine
    interpolation.

    Parameters
    ---------
    timespan: datetime.timedelta
        Amount of time before the actual value of an interval is assumed to be the real value.
    """

    def __init__(self, timespan=datetime.timedelta(seconds=30)):
        super().__init__()
        self._timespan = timespan.total_seconds()

    def smooth(self, data):
        """
        Smooths data using the procedure described above.

        See also
        --------
        `Smoother.smooth`
        """
        if any(time < data["datetime"][0] for time in data["datetime"]):
            raise ValueError(
                "AffineSmoother only supports chronologically ordered time"
                " series."
            )
        for key in data.keys():
            if data.type(key) == float:
                data = self._smooth_field(key, data)
        return data

    def _smooth_field(self, key, data):
        ys = data[key].copy()
        start_time = data["datetime"][0]
        xs = [(time - start_time).total_seconds() for time in data["datetime"]]

        interval_bounds_x, interval_bounds_y = self._get_interval_bounds(
            xs, ys
        )

        approx_intervals_x, approx_intervals_y = self._approximate_intervals(
            interval_bounds_x, interval_bounds_y, ys
        )

        smooth_data = self._smooth_data_from_approx_intervals(
            approx_intervals_x, approx_intervals_y, xs, data, key
        )

        return smooth_data

    def _get_interval_bounds(self, xs, ys):
        x_lb = xs[0]
        y_lb = ys[0]
        interval_bounds_x = []
        interval_bounds_y = []
        for i, (x, x_after, y, y_after) in enumerate(
            zip(xs, xs[1:], ys, ys[1:])
        ):
            if i == len(ys) - 2:
                interval_bounds_x.append((x_lb, x_after))
                interval_bounds_y.append(y)
                break
            if y == y_after:
                continue
            interval_bounds_x.extend([(x_lb, 1 / 2 * (x + x_after))])
            interval_bounds_y.append(y_lb)
            x_lb = 1 / 2 * (x + x_after)
            y_lb = y_after

        return interval_bounds_x, interval_bounds_y

    def _approximate_intervals(self, interval_bounds_x, interval_bounds_y, ys):
        approx_intervals_x = []
        approx_intervals_y = []

        y_before = ys[0]
        # for (x_lb, x_ub), y, y_after in zip(
        #         interval_bounds_x, interval_bounds_y, interval_bounds_y[1:] + [ys[-1]]
        # ):
        for (x_lb, x_ub), y, y_after in zip(
            interval_bounds_x,
            interval_bounds_y,
            interval_bounds_y[1:] + [ys[-1]],
        ):
            if x_ub - x_lb < 2 * self._timespan:
                midpt = 1 / 2 * (x_ub + x_lb)
                approx_intervals_x.extend([(x_lb, midpt), (midpt, x_ub)])
                approx_intervals_y.extend(
                    [(1 / 2 * (y_before + y), y), (y, 1 / 2 * (y_after + y))]
                )
            else:
                lb_ref = x_lb + self._timespan
                ub_ref = x_ub - self._timespan
                approx_intervals_x.extend(
                    [(x_lb, lb_ref), (lb_ref, ub_ref), (ub_ref, x_ub)]
                )
                approx_intervals_y.extend(
                    [
                        (1 / 2 * (y_before + y), y),
                        (y, y),
                        (y, 1 / 2 * (y_after + y)),
                    ]
                )
            y_before = y

        return approx_intervals_x, approx_intervals_y

    def _smooth_data_from_approx_intervals(
        self, approx_intervals_x, approx_intervals_y, xs, data, key
    ):
        x_lb, x_ub = approx_intervals_x[0]
        y_lb, y_ub = approx_intervals_y[0]
        interval_idx = 0
        for i, x in enumerate(xs):
            if x > x_ub:
                interval_idx += 1
                try:
                    x_lb, x_ub = approx_intervals_x[interval_idx]
                    y_lb, y_ub = approx_intervals_y[interval_idx]
                except IndexError:
                    break
                continue

            try:
                mu = (x - x_lb) / (x_ub - x_lb)
                data[key][i] = mu * y_ub + (1 - mu) * y_lb
            except ZeroDivisionError:
                data[key][i] = y_ub

        return data
