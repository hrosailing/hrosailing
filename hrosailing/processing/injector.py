"""
Defines the base class `Injector` to be used as a component of the pipeline.

Also contains the predefined and ready to use injectors:

- `ZeroInjector`.
"""

from abc import ABC, abstractmethod

import numpy as np

from hrosailing.core.data import WeightedPoints
from hrosailing.core.statistics import ComponentWithStatistics


class Injector(ComponentWithStatistics, ABC):
    """
    Base class for all injector classes.
    """

    @abstractmethod
    def inject(self, weighted_points):
        """Method that should produce artificial weighted data points
        that are supposed to be appended to the original data.

        Parameters
        ----------
        weighted_points : WeightedPoints
            The original preprocessed points. `weighted_points.data` has to be an `ndarray`.

        Returns
        -------
        app_points : WeightedPoints
            Points to append to the original points.
        """

    def set_statistics(self, n_injected):
        super().set_statistics(n_injected=n_injected)


class ZeroInjector(Injector):
    """
    Injector which adds a fixed number of points with 0 boat speed.

    Parameters
    ----------
    n_zeros : int
        Number of artificial points to be added at 0 degree and at 360 degree
        respectively.

    See also
    ----------
    `Injector`
    """

    def __init__(self, n_zeros):
        super().__init__()
        self.n_zeros = n_zeros

    def inject(self, weighted_points):
        """Adds `n_zeros` points equally distributed in the `TWS` dimension with
        boat speed 0 and wind angle 0 and 360 respectively.


        Parameters
        ----------
        weighted_points : WeightedPoints
            The original preprocessed points with `weighted_points.data` an
            `numpy.ndarray` of shape (n, 3) containing at least one record.

        Returns
        -------
        app_points : WeightedPoints
            Points to append to the original points.

        See also
        ----------
        `Injector.inject`
        """

        data = weighted_points.data

        if not isinstance(data, np.ndarray):
            raise TypeError(
                "`weighted_points.data` should be an `numpy.ndarray` but is"
                f" {type(data)}."
            )
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(
                f"`weighted_points.data` should be of shape (n, 3)"
            )
        if len(data.T) == 0:
            raise ValueError(
                f"`weighted_points.data` should contain at least one record."
            )

        ws = data[:, 0]
        ws = np.linspace(min(ws), max(ws), self.n_zeros)

        zero = np.zeros(self.n_zeros)
        full = 360 * np.ones(self.n_zeros)
        zeros = np.column_stack((ws, zero, zero))
        fulls = np.column_stack((ws, full, zero))

        self.set_statistics(2 * self.n_zeros)
        return WeightedPoints(
            data=np.concatenate([zeros, fulls]),
            weights=np.ones(2 * self.n_zeros),
        )
