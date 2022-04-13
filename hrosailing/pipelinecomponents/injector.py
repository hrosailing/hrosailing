from abc import ABC, abstractmethod

import numpy as np

import hrosailing.pipelinecomponents as pc


class Injector(ABC):
    @abstractmethod
    def inject(self, weighted_points):
        """Method that should produce artificial weighted data points
        supposed to be appended to the original data"""


class ZeroInjector(Injector):
    """
    Injector which adds a fixed number points with 0 boat speed

    Parameter
    --------
    n_zeros: int,
        number of artificial points to be added at 0 degree and at 360 degree
        respectively
    """

    def __init__(self, n_zeros):
        self.n_zeros = n_zeros

    def inject(self, weighted_points):
        """Adds 'n_zeros' points equally distributed in the TWS dimension with
        boat speed 0 and wind angle 0 and 360 respectively.


        Parameter
        --------
        weighted_points: WeightedPoints,
            the original preprocessed points

        Returns
        -------
        app_points, statistics: WeightedPoints, dict
            points to append to the original points
            'statistics' is empty

        """

        ws = weighted_points.data[:, 0]
        ws = np.linspace(min(ws), max(ws), self.n_zeros)

        zero = np.zeros(self.n_zeros)
        full = 360 * np.ones(self.n_zeros)
        zeros = np.column_stack((ws, zero, zero))
        fulls = np.column_stack((ws, full, zero))

        return pc.WeightedPoints(
            data=np.concatenate([zeros, fulls]),
            weights=np.ones(2 * self.n_zeros),
        ), {}
