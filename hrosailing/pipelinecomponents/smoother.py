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