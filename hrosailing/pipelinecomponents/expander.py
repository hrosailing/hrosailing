"""
Contains the baseclass for Expanders used in the `PolarPipeline` class,
that can also be used to create custom Expanders.

Also contains predefined and useable smoothers:
"""

from abc import ABC, abstractmethod


class Expander(ABC):
    """Base class for all expander classes"""

    @abstractmethod
    def expand(self, data):
        """
        Should be used to expand the data by more data fields resulting from additional data sources.

        Parameters
        ----------
        data: dict
            The data that should be expanded

        Returns
        -----------
        data: dict,
            The processed data
        statistics: dict,
            Dictionary containing relevant statistics"""
        return data, {}


class LazyExpander(Expander):
    """
    Expander that doesn't do anything
    """
    def expand(self, data):
        return data, {}
