"""
Contains the baseclass for Expanders used in the `PolarPipeline` class,
that can also be used to create custom Expanders.

Also contains predefined and useable smoothers:
"""

from abc import ABC, abstractmethod

import pipelinecomponents as pc


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

class WeatherExpander(Expander):
    """
    Expander that uses a weather model to add weather data to given data
    if the fields `datetime`, `lat` and `lon` are defined
    """

    def __init__(self, weather_model):
        self._weather_model = weather_model

    def expand(self, data):
        weather_data = [
            self._weather_model.get_weather([datetime, lat, lon])
            for datetime, lat, lon in data.rows(["datetime", "lat", "lon"])
        ]
        weather_data = pc.data.Data.concatenate(weather_data)
        data.update(weather_data)
        return data, {}


