"""
Contains the baseclass `Expander` used in the `PolarPipeline` class,
that can also be used to create custom expanders.

Also contains predefined and usable expanders:

- `LazyExpander`,
- `WeatherExpander`.
"""

from abc import ABC, abstractmethod

from hrosailing.pipelinecomponents.data import Data


class Expander(ABC):
    """Base class for all expander classes."""

    @abstractmethod
    def expand(self, data):
        """
        Should be used to expand the data by more data fields resulting from additional data sources.

        Parameters
        ----------
        data : dict
            The data that should be expanded.

        Returns
        -------
        data : dict
            The processed data.
        statistics : dict
            Dictionary containing relevant statistics."""
        return data, {}


class LazyExpander(Expander):
    """
    Expander that doesn't do anything.
    """
    def expand(self, data):
        """
        See also
        --------
        `Expander.expand`
        """
        return data, {}


class WeatherExpander(Expander):
    """
    Expander that uses a weather model to add weather data to given data
    if the fields `datetime`, `lat` and `lon` are defined.

    Parameters
    ----------
    weather_model : WeatherModel
        A suitable weather model (yielding weather information for the required times, latitudes and longitudes).
    """

    def __init__(self, weather_model):
        self._weather_model = weather_model

    def expand(self, data):
        """
        Expands given data by the method described above.
        
        See also
        --------
        `Expander.expand`
        """
        weather_data = [
            self._weather_model.get_weather([datetime, lat, lon])
            for datetime, lat, lon in data.rows(
                ["datetime", "lat", "lon"], return_type=tuple
            )
        ]
        weather_data = Data.concatenate(weather_data)
        data.update(weather_data)
        data.hrosailing_standard_format()
        return data, {}


