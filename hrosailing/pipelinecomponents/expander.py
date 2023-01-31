"""
Contains the baseclass `Expander` used in the `PolarPipeline` class,
that can also be used to create custom expanders.

Also contains predefined and usable expanders:

- `LazyExpander`,
- `WeatherExpander`.
"""

from abc import ABC, abstractmethod

from hrosailing.cruising.weather_model import OutsideGridException
from hrosailing.pipelinecomponents._utils import ComponentWithStatistics
from hrosailing.pipelinecomponents.data import Data


class Expander(ComponentWithStatistics, ABC):
    """Base class for all expander classes."""

    @abstractmethod
    def expand(self, data):
        """
        Should be used to expand the data by more data fields resulting from additional data sources.

        Parameters
        ----------
        data : Data
            The data that should be expanded.

        Returns
        -------
        data : Data
            The processed data.
        """

    def set_statistics(self, data):
        super().set_statistics(n_rows=data.n_rows, n_cols=data.n_cols)


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
        self.set_statistics(data)
        return data


class WeatherExpander(Expander):
    """
    Expander that uses a weather model to add weather data to given data
    if the fields `datetime`, `lat` and `lon` are defined.

    Parameters
    ----------
    weather_model : WeatherModel
        A suitable weather model (yielding weather information for the required times, latitudes and longitudes).

    exception_handling_mode: {"delete", "ignore"}, optional
        Describes how to handle weather cases which throw an OutsideGridException.

        - "delete" : delete the occurances
        - "ignore" : fill the corresponding records with `None`

        Defaults to "delete"
    """

    def __init__(self, weather_model, exception_handling_mode="delete"):
        super().__init__()
        self._weather_model = weather_model
        self._exception_handling_mode = exception_handling_mode

    def expand(self, data):
        """
        Expands given data by the procedure described above.

        See also
        --------
        `Expander.expand`
        """
        weather_data = self._get_weather(data)
        weather_data.hrosailing_standard_format()
        weather_data = self._mark_duplicate_keys(weather_data, data)

        data.update(weather_data)
        data.hrosailing_standard_format()
        self.set_statistics(data)
        return data

    def _get_weather(self, data):
        weather_keys = None
        none_idxs = []
        weather_list = []
        for idx, (datetime, lat, lon) in enumerate(
            data.rows(["datetime", "lat", "lon"], return_type=tuple)
        ):
            try:
                weather = self._weather_model.get_weather([datetime, lat, lon])
                weather_list.append(weather)
                if weather_keys is None:
                    weather_keys = weather.keys()
            except OutsideGridException:
                if self._exception_handling_mode == "ignore":
                    weather_list.append(None)
                none_idxs.append(idx)

        if self._exception_handling_mode == "delete":
            data.delete(none_idxs)
        else:
            for idx in none_idxs:
                weather_list[idx] = {key: None for key in weather_keys}

        return Data.concatenate(weather_list)

    def _mark_duplicate_keys(self, weather_data, data):
        for key in list(weather_data.keys()):
            if key in data.keys():
                weather_data.rename(key, f"{key}(duplicate)")
        return weather_data
