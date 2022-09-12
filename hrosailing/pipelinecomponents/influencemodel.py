"""
Defines the `InfluenceModel` abstract base class that can be used
to create custom influence models.

Subclasses of InfluenceModel can be used with

- the PolarPipeline class in the hrosailing.pipeline module
- various functions in the hrosailing.cruising module
"""

from ._utils import data_dict_to_numpy
from hrosailing.wind import convert_apparent_wind_to_true

from abc import ABC, abstractmethod


class InfluenceException(Exception):
    """Raised when removing or adding influence does not work"""


class InfluenceModel(ABC):
    """Base class for all InfluenceModel classes


    Abstract Methods
    ----------------
    remove_influence(data)

    add_influence(pd, influence_data)
    """

    @abstractmethod
    def remove_influence(self, data: dict):
        """This method should be used, given a dictionary containing
        lists of different data at points in time, to get a nx3 array_like
        output where the columns correspond to wind speed, wind angle and
        boat speed respectively.

        The dictionary should contain atleast keys for Wind speed, Wind angle
        and either Speed over ground, Speed over water or Boat speed
        """

    @abstractmethod
    def add_influence(self, pd, influence_data: dict):
        """This method should be used, given a polar diagram and a
        dictionary, to obtain a modified boat speed of that given
        in the polar diagram, based on the influencences presented in
        the given dictionary, such as wave height, underlying currents etc.
        """

    @abstractmethod
    def fit(self, training_data: dict):
        """
        This method should be used to fit parameters of the influence
        model to the given training_data
        """


class IdentityInfluenceModel(InfluenceModel):
    """An influence model which ignores most influences and just calculates
    the true wind if only the apparent wind is given.
    IF 'BSP' is not provided by the data 'SOG' is used instead."""

    def remove_influence(self, data: dict):
        """
        Ignores all influences as described above

        Parameter
        ----------

        data: dict
            data dictionary, must either provide 'BSP' or 'SOG' key as well as
            either the keys 'TWS', 'TWA' or 'AWS', 'AWA'

        Returns
        -------
        (n,3)-array, {}
        """
        return _get_true_wind_data(data), {}

    def add_influence(self, pd, influence_data: dict):
        """
        Ignores all influences and strictly calculates the boat speed using the
        polar diagram.

        Parameter
        -------
        pd: PolarDiagram

        influence_data: dict
            either a dictionary of lists or a dictionary of values containing
            one or more sets of influence data.
            At least the keys 'TWS' and 'TWA' should be provided

        Returns
        ---------
        speeds, statistics: float or list of floats, {}
            the boat speed if 'influence_data' contained values,
            a list of respective boat speeds if 'influence_data' contained
            lists
        """
        if isinstance(influence_data["TWS"], list):
            wind = zip(influence_data["TWS"], influence_data["TWA"])
            speed = [pd(ws, wa) for ws, wa in wind]
        else:
            ws, wa = influence_data["TWS"], influence_data["TWA"]
            speed = pd(ws, wa)
        return speed, {}

    def fit(self, training_data: dict):
        """Does nothing, returns empty dictionary"""
        return {}


def _get_true_wind_data(data: dict):
    speed = "BSP" if "BSP" in data else "SOG"
    if "AWA" in data and "AWS" in data:
        apparent_data = data_dict_to_numpy(data, ["AWS", "AWA", speed])
        return convert_apparent_wind_to_true(apparent_data)
    elif "TWA" in data and "TWS" in data:
        if isinstance(data["TWS"], list):
            return data_dict_to_numpy(data, ["TWS", "TWA", speed])
        else:
            return np.array([data["TWS"], data["TWA"], data[speed]])
    else:
        raise InfluenceException(
            "No sufficient wind data is given in order to apply influence"
            "model. Either give 'AWA' and 'AWS' or 'TWA' and 'TWS'"
        )