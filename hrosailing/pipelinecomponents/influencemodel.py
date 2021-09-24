"""

"""

# Author: Valentin Dannenberg

from abc import ABC, abstractmethod

from hrosailing.polardiagram import PolarDiagram


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
        lists of diffrent data at points in time, to get a nx3 array_like
        output where the columns correspond to wind speed, wind angle and
        boat speed respectively.

        The dictionary should contain atleast keys for Wind speed, Wind angle
        and either Speed over ground, Speed over water or Boat speed
        """

    @abstractmethod
    def add_influence(self, pd: PolarDiagram, influence_data: dict):
        """This method should be used, given a polar diagram and a
        dictionary, to obtain a modified boat speed of that given
        in the polar diagram, based on the influencences presented in
        the given dictionary, such as wave height, underlying currents etc.
        """


class LinearCurrentModel(InfluenceModel):
    """"""

    def __init__(self):
        """"""

    def remove_influence(self, data: dict):
        """"""

    def add_influence(self, pd, influence_data: dict):
        """"""
