"""

"""

# Author: Valentin Dannenberg

from abc import ABC, abstractmethod


class InfluenceModel(ABC):
    """"""

    @abstractmethod
    def remove_influence(self, data: dict):
        """"""

    @abstractmethod
    def add_influence(self, pd, influence_data: dict):
        """"""


class LinearCurrentModel(InfluenceModel):
    """"""

    def __init__(self):
        pass

    def remove_influence(self, data: dict):
        """"""

    def add_influence(self, pd, influence_data: dict):
        """"""
