"""

"""

# Author: Valentin F. Dannenberg / Ente

from abc import ABC, abstractmethod


class InfluenceException(Exception):
    """"""
    pass


class InfluenceModel(ABC):
    """"""

    @abstractmethod
    def remove_influence(self, data):
        pass

    @abstractmethod
    def add_influence(self, data):
        pass


class LinearCurrentModel(InfluenceModel):
    """"""

    def __init__(self):
        pass

    def remove_influence(self, data):
        pass

    def add_influence(self, data):
        pass
