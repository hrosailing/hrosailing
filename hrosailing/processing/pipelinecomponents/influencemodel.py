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
    def remove(self, data):
        pass

    @abstractmethod
    def add(self, data):
        pass


class LinearCurrentModel(InfluenceModel):
    """"""

    def remove(self, data):
        pass

    def add(self, data):
        pass
