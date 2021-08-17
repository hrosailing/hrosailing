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
    def remove(self,):
        pass

    @abstractmethod
    def add(self, ):
        pass


class LinearCurrentModel(InfluenceModel):
    """"""

    def remove(self,):
        pass

    def add(self, ):
        pass
