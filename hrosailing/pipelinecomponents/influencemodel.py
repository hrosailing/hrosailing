"""

"""

# Author: Valentin F. Dannenberg / Ente

from abc import ABC, abstractmethod


class InfluenceModel(ABC):
    """"""

    @abstractmethod
    def remove_influence(self, data: dict):
        pass

    @abstractmethod
    def add_influence(self, pd, influence_data: dict):
        pass


class LinearCurrentModel(InfluenceModel):
    """"""

    def __init__(self):
        pass

    def remove_influence(self, data: dict):
        pass

    def add_influence(self, pd, influence_data: dict):
        pass
