
from abc import ABC, abstractmethod


class Regressor(ABC):

    @abstractmethod
    def regress(self):
        pass