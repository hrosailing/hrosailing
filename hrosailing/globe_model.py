"""
Contains an abstrat base class and several implementations of globe model.
Globe models can be used for computations with lattitude longitude coordinates
for different ways to describe the globe.
"""

from abc import ABC, abstractmethod


class GlobeModel(ABC):
    """ """

    @abstractmethod
    def distance(self, start, end):
        """
        Computes the distance between two points.

        Parameter
        ---------
        start : tuple  of size 2 of ints/floats,
            Lattitude/Longitude representation of the first point
        end : tuple of size 2 of ints/floats,
            Lattitude/Longitude representation of the second point
        """
        pass

    @abstractmethod
    def project(self, points):
        """
        Projects a point given in Lattitude/Longitude to a point in the
        three dimensional model.

        Parameter
        ---------
        point : tuple of size 2 of ints/floats,
            Lattitude/Longitude representation fo the point to be projected.
        """
        pass

    @abstractmethod
    def direct_path(self, start, end, res):
        """
        Computes points along the shortest path from `start` to `end`
        """
