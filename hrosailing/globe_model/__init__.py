"""
Contains a base class and several concrete implementations of globe models
which can be used to perform calculations on a three dimensional model of the
globe using lattitude/longitude coordinates.
"""

from abc import ABC, abstractmethod


class GlobeModel(ABC):
    """
    Abstract base class of globe models.

    Abstract Methods
    ----------------
    project(points)

    lat_lon(points)

    distance(start, end)

    shortest_path(start, end, res)
    """

    @abstractmethod
    def project(self, points):
        """
        Calculates the three dimensional points corresponding to a given
        of points in lattitude/longitude coordinates.

        Parameter
        ----------

        points: numpy.ndarray of shape (n, 2) with dtype float
            The points to project on the globe given in lattitude/longitude
            coordinates.

        Returns
        --------

        projection: numpy.ndarray of shape (n,3) with dtype float
            The points on the globe.
        """
        pass