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

        projection: numpy.ndarray of shape (n,2) or (n,3) with dtype float
            The points on the globe. The exact shape depends on the model.
        """
        pass

    @abstractmethod
    def lat_lon(self, points):
        """
        Computes the lattitude/longitude representations of given points on the
        globe.

        Parameter
        ---------

        points: numpy.ndarray of shape (n,2) or (n,3) with dtype float
            The points on the globe. The exact shape depends on the model.

        Returns
        ---------

        lat_lons: numpy.ndarray of shape (n,2) with dtype float
            The lattitude/longitude coordinates of the given points.
        """
        pass

    @abstractmethod
    def distance(self, start, end):
        """
        Computes the distance from `start` to `end` on the globe.

        Parameter
        ---------

        start: sequence of floats of length 2
            The lattitude/longitude coordinates of the starting position.

        end: sequence of floats of length 2
            The lattitude/longitude coordinates of the goal position.

        Returns
        -------

        distance: float
            The distance of the projections of the points `start` and `end`
            in the globe model.
        """
        pass

    def shortest_path(self, start, end, res=1000):
        """
        Computes points on the shortest path from `start` to `end`, all given
        in lattitude/longitude coordinates.

        Parameter
        ---------

        start: sequence of floats of length 2
            The lattitude/longitude coordinates of the starting position.

        end: sequence of floats of length 2
            The lattitude/longitude coordinates of the goal position.

        res: int
            The number of points to be computed

            Defaults to 1000.

        Returns
        ----------

        path: numpy.ndarray of shape (`res`, 2)
            The lattitude/longitude coordinates of points along the shortest
            path from `start` to `end`.

        """
        pass