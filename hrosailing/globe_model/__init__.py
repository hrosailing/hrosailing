"""
Contains a base class and several concrete implementations of globe models
which can be used to perform calculations on a three dimensional model of the
globe using lattitude/longitude coordinates.
"""

from abc import ABC, abstractmethod
import numpy as np


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

    @abstractmethod
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


class FlatMercatorProjection(GlobeModel):
    """
    Globe model that interprets coordinates via the mercator projection.

    Parameters
    ---------

    mp: float, int or tuple of length 2
        One of the following:

        - The longitude of the midpoint used for the mercator projection.
        - The midpoint used for the mercator projection.

    earth_radius: float/int
        The assumed radius of the (spherical) earth in nautical miles.

        Defaults to 3440.
    """

    def __init__(self, mp, earth_radius=3440):
        if isinstance(mp, (int, float)):
            self._lon_mp = mp
        if isinstance(mp, tuple) and len(mp) == 2:
            self._lon_mp = mp[1]
        self._earth_radius = earth_radius

    def project(self, points):
        """
        Computes the mercator projection with reference point lat_mp of
        given points. Projection has size (n, 2).

        See also
        --------
        `GlobeModel.project`
        """
        points = np.array(points)
        lat, lon = points[:, 0], points[:, 1]

        return self._earth_radius*np.column_stack(
            [
                np.deg2rad(lon - self._lon_mp),
                np.arcsinh(np.tan(np.deg2rad(lat)))
                #np.log(np.tan(np.pi/4 + np.deg2rad(lat)/2))
            ]
        )

    def lat_lon(self, points):
        """
        See also
        ----------
        `GlobeModel.lat_lon`
        """
        points = np.array(points)/self._earth_radius
        x, y = points[:, 0], points[:, 1]
        lat = np.rad2deg(np.arcsin(np.tanh(y)))
        long = np.rad2deg(x) + self._lon_mp
        return np.column_stack([lat, long])

    def distance(self, start, end):
        pass

    def shortest_path(self, start, end, res=1000):
        pass