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
    Contains methods to translate between points given in latitude and
    longitude coordinates and points on a map or a three dimensional globe.

    Abstract Methods
    ----------------
    project(points)

    lat_lon(points)

    distance(start, end)

    shortest_projected_path(start, end, res)
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
    def shortest_projected_path(self, start, end, res=1000):
        """
        Computes the shortest path in the projected globe model.
        This is used in order to find the shortest path on the lattitude-
        longitude plane.

        start : np.ndarray of shape (2,n) or (3,n)
            The coordinates of the starting point on the globe

        end : np.ndarray of shape (2,n) or (3,n)
            The coordinates of the destination point on the globe

        res : int
            The number of points to be computed

            Defaults to 1000.
        """
        pass

    def shortest_path(self, start, end, res=1000):
        """
        Uses the `shortest_projected_path` method to compute points on the
        shortest path from `start` to `end`, all given
        in lattitude/longitude coordinates.

        Parameter
        ---------

        start : sequence of floats of length 2
            The lattitude/longitude coordinates of the starting position.

        end : sequence of floats of length 2
            The lattitude/longitude coordinates of the goal position.

        res : int
            The number of points to be computed

            Defaults to 1000.

        Returns
        ----------

        path: numpy.ndarray of shape (`res`, 2)
            The lattitude/longitude coordinates of points along the shortest
            path from `start` to `end`.

        """
        start, end = _ensure_2d(start, end)
        start, end = self.project(np.row_stack([start, end]))
        path = self.shortest_projected_path(start, end, res)
        return self.lat_lon(path)


class FlatMercatorProjection(GlobeModel):
    """
    Globe model that interprets coordinates via the mercator projection.

    Parameters
    ---------

    earth_radius: float/int
        The assumed radius of the (spherical) earth in nautical miles.

        Defaults to 3440.
    """

    def __init__(
            self, virt_northpole=(90, 0), virt_zero=(0, 0),
            earth_radius=3440
    ):
        pole = _on_ball(np.array(virt_northpole))
        zero = _on_ball(np.array(virt_zero))
        east = np.cross(pole, zero)
        self._transform_mat = np.row_stack([zero, east, pole]).transpose()
        self._inverse_transform_mat = np.linalg.inv(self._transform_mat)

        self._earth_radius = earth_radius

    def _transform_coordinates(self, pts, matrix):
        pts = _on_ball(pts)
        pts = pts.reshape((len(pts), 3, 1))
        pts = matrix@pts
        pts = pts.reshape(len(pts), 3)
        return _on_lat_lon(pts)

    def project(self, points):
        """
        Computes the mercator projection with reference point lat_mp of
        given points. Projection has size (n, 2).

        See also
        --------
        `GlobeModel.project`
        """
        points = np.array(points)
        points = self._transform_coordinates(points, self._transform_mat)
        lat, lon = points[:, 0], points[:, 1]

        return self._earth_radius*np.column_stack(
            [
                np.deg2rad(lon),
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
        long = np.rad2deg(x)
        latlon = np.column_stack([lat, long])
        return self._transform_coordinates(latlon, self._inverse_transform_mat)

    def distance(self, start, end):
        """
        See also
        ---------
        `GlobeModel.distance`
        """
        start, end = _ensure_2d(start, end)
        return np.linalg.norm(self.project(start) - self.project(end))

    def shortest_projected_path(self, start, end, res=1000):
        return np.linspace(start, end, res)


class SphericalGlobe(GlobeModel):
    """
    Globe model that interprets latitude-longitude coordinates on a spherical
    globe.

    Parameters
    -----------
    earth_radius: int/float, optional
        The radius of the assumed globe
    """

    def __init__(self, earth_radius=21600/2/np.pi):
        self._earth_radius = earth_radius

    def project(self, points):
        """
        See also
        ---------
        `GlobeModel.project`
        """
        return self._earth_radius*_on_ball(points)

    def lat_lon(self, points):
        """
        See also
        ---------
        `GlobeModel.lat_lon`
        """
        points = _ensure_2d(points)
        points = points/self._earth_radius
        return _on_lat_lon(points)

    def distance(self, start, end):
        """
        See also
        ---------
        `GlobeModel.distance`
        """

        start, end = _on_ball(np.row_stack([start, end]))
        angle = _angle(start, end)
        return self._earth_radius*angle

    def shortest_projected_path(self, start, end, res=1000):
        """
        See also
        ---------
        `GlobeModel.shortest_projected_path`
        """
        start, end = _ensure_2d(start, end)
        angle = _angle(start, end)

        angles = np.linspace(0, angle, res)

        # (1,0) -> (1, 0) -> (1,0,0) -> start, (cos(angle), sin(angle)) -> (0,1) -> (0,1,0) -> end
        proj1 = np.linalg.inv(np.array([[1, np.cos(angle)], [0, np.sin(angle)]]))
        proj2 = np.transpose(np.array([[1, 0, 0], [0, 1, 0]]))
        proj3 = np.transpose(np.row_stack([start, end, np.array([0, 0, 1])]))
        proj = proj3@proj2@proj1

        # # start -> (1,0,0) -> (1, 0) -> (1, 0), end -> (0,1,0) -> (0,1) -> (cos(angle), sin(angle))
        # inv_proj1 = np.linalg.inv(np.transpose(np.row_stack([start, end, np.array([0, 0, 1])])))
        # inv_proj2 = np.transpose(np.array([[1, 0], [0, 1], [0, 0]]))
        # inv_proj = np.linalg.inv(proj2)@inv_proj2@inv_proj1

        flat_pts = np.array([[[np.cos(ang)], [np.sin(ang)]] for ang in angles])
        return (proj@flat_pts).reshape((len(flat_pts), 3))


def _ensure_2d(*args):
    if len(args) == 1:
        return np.atleast_2d(args[0])
    return (np.atleast_2d(pt) for pt in args)


def _on_ball(pts):
    pts = np.atleast_2d(np.deg2rad(pts))
    lat, lon = pts[:, 0], pts[:, 1]
    return np.column_stack([
        np.cos(lat)*np.cos(lon),
        np.cos(lat)*np.sin(lon),
        np.sin(lat)
    ])


def _on_lat_lon(pts):
    x, y, z = pts.transpose()
    latlon = np.column_stack([
        np.arcsin(z),
        np.arctan2(y, x)
    ])
    return np.rad2deg(latlon)


def _angle(pt1, pt2):
    """Computes angles in radians between points"""
    pt1 = pt1.ravel()
    pt2 = pt2.ravel()
    return np.arccos(np.dot(pt1, pt2)/np.linalg.norm(pt1)/np.linalg.norm(pt2))