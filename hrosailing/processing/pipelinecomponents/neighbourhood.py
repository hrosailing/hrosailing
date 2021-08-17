"""
Defines a baseclass for neighbourhoods used in the
processing.processing.PolarPipeline class,
taht can be used to create custom neighbourhoods for use.

Also contains various predefined and usable neighbourhoods
"""

# Author: Valentin F. Dannenberg / Ente


from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import ConvexHull


def scaled(norm, scal):
    scal = np.array(list(scal))

    def scaled_norm(vec):
        return norm(scal * vec)

    return scaled_norm


def euclidean_norm(vec):
    return np.linalg.norm(vec, axis=1)


class NeighbourhoodException(Exception):
    """Custom exception for errors that may appear whilst
    working with the Neighbourhood class and subclasses
    """

    pass


class Neighbourhood(ABC):
    """Base class for all neighbourhood classes


    Abstract Methods
    ----------------
    is_contained_in(self, pts)
    """

    @abstractmethod
    def is_contained_in(self, pts):
        pass


class Ball(Neighbourhood):
    """A class to describe a closed 2-dimensional ball
    centered around the origin, ie { x in R^2 : ||x|| <= r }

    Parameters
    ----------
    norm : function or callable, optional
        The norm for which the ball is described, ie ||.||

        If nothing is passed, it will default to a scaled version
        of ||.||_2

    radius : positive int or float, optional
        The radius of the ball, ie r

        Defaults to 1

    Raises a NeighbourhoodException if inputs are not
    of the specified types

    Methods
    -------
    is_contained_in(self, pts)
        Checks given points for membership.
    """

    def __init__(self, norm=None, radius=1):
        if norm is None:
            norm = scaled(euclidean_norm, [1 / 40, 1 / 360])

        if not isinstance(radius, (int, float)) or radius <= 0:
            raise NeighbourhoodException(
                "`radius` needs to be positive number"
            )
        if not callable(norm):
            raise NeighbourhoodException("`norm` is not callable")

        self._norm = norm
        self._radius = radius

    def __repr__(self):
        return f"Ball(norm={self._norm.__name__}, radius={self._radius})"

    def is_contained_in(self, pts):
        """Checks given points for membership.

        Parameters
        ----------
        pts : array_like of shape (n, 2)
            Points that will be checked for membership

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing which of the input points
            is a member of the neighbourhood


        Raises a NeighbourhoodException if the input is not
        of the specified type
        """
        pts = _sanity_check(pts)
        return self._norm(pts) <= self._radius


class ScalingBall(Neighbourhood):
    """A class to represent a closed 2-dimensional ball
    centered around the origin, ie { x in R^2 : ||x|| <= r },
    where the radius r will be dynamically determined, such that
    there are always a certain amount of given points contained
    in the ball

    Parameters
    ----------
    min_pts : positive int
        The minimal amount of certain given points that should be
        contained in the scaling ball

    max_pts : positive int
        The "maximal" amount of certain given points that should be
        contained in the scaling ball.

        Mostly used for initial guess of a "good" radius. Also to
        guarantee that on average, the scaling ball will contain
        (min_pts + max_pts) / 2 points of certain given points

        It is also unlikely that the scaling ball will contain
        more than max_pts points

    norm : function or callable, optional
        The norm for which the scaling ball is described, ie ||.||

        If nothing is passed, it will default to a scaled version
        of ||.||_2

    Raises a NeighbourhoodException
        - if inputs are not of the specified types
        - if max_pts is smaller or equal to min_pts


    Methods
    -------
    is_contained_in(self, pts)
        Checks given points for membership.
    """

    def __init__(self, min_pts, max_pts, norm=None):
        if norm is None:
            norm = scaled(euclidean_norm, (1 / 40, 1 / 360))

        if not isinstance(min_pts, int) or min_pts <= 0:
            raise NeighbourhoodException(
                "`min_pts` needs to be a positive integer"
            )
        if not isinstance(max_pts, int) or max_pts <= 0:
            raise NeighbourhoodException(
                "`max_pts` needs to be a positive integer"
            )
        if max_pts <= min_pts:
            raise NeighbourhoodException("`max_pts` is smaller than `min_pts`")
        if not callable(norm):
            raise NeighbourhoodException("`norm` is not callable")

        self._min_pts = min_pts
        self._max_pts = max_pts
        self._norm = norm
        self._avg = (min_pts + max_pts) / 2
        self._first_call = True

        self._n_pts = None
        self._area = None
        self._radius = None

    def __repr__(self):
        return f"ScalingBall(min_pts={self._min_pts}, max_pts={self._max_pts})"

    def is_contained_in(self, pts):
        """Checks given points for membership, and scales
        ball so that at least min_pts points are contained in it

        Parameters
        ----------
        pts : array_like of shape (n, 2)
            Points that will be checked for membership

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing which of the input points
            is a member of the neighbourhood


        Raises a NeighbourhoodException if the input is not
        of the specified type
        """
        if self._first_call:
            # check only in the first call
            pts = _sanity_check(pts)

            # initial guess for suitable radius.
            self._n_pts = pts.shape[0]
            self._area = ConvexHull(pts).volume
            self._radius = np.sqrt(
                self._avg * self._area / (np.pi, *self._n_pts)
            )

            self._first_call = False

        dist = self._norm(pts)
        mask = dist <= self._radius
        if self._min_pts <= len(pts[mask]):
            return mask

        # expand radius the smallest possible amount, such
        # that a new point will be in scaling ball
        self._radius = np.min(dist[np.logical_not(mask)])
        return self.is_contained_in(pts)


class Ellipsoid(Neighbourhood):
    """A class to represent a closed d-dimensional ellipsoid
    centered around the origin, ie T(B), where T is an invertible
    linear transformation, and B is a closed d-dimensional ball,
    centered around the origin.

    It will be represented using the equivalent formulation:
    { x in R^2 : ||T^-1 x|| <= r }

    Parameters
    ----------
    lin_trans: numpy.ndarray with shape (2,2), optional
        The linear transformation which transforms the
        ball into the given ellipsoid, ie T

        If nothing is passed, it will default to I_2, the 2x2
        unit matrix, ie the ellipsoid will be a ball

    norm : function or callable, optional
        The norm for which the ellipsoid is described, ie ||.||

        If nothing is passed, it will default to a scaled
        version of ||.||_2

    radius : positive int or float, optional
        The radius of the ellipsoid, ie r

        Defaults to 1

    Raises a NeighbourhoodException
        - if the inputs are not of the specified or
        functionally equivalent types
        - if lin_trans is not invertible


    Methods
    -------
    is_contained_in(self, pts)
        Checks given points for membership.
    """

    def __init__(self, lin_trans=None, norm=None, radius=1):
        if lin_trans is None:
            lin_trans = np.eye(2)
        lin_trans = np.asarray(lin_trans)
        if norm is None:
            norm = scaled(euclidean_norm, [1 / 40, 1 / 360])

        if not isinstance(radius, (int, float)) or radius <= 0:
            raise NeighbourhoodException(
                "`radius` needs to be positive number"
            )
        if lin_trans.shape != (2, 2):
            raise NeighbourhoodException(
                "`lin_trans` needs to be a square matrix of size 2"
            )
        if not np.linalg.det(lin_trans):
            raise NeighbourhoodException("`lin_trans` needs to be invertible")
        if not callable(norm):
            raise NeighbourhoodException("`norm` is not callable")

        self._T = np.linalg.inv(lin_trans)
        self._norm = norm
        self._radius = radius

    def __repr__(self):
        return (
            f"Ellipsoid(lin_trans={self._T}, "
            f"norm={self._norm.__name__}, radius={self._radius})"
        )

    def is_contained_in(self, pts):
        """Checks given points for membership.

        Parameters
         ----------
        pts : array_like of shape (n, 2)
            Points that will be checked for membership

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing which of the input points
            is a member of the neighbourhood


        Raises a NeighbourhoodException if the input is not
        of the specified type
        """
        pts = _sanity_check(pts)

        # transform the ellipsoid to a ball
        pts = (self._T @ pts.T).T
        return self._norm(pts) <= self._radius


class Cuboid(Neighbourhood):
    """A class to represent a d-dimensional closed cuboid, ie
    { x in R^2 : |x_i| <= b_i, i=1,2 }

    Parameters
    ----------
    norm : function or callable, optional
        The 1-d norm used to measure the length of the x_i, ie |.|

        If nothing is passed, it will default to the absolute value |.|

    dimensions: tuple of length 2, optional
        The 'length' of the 'sides' of the cuboid, ie the b_i

        If nothing is passed, it will default to (1,1)

    Raises a NeighbourhoodException if inputs are not of the
    specified or functionally equivalent types


    Methods
    -------
    is_contained_in(self, pts)
        Checks given points for membership.
    """

    def __init__(self, norm=None, dimensions=None):
        if dimensions is None:
            dimensions = (1, 1)
        if len(dimensions) != 2:
            raise NeighbourhoodException("`dimensions` is not of length 2")

        if norm is None:
            norm = np.abs
        if not callable(norm):
            raise NeighbourhoodException("`norm` is not callable")

        self._norm = norm
        self._size = dimensions

    def __repr__(self):
        return f"Cuboid(norm={self._norm.__name__}, dimensions={self._size})"

    def is_contained_in(self, pts):
        """Checks given points for membership.

        Parameters
         ----------
        pts : array_like of shape (n, 2)
            Points that will be checked for membership

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing which of the input points
            is a member of the neighbourhood


        Raises a NeighbourhoodException
            - if the input is not of the specified or
            functionally equivalent type
        """
        pts = _sanity_check(pts)
        mask = (
            np.ones((pts.shape[0],), dtype=bool)
            & (self._norm(pts[:, 0]) <= self._size[0])
            & (self._norm(pts[:, 1]) <= self._size[1])
        )
        return mask


# TODO Improve __init__
class Polytope(Neighbourhood):
    """A class to represent a general 2-dimensional polytope, ie the
    convex hull P = conv(x_1, ..., x_n) of some n points x_1 ,..., x_n
    or equivalent as the (bounded) intersection of m half spaces
    P = { x in R^2 : Ax <= b }

    Parameters
    ----------
    mat: array_like of shape (m, 2), optional
        matrix to represent the normal vectors a_i of the half
        spaces, ie A = (a_1, ... , a_m)^t

        If nothing is passed, it will default to (I_2, -I_2)^t,
        where I_d is the d-dimensional unit matrix

    b: array_like of shape (m, ), optional
        vector to represent the ... b_i of the half spaces, ie
        b = (b_1, ... , b_m)^t

        If nothing is passed, it will default to (1,...,1)

    Raises a NeighbourhoodException if inputs are not of the
    specified or functionally equivalent types

    Warning
    -------
    Does not check wether the polytope given by mat and b is a polytope,
    ie if P is actually bounded


    Methods
    -------
    is_contained_in(self, pts)
        Checks given points for membership.
    """

    def __init__(self, mat=None, b=None):
        if mat is None:
            mat = np.row_stack((np.eye(2), -np.eye(2)))
        if b is None:
            b = np.ones(4)

        try:
            mat = np.asarray_chkfinite(mat)
        except ValueError as ve:
            raise NeighbourhoodException(
                "`mat` should only have finite and non-NaN entries"
            ) from ve

        try:
            b = np.asarray_chkfinite(b)
        except ValueError as ve:
            raise NeighbourhoodException(
                "`b` should only have finite and non-NaN entries"
            ) from ve

        if mat.ndim != 2:
            raise NeighbourhoodException("`mat` is not 2-dimensional")
        if b.ndim != 1:
            raise NeighbourhoodException("`b` is not 1-dimensional")

        if mat.shape[0] != b.shape[0] or mat.shape[1] != 2:
            raise NeighbourhoodException(
                "`mat` needs to be a matrix of shape (n, 2) and "
                "`b` needs to be a vector of shape (n, )"
            )

        self._mat = mat
        self._b = b

    def __repr__(self):
        return f"Polytope(mat={self._mat}, b={self._b})"

    def is_contained_in(self, pts):
        """Checks given points for membership.

        Parameters
         ----------
        pts : array_like of shape (n, 2)
            Points that will be checked for membership

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing which of the input points
            is a member of the neighbourhood


        Raises a NeighbourhoodException if the input is not
        of the specified type
        """
        pts = _sanity_check(pts)
        mask = np.ones((pts.shape[0],), dtype=bool)
        for ineq, bound in zip(self._mat, self._b):
            mask = mask & (ineq @ pts.T <= bound)
        return mask


def _sanity_check(pts):
    pts = np.asarray(pts)

    if pts.shape[1] != 2 or pts.ndim != 2:
        raise NeighbourhoodException("`pts` has incorrect shape")

    return pts
