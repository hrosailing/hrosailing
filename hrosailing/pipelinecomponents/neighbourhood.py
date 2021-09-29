"""
Classes used to model various geometric shapes centered around
the origin

Defines the Neighbourhood Abstract Base Class that can be used
to create custom geometric shapes

Subclasses of Neighbourhood can be used with the TableExtension and
the PointcloudExtension class in the hrosailing.pipeline module
"""

# Author: Valentin Dannenberg


from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy.spatial import ConvexHull

from ._utils import scaled_euclidean_norm


class NeighbourhoodInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of a Neighbourhood
    """


class Neighbourhood(ABC):
    """Base class for all neighbourhood classes


    Abstract Methods
    ----------------
    is_contained_in(self, pts)
    """

    @abstractmethod
    def is_contained_in(self, pts):
        """This method should be used, given certain points, to
        determine which of these points lie in the neighbourhood
        and which do not, by producing a boolean array of the same
        size as pts
        """


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

        Defaults to 0.05


    Raises a NeighbourhoodInitializationException if radius is nonpositive
    """

    def __init__(
        self,
        norm: Callable = scaled_euclidean_norm,
        radius=0.05,
    ):
        if radius <= 0:
            raise NeighbourhoodInitializationException(
                "`radius` is not positive"
            )

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
        """
        pts = np.asarray(pts)
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


    Raises a NeighbourhoodInitializationException

     - if min_pts or max_pts are nonpositive
     - if max_pts is less than or equal to min_pts
    """

    def __init__(
        self,
        min_pts,
        max_pts,
        norm: Callable = scaled_euclidean_norm,
    ):

        if min_pts <= 0:
            raise NeighbourhoodInitializationException(
                "`min_pts` is not positive"
            )
        if max_pts <= 0:
            raise NeighbourhoodInitializationException(
                "`max_pts` is not positive"
            )
        if max_pts <= min_pts:
            raise NeighbourhoodInitializationException(
                "`max_pts` is smaller than `min_pts`"
            )

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
        """
        if self._first_call:
            pts = np.asarray(pts)

            # initial guess for suitable radius.
            self._n_pts = pts.shape[0]
            self._area = ConvexHull(pts).volume
            self._radius = np.sqrt(
                self._avg * self._area / (np.pi * self._n_pts)
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
    lin_trans: array_like of shape (2,2), optional
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

        Defaults to 0.05


    Raises a NeighbourhoodInitializationException

    - if radius is nonpositive
    - if lin_trans is not a (2,2)-array or is not invertible
    """

    def __init__(
        self,
        lin_trans=None,
        norm: Callable = scaled_euclidean_norm,
        radius=0.05,
    ):
        if lin_trans is None:
            lin_trans = np.eye(2)

        lin_trans = np.asarray_chkfinite(lin_trans)

        if lin_trans.shape != (2, 2):
            raise NeighbourhoodInitializationException(
                "`lin_trans` has incorrect shape"
            )

        if not np.linalg.det(lin_trans):
            raise NeighbourhoodInitializationException(
                "`lin_trans` is singular"
            )

        if radius <= 0:
            raise NeighbourhoodInitializationException(
                "`radius` is not positive"
            )

        # invert lin_trans in initialization to later
        # transform ellipsoid to a ball
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
        """
        pts = np.asarray(pts)

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

    dimensions: subscriptable of length 2, optional
        The 'length' of the 'sides' of the cuboid, ie the b_i

        If nothing is passed, it will default to (0.05, 0.05)

    """

    def __init__(
        self,
        norm: Callable = scaled_euclidean_norm,
        dimensions=(0.05, 0.05),
    ):
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
        """
        mask = (
            np.ones((pts.shape[0],), dtype=bool)
            & (self._norm(pts[:, 0]) <= self._size[0])
            & (self._norm(pts[:, 1]) <= self._size[1])
        )
        return mask


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

        If nothing is passed, it will default to (0.05,...,0.05)


    Raises a NeighbourhoodException if mat and b are not of matching shape


    Warning
    -------
    Does not check wether the polytope given by mat and b is a polytope,
    ie if P is actually bounded
    """

    def __init__(
        self, mat=np.row_stack((np.eye(2), -np.eye(2))), b=0.05 * np.ones(4)
    ):
        # NaN's or infinite values can't be handled
        mat = np.asarray_chkfinite(mat)
        b = np.asarray_chkfinite(b)

        if mat.ndim != 2 or mat.shape[1] != 2:
            raise NeighbourhoodInitializationException(
                "`mat` has incorrect shape"
            )

        if b.ndim != 1 or b.shape[0] != mat.shape[0]:
            raise NeighbourhoodInitializationException(
                "`b` has incorrect shape"
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
        """
        pts = np.asarray(pts)

        mask = np.ones((pts.shape[0],), dtype=bool)
        for ineq, bound in zip(self._mat, self._b):
            mask = mask & (ineq @ pts.T <= bound)
        return mask
