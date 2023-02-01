"""
Classes used to model various geometric shapes centered around
the geometric origin.

Defines the `Neighbourhood` abstract base class that can be used
to create custom geometric shapes.

Subclasses of `Neighbourhood` can be used with the `TableExtension` and
the `PointcloudExtension` classes in the `hrosailing.pipeline` module.
"""


from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ._utils import scaled_euclidean_norm


class NeighbourhoodInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of a `Neighbourhood`.
    """


class Neighbourhood(ABC):
    """Base class for all neighbourhood classes.


    Abstract Methods
    ----------------
    is_contained_in(self, pts)
    """

    @abstractmethod
    def is_contained_in(self, pts):
        """This method should be used, given certain points, to
        determine which of these points lie in the neighbourhood
        and which do not, by producing a boolean array of the same
        size as `pts`.

        Parameters
        ----------
        pts : array_like of shape (n, 2)
            Points that will be checked for membership.

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing which of the input points
            are members of the neighbourhood.
        """


class Ball(Neighbourhood):
    """A class to describe a closed 2-dimensional ball
    centered around the origin, i.e. :math:`\\\\{x \\in R^2 : ||x|| <= r\\\\}`.

    Parameters
    ----------
    radius : positive int or float, optional
        The radius of the ball, i.e. :math:`r`.

        Defaults to `0.05`.

    norm : function or callable, optional
        The norm for which the ball is described, i.e. :math:`||.||`.

        If nothing is passed, it will default to a scaled version
        of :math:`||.||_2` in two dimensions.

    Raises
    ------
    NeighbourhoodInitializationException
        If `radius` is non-positive.
    """

    def __init__(
        self,
        radius=0.05,
        norm: Callable = scaled_euclidean_norm,
    ):
        if radius <= 0:
            raise NeighbourhoodInitializationException(
                "`radius` is non-positive"
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
            Points that will be checked for membership.

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing which of the input points
            are members of the neighbourhood.
        """
        pts = np.asarray(pts)
        try:
            return self._norm(pts) <= self._radius
        except ValueError:
            return np.asarray([self._norm(pt) <= self._radius for pt in pts])


class ScalingBall(Neighbourhood):
    """A class to represent a closed 2-dimensional ball
    centered around the origin, i.e. :math:`\\\\{x \\in R^2 : ||x|| <= r\\\\}`,
    where the radius :math:`r` will be dynamically determined, such that
    there is always a certain amount of given points contained
    in the ball.

    Parameters
    ----------
    min_pts : positive int
        The minimal amount of certain given points that should be
        contained in the scaling ball.

    norm : function or callable, optional
        The norm for which the scaling ball is described, i.e. :math:`||.||`.

        If nothing is passed, it will default to a scaled version
        of :math:`||.||_2`.

    Raises
    ------
    NeighbourhoodInitializationException
        If `min_pts` is non-positive.

    """

    def __init__(
        self,
        min_pts,
        norm: Callable = scaled_euclidean_norm,
    ):
        if min_pts <= 0:
            raise NeighbourhoodInitializationException(
                "`min_pts` is non-positive"
            )

        self._min_pts = min_pts
        self._norm = norm

        self._n_pts = None
        self._area = None
        self._radius = None

    def __repr__(self):
        return (
            f"ScalingBall(min_pts={self._min_pts}, norm={self._norm.__name__})"
        )

    def is_contained_in(self, pts):
        """Checks given points for membership, and scales
        ball so that at least `min_pts` points are contained in it.

        Parameters
        ----------
        pts : array_like of shape (n, 2)
            Points that will be checked for membership.

        Returns
        -------
        points_in_ball : boolean numpy.ndarray of shape (n, )
            Boolean array describing which of the input points
            are members of the neighbourhood.
        """
        pts = np.asarray(pts)

        dist = self._norm(pts)
        self._radius = sorted(dist)[self._min_pts - 1]

        return dist <= self._radius


class Ellipsoid(Neighbourhood):
    """A class to represent a closed d-dimensional ellipsoid
    centered around the origin, i.e. :math:`T(B)`, where :math:`T` is an invertible
    linear transformation, and :math:`B` is a closed d-dimensional ball,
    centered around the origin.

    It will be represented using the equivalent formulation:
    :math:`\\\\{x \\in R^2 : ||T^{-1} x|| \\leq r\\\\}`.

    Parameters
    ----------
    lin_trans : array_like of shape (2,2), optional
        The linear transformation which transforms the
        ball into the given ellipsoid, i.e. :math:`T`.

        If nothing is passed, it will default to the 2x2
        identity matrix, i.e. the ellipsoid will be a ball.

    norm : function or callable, optional
        The norm for which the ellipsoid is described, i.e. :math:`||.||`.

        If nothing is passed, it will default to a scaled
        version of :math:`||.||_2`.

    radius : positive int or float, optional
        The radius of the ellipsoid, i.e. :math:`r`.

        Defaults to `0.05`.


    Raises
    ------
    NeighbourhoodInitializationException
        If `radius` is non-positive.
     NeighbourhoodInitializationException
        If `lin_trans` is not a (2,2)-array or is not invertible.
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
                "`radius` is non-positive"
            )

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
            Points that will be checked for membership.

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing which of the input points
            are members of the neighbourhood.
        """
        pts = np.asarray(pts)

        pts = self._transform_ellipsoid_to_ball(pts)

        return self._norm(pts) <= self._radius

    def _transform_ellipsoid_to_ball(self, pts):
        return (self._T @ pts.T).T


class Cuboid(Neighbourhood):
    """A class to represent a d-dimensional closed cuboid, i.e.
    :math:`\\\\{x \\in R^2 : |x_i| \\leq b_i, i=1,2\\\\}`.

    Parameters
    ----------
    norm : function or callable, optional
        The 1-d norm used to measure the length of :math:`x_i`, i.e. :math:`|.|`.

        If nothing is passed, it will default to the absolute value :math:`|.|`.

    dimensions : subscriptable of length 2, optional
        The length of the sides of the cuboid, i.e. :math:`b_i`.

        If nothing is passed, it will default to `(0.05, 0.05)`.

    """

    def __init__(
        self,
        norm: Callable = np.abs,
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
            Points that will be checked for membership.

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing which of the input points
            are members of the neighbourhood.
        """
        pts = np.asarray(pts)

        mask = (
            np.ones((pts.shape[0],), dtype=bool)
            & (self._norm(pts[:, 0]) <= self._size[0])
            & (self._norm(pts[:, 1]) <= self._size[1])
        )
        return mask


class Polytope(Neighbourhood):
    """A class to represent a general 2-dimensional polytope, i.e. the
    convex hull :math:`P = conv(x_1, ..., x_n)` of some n points :math:`x_1 ,..., x_n`
    or equivalently the (bounded) intersection of m half spaces
    :math:`P = \\\\{x \\in R^2 : Ax \\leq b\\\\}`.

    Parameters
    ----------
    mat : array_like of shape (m, 2), optional
        Matrix to represent the normal vectors :math:`a_i` of the half
        spaces, i.e. :math:`A = (a_1, ... , a_m)^t`.

        If nothing is passed, it will default to :math:`(I_2, -I_2)^t`,
        where :math:`I_d` is the d-dimensional identity matrix.

    b : array_like of shape (m, ), optional
        Vector representing the :math:`b` of the half spaces, i.e.
        :math:`b = (b_1, ... , b_m)^t`.

        If nothing is passed, it will default to `(0.05,...,0.05)`.


    Raises
    ------
    NeighbourhoodInitializationException
        If `mat` and `b` are not of matching shape.


    Warning
    -------
    Does not check whether the polytope given by `mat` and `b` is a polytope,
    i.e. if :math:`P` is actually bounded.
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
            Points that will be checked for membership.

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing which of the input points
            are members of the neighbourhood.
        """
        pts = np.asarray(pts)

        mask = np.ones((pts.shape[0],), dtype=bool)
        for ineq, bound in zip(self._mat, self._b):
            mask = mask & (ineq @ pts.T <= bound)
        return mask
