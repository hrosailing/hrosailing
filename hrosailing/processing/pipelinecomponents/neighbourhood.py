"""
Defines a baseclass for neighbourhoods used in the
processing.processing.PolarPipeline class,
taht can be used to create custom neighbourhoods for use.

Also contains various predefined and usable neighbourhoods
"""

# Author: Valentin F. Dannenberg / Ente


import numpy as np

from abc import ABC, abstractmethod
from scipy.spatial import ConvexHull

from hrosailing.utils import euclidean_norm


class Neighbourhood(ABC):
    """Base class for all
    neighbourhood classes

    Abstract Methods
    ----------------
    is_contained_in(self, pts)
    """

    @abstractmethod
    def is_contained_in(self, pts):
        pass


class Ball(Neighbourhood):
    """A class to describe
    a closed d-dimensional
    ball centered around the
    origin, ie

    { x in R^d : || x || <= r }

    Parameters
    ----------
    d : positive int, optional
        The dimension of
        the ball

        Defaults to 2
    norm : function or callable, optional
        The norm for which the
        ball is described, ie
        ||x||

        If nothing is passed,
        it will default to
        ||.||_2
    radius : positive int or float, optional
        The radius of the ball,
        ie r

        Defaults to 1

    Methods
    -------
    is_contained_in(self, pts)
        Checks given points
        for membership.
    """

    def __init__(self, d=2, norm=None, radius=1):
        if norm is None:
            norm = euclidean_norm

        # Sanity checks
        if not isinstance(d, int) or d <= 0:
            raise ValueError(
                f"The dimension needs to "
                f"be a positive integer, "
                f"but {d} was passed"
            )
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError(
                f"The radius needs to be "
                f"positive number, but "
                f"{radius} was passed"
            )
        if not callable(norm):
            raise ValueError(f"{norm.__name__} is not callable")

        self._dim = d
        self._norm = norm
        self._radius = radius

    def __repr__(self):
        return (
            f"Ball(d={self._dim}, "
            f"norm={self._norm.__name__}, "
            f"radius={self._radius})"
        )

    def is_contained_in(self, pts):
        """Checks given points
        for membership.

        Parameters
        ----------
        pts : array_like of shape (n, d)
            Points that will be
            checked for membership,
            given as a sequence of
            points consisting of
            wind speed, wind angle
            and boat speed
        Returns
        -------

        mask : numpy.ndarray of shape (n, )
            Boolean array describing
            which of the input points
            is a member of the
            neighbourhood
        """

        pts = np.asarray(pts)
        shape = pts.shape
        if not pts.size:
            raise ValueError("No points were passed")
        if len(shape) != 2:
            raise ValueError(
                f"pts needs to be a "
                f"2-dimensional array, "
                f"but {pts} was passed"
            )
        if shape[1] != self._dim:
            raise ValueError(f"Points are not of dimension {self._dim}")

        return self._norm(pts) <= self._radius


# TODO Add dimension
class ScalingBall(Neighbourhood):
    def __init__(self, min_pts, max_pts, norm=None):
        if norm is None:
            norm = euclidean_norm
        if not callable(norm):
            raise ValueError(f"{norm.__name__} is not callable")

        self._min_pts = min_pts
        self._max_pts = max_pts
        self._norm = norm
        self._n_pts = None
        self._area = None
        self._avg = (min_pts + max_pts) / 2

    def __repr__(self):
        return (
            f"ScalingBall("
            f"min_pts={self._min_pts}, "
            f"max_pts={self._max_pts})"
        )

    def is_contained_in(self, pts, r=None):
        pts = np.asarray(pts)
        shape = pts.shape
        if not pts.size:
            raise ValueError("No points were passed")
        if len(shape) != 2:
            raise ValueError(
                f"pts needs to be a "
                f"2-dimensional array, "
                f"but {pts} was passed"
            )
        if shape[1] != 2:
            raise ValueError(f"Points are not of dimension 2")

        if self._n_pts is None:
            self._n_pts = shape[0]
        if self._area is None:
            self._area = ConvexHull(pts).volume
        if r is None:
            r = np.sqrt(self._avg * self._area / (np.pi, *self._n_pts))

        dist = self._norm(pts)
        mask = dist <= r
        if self._min_pts <= len(pts[mask]):
            return mask

        r = np.min(dist[np.logical_not(mask)])
        return self.is_contained_in(pts, r)


class Ellipsoid(Neighbourhood):
    """A class to represent
    a closed d-dimensional
    ellipsoid centered around
    the origin, ie
    T(B) where T is an invertible
    linear transformation, and B
    is a closed d-dimensional ball,
    centered around the origin.

    It will be represented using the
    equivalent formulation:
    { x in R^d : ||T^-1 x|| <= r }

    Parameters
    ----------
    d : positive int, optional
        The dimension of
        the ellipsoid

        Defaults to 2
    lin_trans: numpy.ndarray with shape (d,d), optional
        The invertible linear transformation
        which transforms the ball into
        the given ellipsoid, ie T

        lin_trans needs to have a
        non-zero determinant.

        If nothins is passed, it will
        default to I_d, the dxd unit matrix,
        ie the ellipsoid will be a ball
    norm : function or callable, optional
        The norm for which the
        ellipsoid is described, ie
        ||x||

        If nothing is passed,
        it will default to
        ||.||_2
    radius : positive int or float, optional
        The radius of the ellipsoid,
        ie r

        Defaults to 1

    Methods
    -------
    is_contained_in(self, pts)
        Checks given points
        for membership.
    """

    def __init__(self, d=2, lin_trans=None, norm=None, radius=1):

        if not isinstance(d, int) or d <= 0:
            raise ValueError(
                f"The dimension needs to "
                f"be a positive integer, "
                f"but {d} was passed"
            )

        if lin_trans is None:
            lin_trans = np.eye(d)
        if norm is None:
            norm = euclidean_norm

        lin_trans = np.asarray(lin_trans)
        # Sanity checks
        if not lin_trans.size:
            raise ValueError("lin_trans is an empty array")
        if lin_trans.shape != (d, d):
            raise ValueError(f"lin_trans is not a square matrix of size {d}")
        if not np.linalg.det(lin_trans):
            raise ValueError(f"{lin_trans} is not invertible")

        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError(
                f"The radius needs to be "
                f"positive number, but "
                f"{radius} was passed"
            )
        if not callable(norm):
            raise ValueError(f"{norm.__name__} is not callable")

        # Transform the ellipsoid to a ball
        lin_trans = np.linalg.inv(lin_trans)

        self._dim = d
        self._T = lin_trans
        self._norm = norm
        self._radius = radius

    def __repr__(self):
        return (
            f"Ellipsoid(d={self._dim}, "
            f"lin_trans={self._T}, "
            f"norm={self._norm.__name__}, "
            f"radius={self._radius})"
        )

    def is_contained_in(self, pts):
        """Checks given points
        for membership.

        Parameters
         ----------
        pts : array_like of shape (n, d)
            Points that will be
            checked for membership

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing
            which of the input points
            is a member of the
            neighbourhood
        """

        pts = np.asarray(pts)
        shape = pts.shape
        if not pts.size:
            raise ValueError("No points were passed")
        if len(shape) != 2:
            raise ValueError(
                f"pts needs to be a "
                f"2-dimensional array, "
                f"but {pts} was passed"
            )
        if shape[1] != self._dim:
            raise ValueError(f"Points are not of dimension {self._dim}")

        pts = (self._T @ pts.T).T
        print(pts)

        return self._norm(pts) <= self._radius


class Cuboid(Neighbourhood):
    """A class to represent
    a d-dimensional closed cuboid, ie

    { x in R^d : |x_i| <= b_i, i=1,..,d }

    Parameters
    ----------
    d : positive int, optional
        The dimension of
        the cuboid

        Defaults to 2
    norm : function or callable, optional
        The 1-d norm used to
        measure the length of
        the x_i, ie |.|

        If nothing is passed,
        it will default to
        the absolute value |.|
    dimensions: tuple of length d, optional
        The 'length' of the 'sides'
        of the cuboid, ie the b_i

        If nothing is passed,
        it will default to (1,...,1)

    Methods
    -------
    is_contained_in(self, pts)
        Checks given points
        for membership.
    """

    def __init__(self, d=2, norm=None, dimensions=None):
        if not isinstance(d, int) or d <= 0:
            raise ValueError(
                f"The dimension needs to "
                f"be a positive integer, "
                f"but {d} was passed"
            )

        if norm is None:
            norm = np.abs
        if dimensions is None:
            dimensions = tuple(1 for _ in range(d))

        if not callable(norm):
            raise ValueError(f"{norm.__name__} is not callable")
        if len(dimensions) != d:
            raise ValueError(f"{dimensions} is not of length {d}")

        self._dim = d
        self._norm = norm
        self._size = dimensions

    def __repr__(self):
        return (
            f"Cuboid(d={self._dim}, "
            f"norm={self._norm.__name__}, "
            f"dimensions={self._size})"
        )

    def is_contained_in(self, pts):
        """
        Checks given points
        for membership.

        Parameters
         ----------
        pts : array_like of shape (n, d)
            Points that will be
            checked for membership

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing
            which of the input points
            is a member of the
            neighbourhood
        """

        pts = np.asarray(pts)
        shape = pts.shape
        d = self._dim
        if not pts.size:
            raise ValueError("No points were passed")
        if len(shape) != 2:
            raise ValueError(
                f"pts needs to be a "
                f"2-dimensional array, "
                f"but {pts} was passed"
            )
        if shape[1] != d:
            raise ValueError(f"Points are not of dimension {d}")

        dimensions = self._size
        mask = np.ones((shape[0],), dtype=bool)
        for i in range(d):
            mask = mask & (self._norm(pts[:, i]) <= dimensions[i])
        return mask


class Polytope(Neighbourhood):
    """A class to represent
    a general d-dimensional
    polytope, ie the convex
    hull of some n points

    P = conv(x_1, ..., x_n)

    or equivalent as the (bounded)
    intersection of m half spaces:

    P = { x in R^d : Ax <= b }

    Parameters
    ----------
    d : positive int, optional
        The dimension of
        the polytope

        Defaults to 2
    mat: array_like of shape (m, d), optional
        matrix to represent the
        normal vectors a_i of the half
        spaces, ie A = (a_1, ... , a_m)^t

        If nothing is passed,
        it will default to
        (I_d, -I_d)^t, where I_d
        is the d-dimensional unit
        matrix

    b: array_like of shape (m, ), optional
        vector to represent the ...
        b_i of the half spaces, ie
        b = (b_1, ... , b_m)^t

        If nothing is passed,
        it will default to
        e, where e is the One-vector
        of length 2 * d

    Warning
    -------
    Does not check wether the
    polytope given by mat
    and b is a polytope, ie
    if P is actually bounded

    Methods
    -------
    is_contained_in(self, pts)
        Checks given points
        for membership.
    """

    def __init__(self, d=2, mat=None, b=None):
        if not isinstance(d, int) or d <= 0:
            raise ValueError(
                f"The dimension needs to "
                f"be a positive integer, "
                f"but {d} was passed"
            )

        if mat is None:
            mat = np.row_stack((np.eye(d), -np.eye(d)))
        if b is None:
            b = np.ones(2 * d)

        mat = np.asarray(mat)
        b = np.asarray(b)
        shape_mat = mat.shape
        shape_b = b.shape
        # Sanity checks
        if not mat.size:
            # TODO: Revert to default
            #       value instead of
            #       raising exception?
            raise ValueError("mat is an empty array")
        if len(shape_mat) != 2:
            # TODO: Try broadcast!
            raise ValueError(f"mat is not 2-dimensional")
        if not b.size:
            # TODO: Revert to default
            #       value instead of
            #       raising exception?
            raise ValueError("b is an empty vector")
        if len(shape_b) != 1:
            # TODO: Try broadcast!
            raise ValueError("b is not 1-dimensional")
        if shape_mat[0] != shape_b[0]:
            # TODO: Try broadcast!
            raise ValueError(
                f"mat needs to be a matrix "
                f"of shape (n, d) and "
                f"b needs to be a vector "
                f"of shape (n, ), but "
                f"they are of shape "
                f"{shape_mat} and {shape_b}, "
                f"respectively"
            )
        if shape_mat[1] != d:
            raise ValueError(
                f"mat needs to be a matrix "
                f"of shape (n, {d}), but "
                f"is a matrix of shape "
                f"{shape_mat}"
            )

        self._dim = d
        self._mat = mat
        self._b = b

    def __repr__(self):
        return f"Polytope(d={self._dim}, mat={self._mat}, b={self._b})"

    def is_contained_in(self, pts):
        """
        Checks given points
        for membership.

        Parameters
         ----------
        pts : array_like of shape (n, d)
            Points that will be
            checked for membership

        Returns
        -------
        mask : numpy.ndarray of shape (n, )
            Boolean array describing
            which of the input points
            is a member of the
            neighbourhood
        """

        pts = np.asarray(pts)
        shape = pts.shape
        d = self._dim
        if not pts.size:
            raise ValueError("No points were passed")
        if len(shape) != 2:
            raise ValueError(
                f"pts needs to be a "
                f"2-dimensional array, "
                f"but {pts} was passed"
            )
        if shape[1] != d:
            raise ValueError(f"Points are not of dimension {d}")

        mat = self._mat
        b = self._b
        mask = np.ones((shape[0],), dtype=bool)
        for ineq, bound in zip(mat, b):
            mask = mask & (ineq @ pts.T <= bound)

        return mask
