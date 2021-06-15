"""

"""

# Author: Valentin F. Dannenberg / Ente


from abc import ABC, abstractmethod
import numpy as np

from exceptions import ProcessingException
from utils import euclidean_norm


class Neighbourhood(ABC):

    @abstractmethod
    def is_contained_in(self, vec):
        pass


class Ball(Neighbourhood):

    def __init__(self, d=2, norm=None, radius=1):
        if norm is None:
            norm = euclidean_norm

        if radius < 0:
            raise ProcessingException("")

        self._dim = d
        self._norm = norm
        self._r = radius

    def __repr__(self):
        return f"Ball(d={self.dimension}, norm={self.norm.__name__}," \
               f"radius={self.radius})"

    @property
    def dimension(self):
        return self._dim

    @property
    def norm(self):
        return self._norm

    @property
    def radius(self):
        return self._r

    def is_contained_in(self, vec):
        vec = np.asarray(vec)
        if len(vec[0]) != self.dimension:
            raise ProcessingException("")

        return self.norm(vec) <= self.radius


class Ellipsoid(Neighbourhood):

    def __init__(self, d=2, lin_trans=None,
                 norm=None, radius=1,):
        if lin_trans is None:
            lin_trans = np.eye(d)
        if norm is None:
            norm = euclidean_norm

        # Sanity checks
        if not lin_trans.size:
            raise ProcessingException("")
        if lin_trans.shape != (d, d):
            raise ProcessingException("")
        if not np.linalg.det(lin_trans):
            raise ProcessingException("")
        if radius < 0:
            raise ProcessingException("")

        # Transform the ellipsoid to a ball
        lin_trans = np.linalg.inv(lin_trans)

        self._dim = d
        self._lin_trans = lin_trans
        self._norm = norm
        self._r = radius

    def __repr__(self):
        return f"Ellipsiod(d={self.dimension}, " \
               f"lin_trans={self.linear_transformation}," \
               f"norm={self.norm.__name__}," \
               f"radius={self.radius})"

    @property
    def dimension(self):
        return self._dim

    @property
    def linear_transformation(self):
        return self._lin_trans

    @property
    def norm(self):
        return self.norm

    @property
    def radius(self):
        return self._r

    def is_contained_in(self, vec):
        vec = np.asarray(vec)
        if len(vec[0]) != self.dimension:
            raise ProcessingException("")

        for i, v in enumerate(vec):
            vec[i] = self.linear_transformation @ v.T

        return self.norm(vec) <= self.radius


class Cuboid(Neighbourhood):

    def __init__(self, d=2, norm=None, dimensions=(1, 1, 1)):
        if norm is None:
            norm = np.abs

        self._dim = d
        self._norm = norm
        self._size = dimensions

    def __repr__(self):
        return f"Cuboid(d={self.dimension}, norm={self.norm.__name__}," \
               f"radius={self.dimensions})"

    @property
    def dimension(self):
        return self._dim

    @property
    def norm(self):
        return self._norm

    @property
    def dimensions(self):
        return self._size

    def is_contained_in(self, vec):
        vec = np.asarray(vec)
        d = self.dimension
        if len(vec[0]) != d:
            raise ProcessingException("")

        dimensions = self.dimensions
        mask = np.ones((len(vec),), dtype=bool)
        for i in range(d):
            mask = mask & (self.norm(vec) <= dimensions[i])

        return mask


class Polytope(Neighbourhood):

    def __init__(self, d=2, mat=None, b=None):
        if mat is None:
            mat = np.row_stack((np.eye(d), -np.eye(d)))
        if b is None:
            b = np.ones(2*d)

        mat = np.asarray(mat)
        b = np.asarray(b)

        # Sanity checks
        if not mat.size:
            raise ProcessingException("")
        if not b.size:
            raise ProcessingException("")
        if len(mat) != len(b):
            raise ProcessingException("")
        if len(mat[0]) != d:
            raise ProcessingException("")

        self._dim = d
        self._mat = mat
        self._b = b

    @property
    def dimension(self):
        return self._dim

    @property
    def inequalities(self):
        return self._mat

    @property
    def bounds(self):
        return self._b

    def is_contained_in(self, vec):
        vec = np.asarray(vec)
        d = self.dimension
        if len(vec[0]) != d:
            raise ProcessingException("")

        mat = self.inequalities
        b = self.bounds
        mask = np.ones((len(vec),), dtype=bool)
        for ineq, bound in zip(mat, b):
            mask = mask & (ineq @ vec.T <= bound)

        return mask
