"""
Collection of various function to
determine weights of data points

Each function returns an array
containing the weights of
the corresponding data point
"""

# Author: Valentin F. Dannenberg / Ente


from abc import ABC, abstractmethod
import numpy as np

from exceptions import ProcessingException
from utils import convert_wind, euclidean_norm


class WeightedPoints:
    """
    """
    def __init__(self, points, weights=None,
                 weigher=None, tw=True):
        points = np.asarray(points)

        if len(points[0]) != 3:
            try:
                points = points.reshape(-1, 3)
            except ValueError:
                raise ProcessingException(
                    "points could not be broadcasted "
                    "to an array of shape (n,3)")

        self._points = convert_wind(points, tw)

        if weigher is None:
            weigher = CylindricMeanWeigher

        if not isinstance(weigher, Weigher):
            raise ProcessingException("")

        if weights is None:
            self._weights = weigher.weigh(points)
        else:
            weights = np.asarray(weights)
            no_pts = len(points)

            if len(weights) != no_pts:
                try:
                    weights = weights.reshape(no_pts, )
                except ValueError:
                    raise ProcessingException(
                        f"weights could not be broadcasted"
                        f"to an array of shape ({no_pts}, )")

            self._weights = weights

    @property
    def points(self):
        return self._points.copy()

    @property
    def weights(self):
        return self._weights.copy()

    def __repr__(self):
        return f"""WeightedPoints(points={self.points},
        weights={self.weights})"""

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.points):
            self.index += 1
            return self.points[self.index], \
                self.weights[self.index]

        raise StopIteration

    def __getitem__(self, mask):
        return WeightedPoints(
            points=self.points[mask],
            weights=self.weights[mask])


class Weigher(ABC):

    @abstractmethod
    def weigh(self, points):
        pass


class CylindricMeanWeigher(Weigher):

    def __init__(self, radius=1, norm=None):
        if not isinstance(radius, (int, float)):
            raise ProcessingException("")

        if norm is None:
            norm = np.abs

        self._radius = radius
        self._norm = norm

    @property
    def radius(self):
        return self._radius

    @property
    def norm(self):
        return self._norm

    def weigh(self, points):
        points = np.asarray(points)
        if not points.size:
            raise ProcessingException("")
        if len(points[0]) != 3:
            raise ProcessingException("")

        weights = np.zeros(len(points))

        for i, point in enumerate(points):
            mask = self.norm(points[:, :2] - point[:2]) <= self.radius
            cylinder = points[mask][:, 2]
            std = np.std(cylinder)
            mean = np.mean(cylinder)
            weights[i] = np.abs(mean - point[2]) / std

        return weights / max(weights)


class CylindricMemberWeigher(Weigher):

    def __init__(self, radius=1, length=1, norm=None):
        if not isinstance(radius, (int, float)):
            raise ProcessingException("")

        if not isinstance(length, (int, float)):
            raise ProcessingException("")

        if norm is None:
            norm = euclidean_norm

        self._radius = radius
        self._length = length
        self._norm = norm

    @property
    def radius(self):
        return self._radius

    @property
    def length(self):
        return self._length

    @property
    def norm(self):
        return self._norm

    def weigh(self, points):
        points = np.asarray(points)
        if not points.size:
            raise ProcessingException("")
        if len(points) != 3:
            raise ProcessingException("")

        weights = np.zeros(len(points))

        for i, point in enumerate(points):
            mask_l = np.abs(points[:, 0] - point[0]) <= self.length
            mask_r = self.norm(points[:, 1:] - point[1:]) <= self.radius

            weights[i] = len(points[mask_l & mask_r]) - 1

        return weights / max(weights)
