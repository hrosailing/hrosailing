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

    lat_lon(point)

    distance(start, end)

    shortest_path(start, end, res)
    """
    pass