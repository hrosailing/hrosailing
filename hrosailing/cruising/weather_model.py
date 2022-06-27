"""
Module contains the base class and inheriting classes for the handling of
weather information.
"""

import numpy as np
import itertools
from bisect import bisect_left


class OutsideGridException(Exception):
    """Exception raised if point accessed in weather model lies
    outside the available grid"""


class WeatherModel:
    """Models a weather model as a 3-dimensional space-time grid
    where each space-time point has certain values of a given list
    of attributes. Points in between are approximated affinely.

    Parameters
    ----------
    data : array_like of shape (n, m, r, s)
        Weather data at different space-time grid points

    times : list of length n
        Sorted list of time values of the space-time grid

    lats : list of length m
        Sorted list of lattitude values of the space-time grid

    lons : list of length r
        Sorted list of longitude values of the space-time grid

    attrs : list of length s
        List of different (scalar) attributes of weather
    """

    def __init__(self, data, times, lats, lons, attrs):
        self._times = times
        self._lats = lats
        self._lons = lons
        self._attrs = attrs
        self._data = data

    def _grid(self):
        return self._times, self._lats, self._lons

    def get_weather(self, point):
        """Given a space-time point, uses the available weather model
        to calculate the weather at that point

        If the point is not a grid point, the weather data will be
        affinely interpolated, starting with the time-component, using
        the (at most) 8 grid points that span the vertices of a cube, which
        contains the given point

        Parameters
        ----------
        point: tuple of length 3
            Space-time point given as tuple of time, lattitude
            and longitude

        Returns
        -------
        weather : dict
            The weather data at the given point.

            If it is a grid point, the weather data is taken straight
            from the model, else it is interpolated as described above
        """
        # check if given point lies in the grid
        fst = (self._times[0], self._lats[0], self._lons[0])
        lst = (self._times[-1], self._lats[-1], self._lons[-1])

        outside_left = [pt < left for pt, left in zip(point, fst)]
        outside_right = [pt > right for pt, right in zip(point, lst)]

        if any(outside_left) or any(outside_right):
            raise OutsideGridException(
                "`point` is outside the grid. Weather data not available."
            )

        grid = self._grid()
        idxs = [
            bisect_left(grid_comp, comp)
            for grid_comp, comp in zip(grid, point)
        ]
        flags = [
            grid_pt[idx] == pt
            for grid_pt, idx, pt in zip(
                grid,
                idxs,
                point,
            )
        ]

        cuboid = [
            [idx - 1, idx] if not flag else [idx]
            for idx, flag in zip(idxs, flags)
        ]

        cuboid = np.meshgrid(*cuboid)
        idxs = np.vstack(tuple(map(np.ravel, cuboid))).T

        val = _interpolate_weather_data(self._data, idxs, point, flags, grid)
        return dict(zip(self._attrs, val))


def _interpolate_weather_data(data, idxs, point, flags, grid):
    """"""
    # point is a grid point
    if len(idxs) == 1:
        i, j, k = idxs.T
        return data[i, j, k, :]

    # lexicograpic first and last vertex of cube
    start = idxs[0]
    end = idxs[-1]

    # interpolate along time edges first
    if flags[0] and flags[1] and not flags[2]:
        idxs[[1, 2]] = idxs[[2, 1]]

    face = [i for i, flag in enumerate(flags) if not flag]

    if len(face) == 1:
        edges = [idxs[0], idxs[1]]
    else:
        edges = [0, 1] if len(face) == 2 else [0, 1, 4, 5]
        edges = [(idxs[i], idxs[i + 2]) for i in edges]
        flatten = itertools.chain.from_iterable
        edges = list(flatten(edges))

    interim = [data[i, j, k, :] for i, j, k in edges]

    for i in face:
        mu = (point[i] - grid[i][end[i]]) / (
            grid[i][start[i]] - grid[i][end[i]]
        )
        it = iter(interim)
        interim = [mu * left + (1 - mu) * right for left, right in zip(it, it)]

    return interim[0]