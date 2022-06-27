"""
Module contains the abstract base class and inheriting classes for the
handling of weather information.
"""

import numpy as np
import itertools
from bisect import bisect_left
from abc import ABC, abstractmethod
from datetime import timedelta

from hrosailing.globe_model import SphericalGlobe


class OutsideGridException(Exception):
    """Exception raised if point accessed in weather model lies
    outside the available grid"""


class WeatherModel(ABC):
    """
    Base class for handling and approximating weather data.
    How the weather data is organized and how the approximation is executed
    depends on the inheriting classes.

    Abstract Methods
    ----------------
    get_weather
    """

    @abstractmethod
    def get_weather(self, point):
        """Given a space-time point, uses the available weather model
        to calculate the weather at that point

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
        pass


class GriddedWeatherModel(WeatherModel):
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

    Abstract Methods
    ----------------
    interpolate_weather_data
    """

    def __init__(self, data, times, lats, lons, attrs):
        self._times = times
        self._lats = lats
        self._lons = lons
        self._attrs = attrs
        self._data = data

    @property
    def grid(self):
        return self._times, self._lats, self._lons

    def get_weather(self, point):
        """Given a space-time point, uses the available weather model
        to calculate the weather at that point

        If the point is not a grid point, the weather data will be
        interpolated via the `interpolate_weather_data` method in dependency
        of the up-to eight grid points which form a cuboid around the `points`.

        See also
        --------
        `WeatherModel.get_weather`
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

        grid = self.grid
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

        val = self.interpolate_weather_data(idxs, point, flags)
        return dict(zip(self._attrs, val))

    @abstractmethod
    def interpolate_weather_data(self, idxs, point, flags):
        """
        Method to interpolate the weather data of the model for a point not
        represented on the grid from reference points.

        Parameter:
        ---------
        idxs: sequence of ints,
            the indices of the reference points

        point: sequence of length 3,
            contains time, latitude and longitude of the observed point

        flags: sequence of bools of length 3,
            Boolean values containing the information weather the lattitude
            of the observed point is supported by the grid, the longitude is
            supported by the grid and if the time is supported by the grid

        Returns
        -------
        val: np.ndarray
            The interpolated weather data
        """


class FlatWeatherModel(GriddedWeatherModel):
    """
    A weather model which organizes weather data in gridded form and
    approximates the weather by using affine interpolations.

    See also
    ---------
    `GriddedWeatherModel`
    """

    def interpolate_weather_data(self, idxs, point, flags):
        """
        Method to interpolate the weather data of the model for a point not
        represented on the grid from reference points.

        See also
        --------
        `GriddedWeatherModel.interpolate_weather_data`
        """
        # point is a grid point
        if len(idxs) == 1:
            i, j, k = idxs.T
            return self.data[i, j, k, :]

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

        interim = [self.data[i, j, k, :] for i, j, k in edges]

        for i in face:
            mu = (point[i] - self.grid[i][end[i]]) / (
                self.grid[i][start[i]] - self.grid[i][end[i]]
            )
            it = iter(interim)
            interim = [mu * left + (1 - mu) * right for left, right in zip(it, it)]

        return interim[0]


class GlobeWeatherModel(GriddedWeatherModel):
    """Models a weather model as a 3-dimensional space-time grid
    where each space-time point has certain values of a given list
    of attributes.
    Points in between are approximated using a given globe model.

    Parameters
    ----------

    globe_model: GlobeModel, optional
        The `GlobeModel` used to approximate points in between.

        Defaults to `SphericalGlobe()`

    time_scale: float, optional

        A scalar used to compare distances in space and time.
        The total distance will be computed as

        math: \sqrt{l^2 + (`time_scale`\cdot t)^2}

        where `l` is the distance in space (according to the globe model) and
        `t` is the distance in time.

        Defaults to 1

    See also
    --------
    `GriddedWeatherModel`
    """

    def __init__(
            self, data, times, lats, lons, attrs,
            globe_model=SphericalGlobe(), time_scale=1
    ):
        super().__init__(data, times, lats, lons, attrs)
        self._globe_model = globe_model
        self._time_scale = time_scale

    def interpolate_weather_data(self, idxs, point, flags):
        """
        Interpolates the weather model as a weighted mean, weighted according
        to the distances of the point to the reference points.

        See also
        ---------
        `GriddedWeatherModel.interpolate_weather_data`
        """
        ref_pts = np.row_stack([
            [
                self._times[idx[0]],
                self._lats[idx[1]],
                self._lons[idx[2]]
            ]
            for idx in idxs
        ])
        weather = np.row_stack([
            self._data[tuple(idx)]
            for idx in idxs
        ])
        if len(weather) == 1:
            return weather[0]

        place_distances = np.array([
            self._globe_model.distance(
                np.array(point[1:], dtype=float),
                np.array(ref_pt[1:], dtype=float)
            )
            for ref_pt in ref_pts
        ])

        time_distances = [
            abs(ref_pt[0] - point[0])
            for ref_pt in ref_pts
        ]

        #If datetime has been used, transform to float value of hours
        time_distances = np.array([
            t.total_seconds()/3600
            if isinstance(t, timedelta) else time_distances
            for t in time_distances
        ])

        distances = np.sqrt(place_distances**2 + time_distances**2)

        #The distances should not be zero
        return np.average(weather, axis=0, weights=1/distances)