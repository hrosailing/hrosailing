"""
Module contains the abstract base class and inheriting classes for the
handling of weather information.
"""

import numpy as np
import itertools
from bisect import bisect_left
from abc import ABC, abstractmethod
from datetime import timedelta, datetime
#from math import prod

import json


def prod(list_):
    res = 1
    for l in list_:
        res *= l
    return res

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
            Space-time point given as tuple of time, latitude
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

    """

    def __init__(self, data, times, lats, lons, attrs):
        self._times = times
        self._lats = np.array(lats)
        self._lons = np.array(lons)
        self._attrs = attrs
        self._data = np.array(data)

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
                f"{point} is outside the grid. Weather data not available."
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

        cuboid_vals = [
            [grid[dim][idx] for idx in c]
            for dim, c in enumerate(cuboid)
        ]

        def recursive_affine_interpolation(data, completed=[]):
            # get first entry which has not been computed yet

            dim = len(completed)

            if dim==len(data): # terminate
                return self._data[tuple(completed)]

            comp = data[dim]

            if comp in cuboid_vals[dim]:
                j = cuboid_vals[dim].index(data[dim])
                return recursive_affine_interpolation(data, completed + [j])

            idx0 = cuboid[dim][0]
            idx1 = cuboid[dim][1]

            val1 = grid[dim][idx1]
            val0 = grid[dim][idx0]

            lamb = (comp - val1)/(val0-val1)

            data0 = data[:dim] + [val0] + data[dim+1:]
            data1 = data[:dim] + [val1] + data[dim+1:]

            term0 = recursive_affine_interpolation(data0, completed + [idx0])
            term1 = recursive_affine_interpolation(data1, completed + [idx1])

            return lamb*term0 + (1-lamb)*term1

        val = recursive_affine_interpolation(list(point))

        return dict(zip(self._attrs, val))

    @classmethod
    def from_meteostat(cls, lats, lons, start_time, end_time, keys):
        """
        Uses the `meteostat` module to fetch gridded weather data from the web.
        To use this method, you need to have the module `meteostat` installed
        and a connection to the internet.
        The time component of the resulting gridded data will have an hourly
        resolution.

        Parameter
        -----------

        lats : list of floats
            Sorted list of lattitude values of the space-time grid

        lons : list floats
            Sorted list of longitude values of the space-time grid

        start_time : datetime.datetime
            Smallest time component of the grid

        end_time : datetime.datetime
            (Approximately) the biggest time component of the grid

        keys : list of str,
            meteostat keys to be included in the weather model

        Returns
        -------
        wm: `GriddedWeatherModel` as described above

        """
        try:
            import meteostat
            import pandas as pd
        except ImportError:
            raise ImportError(
                f"The modules `meteostat` and `pandas` are necessary in order "
                f"to use `from_meteostat`"
            )

        # fetch data
        lat_datas = []
        for lat in lats:
            lon_data = []
            for lon in lons:
                loc = meteostat.Point(lat, lon, 0)
                data = meteostat.Hourly(loc, start_time, end_time).fetch()
                times = data.index.values
                np_data = data[keys].to_numpy()
                if np_data.shape[0] == 0:
                    np_data = []
                lon_data.append(np_data)
            max_len = max(len(d) for d in lon_data)
            lon_data = [
                np.zeros((max_len, len(keys))) if d == [] else d
                for d in lon_data
            ]
            lat_datas.append(np.stack(lon_data, axis=0))

        times = [pd.to_datetime(t) for t in times]
        data=np.stack(lat_datas, axis=0)
        data = np.transpose(np.stack(lat_datas, axis=0), axes=(2, 0, 1, 3))

        return cls(data, times, lats, lons, keys)

    def to_file(self, path):
        """
        Writes the data of the weather model to a `json` file such that it can
        be read via `from_file` method.

        Parameter
        ---------
        path : path like
            the path of the written file
        """
        with open(path, "w") as file:
            file.write(json.dumps(self, cls=_GriddedWeatherModelEncoder))

    @classmethod
    def from_file(cls, path):
        """
        Reads a gridded weather model from a compatible `json` file
        (for example those created via the `to_file` method)

        Parameter
        ---------
        path : path like
            The path of the file to be read
        """
        with open(path, "r") as file:
            data, times, *rest = json.loads(file.read())
            times = [datetime.strptime(t, "%d.%m.%Y:%X") for t in times]
            return cls(data, times, *rest)


class _GriddedWeatherModelEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, GriddedWeatherModel):
            return [obj._data, obj._times, obj._lats, obj._lons, obj._attrs]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%d.%m.%Y:%X")
        raise TypeError(
            f"Object of type {type(obj)} is not JSON serializable :("
        )