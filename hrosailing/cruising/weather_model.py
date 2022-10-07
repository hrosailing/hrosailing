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
        val = _recursive_affine_interpolation(point, self.grid, self._get_data)

        return dict(zip(self._attrs, val))

    def _get_data(self, idxs):
        return self._data[idxs]

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
        data = np.stack(lat_datas, axis=0)
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


def _recursive_affine_interpolation(point, grid, get_data):
    fst = tuple(dimension[0] for dimension in grid)
    lst = tuple(dimension[-1] for dimension in grid)

    outside_left = [pt < left for pt, left in zip(point, fst)]
    outside_right = [pt > right for pt, right in zip(point, lst)]

    if any(outside_left) or any(outside_right):
        raise OutsideGridException(
            f"{point} is outside the grid. Weather data not available."
        )

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

    def recursion(point_, completed=[]):
        # get first entry which has not been computed yet

        dim = len(completed)

        if dim==len(point_): # terminate
            return get_data(tuple(completed))

        comp = point_[dim]

        if comp in cuboid_vals[dim]:
            j = cuboid_vals[dim].index(point_[dim])
            return recursion(point_, completed + [j])

        idx0 = cuboid[dim][0]
        idx1 = cuboid[dim][1]

        val1 = grid[dim][idx1]
        val0 = grid[dim][idx0]

        lamb = (comp - val1)/(val0-val1)

        data0 = point_[:dim] + [val0] + point_[dim + 1:]
        data1 = point_[:dim] + [val1] + point_[dim + 1:]

        term0 = recursion(data0, completed + [idx0])
        term1 = recursion(data1, completed + [idx1])

        return lamb*term0 + (1-lamb)*term1

    return recursion(list(point))


class NetCDFWeatherModel(GriddedWeatherModel):
    """
    A weather model that uses gridded data from a NetCDF (.nc or .nc4) file.
    Uses the same interpolation method as `GriddedWeatherModel`.
    The module netCDF4 has to be installed in order to use this class.
    The methods `from_file` and `to_file` are not supported.

    Parameter
    ----------
    path: str
        path to the NetCDF file to be used.

    aliases: dict with keys ["lat", "lon", "datetime"], optional
        Contains the aliases for Latitude, Longitude and the Timestamp used in the NetCDF file.

        Defaults to `{"lat": "latitude", "lon": "longitude", "datetime": "time"}`.

    See also
    --------
    `GriddedWeatherModel`
    """

    def __init__(
            self,
            path,
            aliases={"lat": "latitude", "lon": "longitude", "datetime": "time"}
    ):
        try:
            import netCDF4 as nc
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Install netCDF4 in order to use NetCDFWeatherModel.")
        self._dataset = nc.Dataset(path)
        self._aliases = aliases

        lats = self._dataset[aliases["lat"]]
        lons = self._dataset[aliases["lon"]]
        #check if in descending order and change order if necessary
        self._lats_flipped, self._lons_flipped = lats[-1] < lats[0], lons[-1] < lons[0]
        if self._lats_flipped:
            lats = np.flip(lats)
        if self._lons_flipped:
            lons = np.flip(lons)

        time = aliases["datetime"]
        plain_times = self._dataset[time][:]
        unit_str = self._dataset.variables[time].units
        if unit_str.startswith("hours since "):
            timestep = timedelta(hours=1)
            datetime_str = unit_str[12:]
        elif unit_str.startswith("seconds since "):
            timestep = timedelta(seconds=1)
            datetime_str = unit_str[14:]
        else:
            raise NotImplementedError(f"NetCDFWeatherModel can not interpret time units '{unit_str}'.")
        fmts = [
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S"
        ]
        starting_time = None
        for fmt in fmts:
            try:
                starting_time = datetime.strptime(datetime_str, fmt)
                break
            except ValueError:
                pass
        if starting_time is None:
            raise NotImplementedError(
                f"NetCDFWeatherModel does not support the time format of `{datetime_str}`. "
                f"Use one of {fmts} instead."
            )

        times = [starting_time + int(plain_time)*timestep for plain_time in plain_times]

        attrs = [var for var in self._dataset.variables if var not in aliases.values()]

        super().__init__(None, times, lats, lons, attrs)

    def _get_data(self, idxs):
        times, lats, lons = self.grid
        time_idx = idxs[0]
        lat_idx = len(lats) - idxs[1] - 1 if self._lats_flipped else idxs[1]
        lon_idx = len(lons) - idxs[2] - 1 if self._lons_flipped else idxs[2]

        idxs = (time_idx, lat_idx, lon_idx)

        attr_list = []
        for attr in self._attrs:
            try:
                coords = self._dataset[attr].coordinates.split(" ")
            except AttributeError:
                attr_list.append(self._dataset[attr][idxs])
                continue
            new_idxs = []
            for coord in coords:
                if coord == self._aliases["lat"]:
                    new_idxs.append(lat_idx)
                elif coord == self._aliases["lon"]:
                    new_idxs.append(lon_idx)
                elif coord == self._aliases["datetime"]:
                    new_idxs.append(time_idx)
                else:
                    raise NotImplementedError(
                        f"No handling of variables with coordinates other than latitude, longitude and time as `{coord}` are supported"
                    )
            attr_list.append(self._dataset[attr][new_idxs])


        return np.asarray(attr_list)


class MultiWeatherModel(WeatherModel):
    """
    Weather model that manages multiple weather models at once and combines their output.

    Parameter
    ----------
    *args
        An arbitrary number of weather models.

    exception_sensitive: bool, optional
        If `False` any `ValueError`, `TypeError`, `KeyError`, `NotImplementedError` or `OutsideGridException`
        raised by some submodel is ignored. If `True` these exceptions are not ignored.

        Defaults to `False`.
    """
    def __init__(self, *args, exception_sensitive=False):
        self._models = args
        self._exception_sensitive = exception_sensitive

    def get_weather(self, point):
        """
        Evaluates all submodels and combines the results.

        See also
        --------
        `WeatherModel.get_weather`
        """
        weather_data = {}
        for model in self._models:
            try:
                weather_data.update(
                    model.get_weather(point)
                )
            except (ValueError, TypeError, KeyError, OutsideGridException) as e:
                if self._exception_sensitive:
                    raise e
        return weather_data