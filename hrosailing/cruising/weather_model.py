"""
This Module contains the abstract base class and inheriting classes for the
handling of weather information.
"""

import json
from abc import ABC, abstractmethod
from bisect import bisect_left
from datetime import datetime, timedelta

import numpy as np

try:
    import netCDF4 as nc

    installed_netcdf = True
except ModuleNotFoundError:
    installed_netcdf = False

try:
    import pandas as pd

    installed_pandas = True
except ModuleNotFoundError:
    installed_pandas = False

try:
    import meteostat

    installed_meteostat = True
except ModuleNotFoundError:
    installed_meteostat = False


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
        to calculate the weather at that point.

        Parameters
        ----------
        point: tuple of length 3
            Space-time point given as tuple of time, latitude
            and longitude.

        Returns
        -------
        weather : dict
            The weather data at the given point.

            If it is a grid point, the weather data is taken straight
            from the model, else it is interpolated as described above.
        """


class GriddedWeatherModel(WeatherModel):
    """Models a weather model as a 3-dimensional space-time grid
    where each space-time point has certain values of a given list
    of attributes. Points in between are approximated affinely.

    Parameters
    ----------
    data : array_like of shape (n, m, r, s)
        Weather data at different space-time grid points.

    times : list of length n
        Sorted list of time values of the space-time grid.

    lats : list of length m
        Sorted list of latitude values of the space-time grid.

    lons : list of length r
        Sorted list of longitude values of the space-time grid.

    attrs : list of length s
        List of different (scalar) attributes of weather.

    """

    def __init__(self, data, times, lats, lons, attrs):
        self._times = times
        self._lats = np.array(lats)
        self._lons = np.array(lons)
        self._attrs = attrs
        self._data = np.array(data)

    @property
    def grid(self):
        return self._times, self._lats.copy(), self._lons.copy()

    @property
    def data(self):
        return self._data.copy()

    @property
    def attrs(self):
        return self._attrs

    def get_weather(self, point):
        """Given a space-time point, uses the available weather model
        to calculate the weather at that point.

        If the point is not a grid point, the weather data will be
        interpolated via recursive affine interpolation
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

        Parameters
        -----------

        lats : list of floats
            Sorted list of latitude values of the space-time grid.

        lons : list floats
            Sorted list of longitude values of the space-time grid.

        start_time : datetime.datetime
            The smallest time component of the grid.

        end_time : datetime.datetime
            (Approximately) the biggest time component of the grid.

        keys : list of str
            `meteostat` keys to be included in the weather model.

        Returns
        -------
        wm: `GriddedWeatherModel` as described above

        """
        if (not installed_meteostat) or (not installed_pandas):
            raise ImportError(
                """The modules `meteostat` and `pandas` are
                necessary to use `GriddedWeatherModel.from_meteostat`"""
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

        Parameters
        ---------
        path : path like
            The path of the written file.
        """
        with open(path, "w", encoding="utf-8") as file:
            file.write(json.dumps(self, cls=_GriddedWeatherModelEncoder))

    @classmethod
    def from_file(cls, path):
        """
        Reads a gridded weather model from a compatible `json` file
        (for example those created via the `to_file` method)

        Parameters
        ---------
        path : path like
            The path of the file to be read.
        """
        with open(path, "r", encoding="utf-8") as file:
            data, times, *rest = json.loads(file.read())
            times = [datetime.strptime(t, "%d.%m.%Y:%X") for t in times]
            return cls(data, times, *rest)


class _GriddedWeatherModelEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, GriddedWeatherModel):
            data = o.data
            times, lats, lons = o.grid
            attrs = o.attrs
            return [data, times, lats, lons, attrs]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if installed_pandas and isinstance(o, pd.Timestamp):
            return o.strftime("%d.%m.%Y:%X")
        if isinstance(o, datetime):
            return o.strftime("%d.%m.%Y:%X")
        raise TypeError(
            f"Object of type {type(o)} is not JSON serializable :("
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
        bisect_left(grid_comp, comp) for grid_comp, comp in zip(grid, point)
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
        [idx - 1, idx] if not flag else [idx] for idx, flag in zip(idxs, flags)
    ]

    cuboid_vals = [
        [grid[dim][idx] for idx in c] for dim, c in enumerate(cuboid)
    ]

    def recursion(point_, completed=None):
        # get first entry which has not been computed yet
        if completed is None:
            completed = []

        dim = len(completed)

        if dim == len(point_):  # terminate
            return get_data(tuple(completed))

        comp = point_[dim]

        if comp in cuboid_vals[dim]:
            j = cuboid_vals[dim].index(point_[dim])
            return recursion(point_, completed + [j])

        idx0 = cuboid[dim][0]
        idx1 = cuboid[dim][1]

        val1 = grid[dim][idx1]
        val0 = grid[dim][idx0]

        lamb = (comp - val1) / (val0 - val1)

        data0 = point_[:dim] + [val0] + point_[dim + 1 :]
        data1 = point_[:dim] + [val1] + point_[dim + 1 :]

        term0 = recursion(data0, completed + [idx0])
        term1 = recursion(data1, completed + [idx1])

        return lamb * term0 + (1 - lamb) * term1

    return recursion(list(point))


class NetCDFWeatherModel(GriddedWeatherModel):
    """
    A weather model that uses gridded data from a NetCDF (.nc or .nc4) file.
    Uses the same interpolation method as `GriddedWeatherModel`.
    The module `netCDF4` has to be installed in order to use this class.
    The methods `from_file`, `to_file` and `from_meteostat` are not supported.

    Parameters
    ----------
    path: str
        Path to the NetCDF file to be used.

    aliases: dict with keys ["lat", "lon", "datetime"], optional
        Contains the aliases for latitude, longitude and the timestamp used in the NetCDF file.

        Defaults to `{"lat": "latitude", "lon": "longitude", "datetime": "time"}`.

    See also
    --------
    `GriddedWeatherModel`
    """

    def __init__(
        self,
        path,
        aliases=None,
        further_indices=None,
    ):
        if not installed_netcdf:
            raise ModuleNotFoundError(
                "Install netCDF4 to use NetCDFWeatherModel"
            )

        if aliases is None:
            aliases = {
                "lat": "latitude",
                "lon": "longitude",
                "datetime": "time",
            }
        if further_indices is None:
            further_indices = {}

        self._dataset = nc.Dataset(path)
        self._aliases = aliases
        self._further_indices = further_indices

        lats = self._dataset[aliases["lat"]]
        lons = self._dataset[aliases["lon"]]
        # check if in descending order and change order if necessary
        self._lats_flipped, self._lons_flipped = (
            lats[-1] < lats[0],
            lons[-1] < lons[0],
        )
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
            raise NotImplementedError(
                "NetCDFWeatherModel can not interpret time units"
                f" '{unit_str}'."
            )
        fmts = ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"]
        starting_time = None
        for fmt in fmts:
            try:
                starting_time = datetime.strptime(datetime_str, fmt)
                break
            except ValueError:
                pass
        if starting_time is None:
            raise NotImplementedError(
                "NetCDFWeatherModel does not support the time format of"
                f" `{datetime_str}`. Use one of {fmts} instead."
            )

        times = [
            starting_time + int(plain_time) * timestep
            for plain_time in plain_times
        ]

        attrs = [
            var
            for var in self._dataset.variables
            if var not in aliases.values()
        ]

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
                coords = self._get_dimensions(attr)
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
                elif coord in self._further_indices:
                    new_idxs.append(self._further_indices[coord])
                else:
                    raise NotImplementedError(
                        "For the handling of variables other than 'latitude',"
                        f" 'longitude' and 'time' as '{coord}' initialize the"
                        " NetCDFWeatherModel with "
                        f"'further_indices={{'{coord}': a}}'"
                    )
            attr_list.append(self._dataset[attr][tuple(new_idxs)])

        return np.asarray(attr_list)

    def _get_dimensions(self, attr):
        dimensions = str(self._dataset[attr]).split("\n")[1].split(" ")[1:]
        dimensions = [
            c.split("(")[1].strip(",)") if "(" in c else c.strip(",)")
            for c in dimensions
        ]
        return dimensions


class MultiWeatherModel(WeatherModel):
    """
    Weather model that manages multiple weather models at once and combines their output.

    Parameters
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
                weather_data.update(model.get_weather(point))
            except (
                ValueError,
                TypeError,
                KeyError,
                OutsideGridException,
            ) as e:
                if self._exception_sensitive:
                    raise e
        return weather_data
