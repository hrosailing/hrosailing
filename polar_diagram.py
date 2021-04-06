import csv
# import logging
# import logging.handlers
import pickle
# import sys
from abc import ABC, abstractmethod
from collections import Iterable
from tabulate import tabulate
from utils import *


# logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
# LOG_FILE = "PolarDiagram.log"

# polardiagram_logger = logging.getLogger(__name__)
# console_handler = logging.StreamHandler(sys.stdout)
# file_handler = logging.handlers.TimedRotatingFileHandler(LOG_FILE, when='midnight')
# polardiagram_logger.addHandler(console_handler)
# polardiagram_logger.setLevel(logging.DEBUG)

# V: Noch in Arbeit
def convert(obj, convert_type):
    """Converts a PolarDiagram object to another type of PolarDiagram object

    :param obj:
        PolarDiagram object to be converted
    :type obj: ``PolarDiagram``
    :param convert_type:

    :type convert_type: ``PolarDiagram``
    """
    if convert_type is type(obj):
        return obj

    wind_speeds = obj.wind_speeds
    wind_angles = obj.wind_angles
    boat_speeds = obj.boat_speeds
    data = []
    cols = len(wind_speeds)
    rows = len(wind_angles)

    if convert_type is PolarDiagramTable:
        for i in range(rows):
            row = []
            for j in range(cols):
                point = boat_speeds[boat_speeds[:1] == [wind_speeds[j], wind_angles[i]]][2]
                row.append(point)
            data.append(row)

        return PolarDiagramTable(wind_angle_resolution=wind_angles,
                                 wind_speed_resolution=wind_speeds,
                                 data=data)

    if convert_type is PolarDiagramPointcloud:
        for i in range(rows):
            for j in range(cols):
                data.append([wind_speeds[j], wind_angles[i], boat_speeds[i][j]])

        return PolarDiagramPointcloud(data=np.array(data))


def to_csv(csv_path, obj):
    """Writes a PolarDiagram object to a .csv-file

    :param csv_path:
        Path to a .csv-file
    :type csv_path: ``str``
    :param obj:
        PolarDiagram object
    :type obj: ``PolarDiagram``
    """
    obj.to_csv(csv_path)


def from_csv(csv_path):
    """Creates a PolarDiagram object from a .csv-file

    :param csv_path:
        Path to a .csv-file
    :type csv_path: ``str``
    """
    with open(csv_path, 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=',', quotechar='"')
        row1 = next(csv_reader)[0]
        if row1 == "PolarDiagramTable":
            next(csv_reader)
            wind_speed_resolution = [eval(s) for s in next(csv_reader)]
            if len(wind_speed_resolution) == 1:
                wind_speed_resolution = wind_speed_resolution[0]
            next(csv_reader)
            wind_angle_resolution = [eval(a) for a in next(csv_reader)]
            if len(wind_angle_resolution) == 1:
                wind_angle_resolution = wind_angle_resolution[0]
            data = []
            next(csv_reader)
            for row in csv_reader:
                data.append([eval(entry) for entry in row])
            return PolarDiagramTable(data=np.array(data),
                                     wind_angle_resolution=wind_angle_resolution,
                                     wind_speed_resolution=wind_speed_resolution)

        elif row1 == "PolarDiagramPointcloud":
            next(csv_reader)
            data = []
            for row in csv_reader:
                data.append([eval(entry) for entry in row])
            return PolarDiagramPointcloud(data=np.array(data))


def pickling(pkl_path, obj):
    """Writes a PolarDiagram object to a .pkl-file

    :param pkl_path:
        Path to a .pkl-file
    :type pkl_path: ``str``
    :param obj:
        PolarDiagram object
    :type obj: ``PolarDiagram``
    """
    obj.pickling(pkl_path)


def depickling(pkl_path):
    """Creates a PolarDiagram object from a .pkl-file

    :param pkl_path:
        Path to a .pkl-file
    :type pkl_path: ``str``
    """
    with open(pkl_path, 'rb') as file:
        return pickle.load(file)


class PolarDiagramException(Exception):
    def __init__(self, exception_type, *args):
        message_dict = {
            "Wrong dimension": "Expecting 2 dimensional array to be viewed as Polar Diagram Tableau," +
                               f"\n got {args[0]} dimensional array instead.",
            "Wrong resolution": "Expecting resolution of type 'Iterable' or 'int/float'," +
                                f"\n got resolution of type {args[0]} instead",
            "Wrong shape": f"Expecting array with shape {args[0]},\n got array with shape {args[1]} instead",
            "Wind speed not in resolution": f"Expecting wind speed to be in {args[0]},\n got {args[1]} instead",
            "Wind angle not in resolution": f"Expecting wind angle to be in {args[0]},\n got {args[1]} instead",
            "No points found": f"The given true wind speed {args[0]} yielded no points in the current point cloud",
        }
        if exception_type in message_dict:
            super().__init__(message_dict[exception_type])
        else:
            super().__init__(exception_type)


class PolarDiagram(ABC):
    """
    An abstract base class

    ...

    Methods
    ------
    pickling(pkl_path):

    Abstract Methods
    ----------------
    __str__()
    __repr__()
    wind_speeds
    wind_angles
    boat_speeds
    to_csv(csv_path)
    polar_plot_slice(wind_speed, **kwargs)
    flat_plot_slice(wind_speed, **kwargs)
    plot_3d()
    plot_convex_hull_slice(wind_speed, **kwargs)
    plot_convex_hull_3d()
    """
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @property
    @abstractmethod
    def wind_speeds(self):
        pass

    @property
    @abstractmethod
    def wind_angles(self):
        pass

    @property
    @abstractmethod
    def boat_speeds(self):
        pass

    @abstractmethod
    def to_csv(self, csv_path):
        pass

    def pickling(self, pkl_path):
        """

        :param pkl_path:

        :type pkl_path: ``string``
        """
        with open(pkl_path, 'wb') as file:
            pickle.dump(self, file)

    @abstractmethod
    def polar_plot_slice(self, wind_speed, **kwargs):
        pass

    @abstractmethod
    def flat_plot_slice(self, wind_speed, **kwargs):
        pass

    @abstractmethod
    def plot_3d(self):
        pass

    @abstractmethod
    def plot_convex_hull_slice(self, wind_speed, **kwargs):
        pass

    @abstractmethod
    def plot_convex_hull_3d(self):
        pass


class PolarDiagramTable(PolarDiagram):
    """
    A class to represent, visualize and work with a polar performance diagram in form of a table.

    ...

    Attributes
    ----------
    _resolution_wind_speed : ``list`` of length cdim

    _resolution_wind_angle : ``list`` of length rdim

    _data : ``numpy.ndarray`` of shape (rdim, cdim)

    Methods
    -------
    __init__(**kwargs):
        Initializes a PolarDiagramTable object
    __str__()
    __repr__()
    wind_angles:
        Returns a read only verion of self._wind_angle_resolution
    wind_speeds:
        Returns a read only version of self._wind_speed_resolution
    boat_speeds:
        Returns a read only version of self._data
    to_csv(csv_path):
        Writes object to a .csv-file
    change_entry(data, **kwargs):
        Changes entries in table
    get_slice_data(wind_speed):
        Returns a column of table
    polar_plot_slice(wind_speed, **kwargs):
        Polar plot of a slice(column) of the diagram
    flat_plot_slice(wind_speed, **kwargs):
        Kartesian plot of a slice(column) of the diagram
    plot_3d():
        3D-plot of the diagram
    plot_convex_hull_slice(wind_speed, **kwargs):
        Polar plot of the convex hull of a slice(column) of the diagram
    plot_convex_hull_3d():
        3D-plot of the convex hull of the diagram
    """
    def __init__(self, **kwargs):
        """Initializes a PolarDiagramTable object

        :param kwargs:
            See below

        :Keyword arguments:
            * *true_wind_speed* (``Iterable``, ``int`` or ``float``) --

            * *apparent_wind_speed* (``Iterable``, ``int`` or ``float``) --

            * *true_wind_angle* (``Iterable``, ``int`` or ``float``) --

            * *apparent_wind_angle* (``Iterable``, ``int`` or ``float``) --

        """
        wind_speed_resolution, wind_angle_resolution = convert_wind(kwargs)

        if wind_angle_resolution is not None:
            if isinstance(wind_angle_resolution, Iterable):
                self._resolution_wind_angle = list(wind_angle_resolution)
            elif isinstance(wind_angle_resolution, (int, float)):
                self._resolution_wind_angle = list(np.arange(wind_angle_resolution, 360,
                                                             wind_angle_resolution))
            else:
                raise PolarDiagramException("Wrong resolution", type(wind_angle_resolution))
        else:
            self._resolution_wind_angle = list(np.arange(0, 360, 5))

        if wind_speed_resolution is not None:
            if isinstance(wind_speed_resolution, Iterable):
                self._resolution_wind_speed = list(wind_speed_resolution)
            elif isinstance(wind_speed_resolution, (int, float)):
                self._resolution_wind_speed = list(np.arange(wind_speed_resolution, 40,
                                                             wind_speed_resolution))
            else:
                raise PolarDiagramException("Wrong resolution", type(wind_speed_resolution))
        else:
            self._resolution_wind_speed = list(np.arange(2, 42, 2))

        if "data" in kwargs:
            data = np.array(kwargs["data"])
            if data.ndim != 2:
                raise PolarDiagramException("Wrong dimension", data.ndim)
            if data.shape != (len(self._resolution_wind_angle), len(self._resolution_wind_speed)):
                raise PolarDiagramException("Wrong shape",
                                            (len(self._resolution_wind_angle), len(self._resolution_wind_speed)),
                                            data.shape)
            self._data = data
        else:
            self._data = np.zeros((len(self._resolution_wind_angle), len(self._resolution_wind_speed)))

    def __str__(self):
        """Returns a tabulate of the first 15 columns of the polar diagram table"""
        table = np.c_[self._resolution_wind_angle, self._data[:15]]
        return tabulate(table, headers=["TWA \\ TWS"] + self._resolution_wind_speed[:15])

    def __repr__(self):
        """Returns a string representation of the polar diagram table"""
        return f"PolarDiagramTable(wind_speed_resolution={self._resolution_wind_speed}, " \
               f"wind_angle_resolution={self._resolution_wind_speed}, data={self._data}"

    @property
    def wind_angles(self):
        """Returns a read only version of self._resolution_wind_angle"""
        return self._resolution_wind_angle

    @property
    def wind_speeds(self):
        """Returns a read only version of self._resolution_wind_speed"""
        return self._resolution_wind_speed

    @property
    def boat_speeds(self):
        """Returns a read only version of self._data"""
        return self._data

    def to_csv(self, csv_path):
        """Creates a .csv-file with the following entries
            PolarDiagramTable
            Wind speed resolution:
            self.wind_speeds
            Wind angle resolution:
            self.wind_angles
            Boat speeds:
            self.boat_speeds

        :param csv_path:
            Path to .csv-file
        :type csv_path: ``str``
        """
        with open(csv_path, 'w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"')
            csv_writer.writerow(["PolarDiagramTable"])
            csv_writer.writerow(["Wind speed resolution:"])
            csv_writer.writerow(self.wind_speeds)
            csv_writer.writerow(["Wind angle resolution:"])
            csv_writer.writerow(self.wind_angles)
            csv_writer.writerow(["Boat speeds:"])
            csv_writer.writerows(self.boat_speeds)

    # V: Noch in Arbeit.
    def change_entry(self, data, **kwargs):
        r"""Changes specified entries in self._data

        :param data:
            New data that is to be added
        :type data: ``array_like``

        :param kwargs:
            See below

        :Keyword Arguments:
            * *true_wind_speed* (``Iterable``, ``int`` or ``float``) --

            * *apparent_wind_speed* (``Iterable``, ``int`` or ``float``) --

            * *true_wind_angle* (``Iterable``, ``int`` or ``float``) --

            * *apparent_wind_angle* (``Iterable``, ``int`` or ``float``) --
        """
        wind_speeds, wind_angles = convert_wind(kwargs)

        if wind_speeds is not None:
            wind_speeds = list(wind_speeds)
            speed_ind = [i for i in range(len(self.wind_speeds))
                         if self.wind_speeds[i] in wind_speeds]
            if len(speed_ind) < len(wind_speeds):
                raise PolarDiagramException("Wind speed not in resolution", self.wind_speeds, wind_speeds)
            if wind_angles is not None:
                wind_angles = list(wind_angles)
                angle_ind = [i for i in range(len(self.wind_angles))
                             if self.wind_angles[i] in wind_angles]
                if len(angle_ind) < len(wind_angles):
                    raise PolarDiagramException("Wind angle not in resolution", self.wind_angles, wind_angles)
                data = np.array(data)
                mask = np.zeros(self.boat_speeds.shape, dtype=bool)
                for i in angle_ind:
                    for j in speed_ind:
                        mask[i, j] = True
                try:
                    self._data[mask] = data.flat
                except ValueError:
                    raise PolarDiagramException("Wrong Shape", (len(angle_ind), len(speed_ind)),
                                                data.shape)
            else:
                data = np.array(data)
                try:
                    self._data[:, speed_ind] = data
                except ValueError:
                    raise PolarDiagramException("Wrong Shape", (len(speed_ind), ), data.shape)
        elif wind_angles is not None:
            wind_angles = list(wind_angles)
            angle_ind = [i for i in range(len(self.wind_angles))
                         if self.wind_angles[i] in wind_angles]
            if len(angle_ind) < len(wind_angles):
                raise PolarDiagramException("Wind angle not in resolution", self.wind_angles, wind_angles)
            data = np.array(data)
            try:
                self._data[angle_ind, :] = data
            except ValueError:
                raise PolarDiagramException("Wrong Shape", (len(angle_ind), ), data.shape)
        else:
            data = np.array(data)
            if data.shape != (len(self.wind_angles), len(self.wind_speeds)):
                raise PolarDiagramException("Wrong shape", (len(self.wind_angles), len(self.wind_speeds)),
                                            data.shape)
            self._data = data

    def get_slice_data(self, wind_speed):
        """

        :param wind_speed:

        :type wind_speed: ``int`` or ``float``
        """
        if not isinstance(wind_speed, (int, float)):
            pass

        try:
            column = self.wind_speeds.index(wind_speed)
            return list(self.boat_speeds[:, column])
        except ValueError:
            raise PolarDiagramException("Wind speed not in resolution", self.wind_speeds, wind_speed)

    def polar_plot_slice(self, wind_speed, **kwargs):
        """

        :param wind_speed:

        :type wind_speed: ``int`` or ``float``
        :param kwargs:
            See below

        :Keyword arguments:
            Function accepts the same keyword arguments as the
            matplotlib.pyplot.plot function
        """
        wind_angles = list(np.deg2rad(self.wind_angles))
        boat_speeds = self.get_slice_data(wind_speed)
        return plot_polar(wind_angles, boat_speeds, **kwargs)

    def flat_plot_slice(self, wind_speed, **kwargs):
        """

        :param wind_speed:

        :type wind_speed: ``int`` or ``float``
        :param kwargs:
            See below

        :Keyword arguments:
            Function accepts the same keyword arguments as the
            matplotlib.pyplot.plot function
        """
        wind_angles = self.wind_angles
        boat_speeds = self.get_slice_data(wind_speed)
        return plot_flat(wind_angles, boat_speeds, **kwargs)

    def plot_3d(self):
        pass

    def plot_convex_hull_slice(self, wind_speed, **kwargs):
        """

        :param wind_speed:

        :type wind_speed: ``int`` or ``float``
        :param kwargs:
            See below

        :Keyword arguments:
            Function accepts the same keyword arguments as the
            matplotlib.pyplot.plot function
        """
        angles = list(np.deg2rad(self.wind_angles))
        speeds = self.get_slice_data(wind_speed)
        return plot_convex_hull(angles, speeds, **kwargs)

    def plot_convex_hull_3d(self, **kwargs):
        pass


class PolarDiagramCurve(PolarDiagram):
    """
    A class to represent, visualize and work with a polar performance diagram given by a fitted curve and
    optimal parameters.

    ...

    Attributes
    ----------
    _f : ``function``

    _params : ``list``

    Methods
    -------
    __init__(f, *params):
        Initializes a PolarDiagramCurve object
    __str__():

    __repr__():

    curve:
        Returns a read only version of self._f
    parameters:
        Returns a read only version of self._params
    to_csv(csv_path):
        Writes object to a .csv-file
    polar_plot_slice(wind_speed, **kwargs):

    flat_plot_slice(wind_speed, **kwargs):

    plot_3d():

    plot_convex_hull_slice(wind_speed, **kwargs):

    plot_convex_hull_3d():

    """
    def __init__(self, f, *params):
        """Initializes a PolarDiagramCurve object

        :param f:
            A function of wind angle, wind speed and certain parameters
        :type f: ``function``
        :param params:


        """
        self._f = f
        self._params = params

    def __str__(self):
        """"""
        return f"Function: {self._f}\n Optimal parameters: {self._params}"

    def __repr__(self):
        """"""
        return f"PolarDiagramCurve(f={self._f}, params={self._params})"

    @property
    def curve(self):
        """"""
        return self._f

    @property
    def parameters(self):
        """"""
        return self._params

    # dummy property
    @property
    def wind_angles(self):
        return

    # dummy property
    @property
    def wind_speeds(self):
        return

    # dummy property
    @property
    def boat_speeds(self):
        return

    def to_csv(self, csv_path):
        """Creates a .csv-file with the following entries
            PolarDiagramCurve
            Function:
            self.curve
            Parameters:
            self.parameters

        :param csv_path:
            Path to .csv-file
        :type csv_path: ``str``
        """
        with open(csv_path, 'w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"')
            csv_writer.writerow(["PolarDiagramCurve"])
            csv_writer.writerow(["Function:"])
            csv_writer.writerow([self.curve])
            csv_writer.writerow(["Parameters:"])
            csv_writer.writerow(self.parameters)

    def polar_plot_slice(self, wind_speed, **kwargs):
        """

        :param wind_speed:

        :type wind_speed: ``int`` or ``float``
        :param kwargs:
            See below

        :Keyword arguments:
            Function accepts the same keyword arguments as the
            matplotlib.pyplot.plot function
        """
        wind_angles = list(np.deg2rad(np.linspace(0, 360, 1000)))
        boat_speeds = [self.curve(wind_angle, wind_speed, self.parameters)
                       for wind_angle in wind_angles]
        return plot_polar(wind_angles, boat_speeds, **kwargs)

    def flat_plot_slice(self, wind_speed, **kwargs):
        """

        :param wind_speed:

        :type wind_speed: ``int`` or ``float``
        :param kwargs:
            See below

        :Keyword arguments:
            Function accepts the same keyword arguments as the
            matplotlib.pyplot.plot function
        """
        wind_angles = list(np.linspace(0, 360, 1000))
        boat_speeds = [self.curve(wind_angle, wind_speed, self.parameters)
                       for wind_angle in wind_angles]
        return plot_polar(wind_angles, boat_speeds, **kwargs)

    def plot_3d(self):
        pass

    def plot_convex_hull_slice(self, wind_speed, **kwargs):
        wind_angles = list(np.deg2rad(np.linspace(0, 360, 1000)))
        boat_speeds = [self.curve(wind_angle, wind_speed, self.parameters)
                       for wind_angle in wind_angles]
        return plot_convex_hull(wind_angles, boat_speeds, **kwargs)

    def plot_convex_hull_3d(self):
        pass


class PolarDiagramPointcloud(PolarDiagram):
    """
    A class to represent, visualize and work with a polar performance diagram given by a point cloud

    ...

    Attributes
    ----------
    self._data : ``numpy.ndarray`` of shape (x, 3)

    Methods
    -------
    __init__(data, **kwargs):
        Initializes a PolarDiagramPointcloud object
    __str__():

    __repr__():

    wind_speeds:
        Returns a list of all occuring wind speeds
    wind_angles:
        Returns a list of all occuring wind angles
    points:
        Returns a read only version of self.data
    to_csv:
        Writes object to a .csv-file
    add_points(new_points):
        Appends given points to self._data
    change_points():

    polar_plot_slice(wind_speed, **kwargs):

    flat_plot_slice(wind_speed, **kwargs):

    plot_convex_hull_slice(wind_speed, **kwargs):
    """
    def __init__(self, data=np.array([[0, 0, 0]]), **kwargs):
        """Initializes a PolarDiagramPointcloud object

        :param data:

        :type data: ``array_like`` of shape (x, 3)
        :param kwargs:
            See below

        :Keyword arguments:

        """
        if len(data[0]) != 3:
            raise PolarDiagramException("Wrong shape", (len(data[0]), "x"), (3, "x"))
        self._data = np.array(data)

    def __str__(self):
        """"""
        return tabulate(self._data, headers=["TWS", "TWA", "BSP"])

    def __repr__(self):
        """"""
        return f"PolarDiagramPointcloud(data={self.points})"

    @property
    def wind_speeds(self):
        """Returns a list of all occuring wind speeds"""
        return list(dict.fromkeys(self._data[:, 0]))

    @property
    def wind_angles(self):
        """Returns a list of all occuring wind angles"""
        return list(dict.fromkeys(self._data[:, 1]))

    # dummy property
    @property
    def boat_speeds(self):
        return self._data

    @property
    def points(self):
        """Returns a read only version of self._data"""
        return self._data

    def to_csv(self, csv_path):
        """Creates a .csv-file with the following entries
            PolarDiagramPointcloud
            True wind speed: , True wind angle: , Boat speed:
            self.get_points

        :param csv_path:
            Path to .csv-file
        :type csv_path: ``str``
        """
        with open(csv_path, 'w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"')
            csv_writer.writerow(["PolarDiagramPointcloud"])
            csv_writer.writerow(["True wind speed ", "True wind angle ", "Boat speed "])
            csv_writer.writerows(self.points)

    def add_points(self, new_points):
        """Appends given points to self._data

        :param new_points:
            Points to be added to point cloud
        :type new_points: ``array like`` of shape (x, 3)
        """
        try:
            self._data = np.r_[self.points, np.array(new_points)]
        except ValueError:
            raise PolarDiagramException("Wrong shape", ("x", 3), ("x", len(new_points[0])))

    def polar_plot_slice(self, wind_speed, **kwargs):
        """

        :param wind_speed:

        :type wind_speed: ``int`` or ``float``

        :param kwargs:
            See below

        :Keyword arguments:
            Function accepts the same keyword arguments as the
            matplotlib.pyplot.plot function
        """
        points = self.points[self.points[:, 0] == wind_speed][:, 1:]
        try:
            wind_angles = list(np.deg2rad(points[:, 0]))
            boat_speeds = points[:, 1]
        except (ValueError, IndexError):
            raise PolarDiagramException("No points found", wind_speed)
        return plot_polar(wind_angles, boat_speeds, **kwargs)

    def flat_plot_slice(self, wind_speed, **kwargs):
        """

        :param wind_speed:

        :type wind_speed: ``int`` or ``float``
        :param kwargs:
            See below

        :Keyword arguments:
            Function accepts the same keyword arguments as the
            matplotlib.pyplot.plot function
        """
        points = self.points[self.points[:, 0] == wind_speed][:, 1:]
        try:
            wind_angles = points[:, 0]
            boat_speeds = points[:, 1]
        except (ValueError, IndexError):
            raise PolarDiagramException("No points found", wind_speed)
        return plot_flat(wind_angles, boat_speeds, **kwargs)

    def plot_3d(self):
        """"""
        pass

    def plot_convex_hull_slice(self, wind_speed, **kwargs):
        """

        :param wind_speed:

        :type wind_speed: ``int`` or ``float``
        :param kwargs:
            See below

        :Keyword arguments:
            Function accepts the same keyword arguments as the
            matplotlib.pyplot.plot function
        """
        points = self.points[self.points[:, 0] == wind_speed][:, 1:]
        try:
            angles = list(np.deg2rad(points[:, 0]))
            speeds = points[:, 1]
        except (ValueError, IndexError):
            raise PolarDiagramException("No points found", wind_speed)
        return plot_convex_hull(angles, speeds, **kwargs)

    def plot_convex_hull_3d(self):
        """"""
        pass
