# import logging
# import logging.handlers
import pickle
# import sys
from abc import ABC, abstractmethod
from tabulate import tabulate
from _utils import *
from _exceptions import *


# logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
# LOG_FILE = "PolarDiagram.log"

# polardiagram_logger = logging.getLogger(__name__)
# console_handler = logging.StreamHandler(sys.stdout)
# file_handler = logging.handlers.TimedRotatingFileHandler(LOG_FILE, when='midnight')
# polardiagram_logger.addHandler(console_handler)
# polardiagram_logger.setLevel(logging.DEBUG)

# V: In Arbeit
def convert(obj, convert_type):
    """Converts a PolarDiagram object to another type of PolarDiagram object"""
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

        return PolarDiagramTable(wind_speed_resolution=wind_speeds,
                                 wind_angle_resolution=wind_angles,
                                 data=data)

    if convert_type is PolarDiagramPointcloud:
        for i in range(rows):
            for j in range(cols):
                data.append([wind_speeds[j], wind_angles[i], boat_speeds[i][j]])

        return PolarDiagramPointcloud(points=data)


# V: Soweit in Ordnung
def to_csv(csv_path, obj):
    """Writes a PolarDiagram object to a .csv-file"""
    obj.to_csv(csv_path)


# V: In Arbeit
def from_csv(csv_path, fmt='own', tws=True, twa=True):
    """Creates a PolarDiagram instance from a .csv-file"""
    if fmt == "own":
        with open(csv_path, 'r', newline='') as file:
            csv_reader = csv.reader(file, delimiter=',', quotechar='"')
            first_row = next(csv_reader)[0]
            if first_row == "PolarDiagramTable":
                wind_speed_resolution, wind_angle_resolution, data = read_table(csv_reader)
                return PolarDiagramTable(wind_speed_resolution=wind_speed_resolution,
                                         wind_angle_resolution=wind_angle_resolution,
                                         data=data, tws=tws, twa=twa)
            elif first_row == "PolarDiagramPointcloud":
                data = read_pointcloud(csv_reader)
                return PolarDiagramPointcloud(points=data, tws=tws, twa=twa)

    elif fmt == "orc":
        wind_speed_resolution, wind_angle_resolution, data = read_orc_csv(csv_path)
        return PolarDiagramTable(wind_speed_resolution=wind_speed_resolution,
                                 wind_angle_resolution=wind_angle_resolution,
                                 data=data, tws=tws, twa=twa)

    elif fmt == "array":
        wind_speed_resolution, wind_angle_resolution, data = read_array_csv(csv_path)
        return PolarDiagramTable(wind_speed_resolution=wind_speed_resolution,
                                 wind_angle_resolution=wind_angle_resolution,
                                 data=data, tws=tws, twa=twa)

    else:
        raise PolarDiagramException("Not yet implemented")


# V: Soweit in Ordnung
def pickling(pkl_path, obj):
    """Writes a PolarDiagram object to a .pkl-file"""
    obj.pickling(pkl_path)


# V: Soweit in Ordnung
def depickling(pkl_path):
    """Creates a PolarDiagram object from a .pkl-file"""
    with open(pkl_path, 'rb') as file:
        return pickle.load(file)


# V: In Arbeit
def symmetric_polar_diagram(obj):
    """"""
    if isinstance(obj, PolarDiagramTable):
        wind_angles = np.concatenate([obj.wind_angles, 360-np.flip(obj.wind_angles)])
        data = np.concatenate([obj.boat_speeds, np.flip(obj.boat_speeds, axis=0)])
        return PolarDiagramTable(wind_speed_resolution=obj.wind_speeds,
                                 wind_angle_resolution=wind_angles,
                                 data=data)

    elif isinstance(obj, PolarDiagramPointcloud):
        sym_points = obj.points
        sym_points[:, 1] = 360 - sym_points[:, 1]
        return PolarDiagramPointcloud(points=np.vstack((obj.points, sym_points)))

    else:
        raise PolarDiagramException("Not yet implemented")


class PolarDiagram(ABC):
    """
    Methods
    ------
    pickling(pkl_path):
        Writes a PolarDiagram object to a .pkl-file
    Abstract Methods
    ----------------
    __str__()
    __repr__()
    wind_speeds
    wind_angles
    boat_speeds
    to_csv(csv_path)
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
        """Writes a PolarDiagram object to a .pkl-file"""
        with open(pkl_path, 'wb') as file:
            pickle.dump(self, file)

    @abstractmethod
    def polar_plot_slice(self, wind_speed, ax=None, **kwargs):
        pass

    @abstractmethod
    def flat_plot_slice(self, wind_speed, ax=None, **kwargs):
        pass

    @abstractmethod
    def polar_plot(self, wind_speed_range, ax=None, min_color=np.array([1, 0, 0]),
                   max_color=np.array([0, 1, 0]), **kwargs):
        pass

    @abstractmethod
    def flat_plot(self, wind_speed_range, ax=None, min_color=np.array([1, 0, 0]),
                  max_color=np.array([0, 1, 0]), **kwargs):
        pass

    @abstractmethod
    def plot_3d(self):
        pass

    @abstractmethod
    def plot_color_gradient(self, ax=None, min_color=np.array([1, 0, 0]),
                            max_color=np.array([0, 1, 0])):
        pass

    @abstractmethod
    def plot_convex_hull_slice(self, wind_speed, ax=None, **kwargs):
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
    __init__(wind_speed_resolution=None, wind_angle_resolution=None,
             data=None, tws=True, twa=True):
        Initializes a PolarDiagramTable object
    __str__():
    __repr__():
        Returns a string representation of the PolarDiagramTable instance
    __getitem__(wind_tup):
    wind_angles:
        Returns a read only verion of self._wind_angle_resolution
    wind_speeds:
        Returns a read only version of self._wind_speed_resolution
    boat_speeds:
        Returns a read only version of self._data
    to_csv(csv_path):
        Writes object to a .csv-file
    change_entries(data, wind_speeds=None, wind_angles=None, tws=True, twa=True):
        Changes entries in table
    """
    # V: In Arbeit
    def __init__(self, wind_speed_resolution=None, wind_angle_resolution=None,
                 data=None, tws=True, twa=True):
        """Initializes a PolarDiagramTable instance"""
        wind_dict = convert_wind({"wind_speed": wind_speed_resolution,
                                  "wind_angle": wind_angle_resolution}, tws, twa)
        self._resolution_wind_speed = speed_resolution(wind_dict["wind_speed"])
        self._resolution_wind_angle = angle_resolution(wind_dict["wind_angle"])
        rows = len(self._resolution_wind_angle)
        cols = len(self._resolution_wind_speed)
        if data is None:
            self._data = np.zeros(rows, cols)
        else:
            data = np.array(data)
            if data.ndim != 2:
                raise PolarDiagramException("Wrong dimension", data.ndim)
            if data.shape != (rows, cols):
                raise PolarDiagramException("Wrong shape", (rows, cols), data.shape)
            self._data = data

    # V: In Arbeit
    def __str__(self):
        """Returns a tabulate of the first and last 5 columns of the polar diagram table"""
        length = len(self.wind_angles)
        table = np.c_[self.wind_angles, self.boat_speeds[:, :5], np.array(["..."]*length), self.boat_speeds[:, -5:]]
        return tabulate(table, headers=["TWA \\ TWS"] + list(self.wind_speeds)[:5] +
                                       ["..."] + list(self.wind_speeds)[-5:])

    # V: Soweit in Ordnung
    def __repr__(self):
        """Returns a string representation of the PolarDiagramTable instance"""
        return f"PolarDiagramTable(wind_speed_resolution={self.wind_speeds}, " \
               f"wind_angle_resolution={self.wind_angles}, data={self.boat_speeds}"

    # V: In Arbeit
    def __getitem__(self, wind_tup):
        """"""
        wind_speed, wind_angle = wind_tup
        ind_c = get_indices(wind_speed, self.wind_speeds)
        ind_r = get_indices(wind_angle, self.wind_angles)
        return self.boat_speeds[ind_r, ind_c]

    @property
    def wind_angles(self):
        """Returns a read only version of self._resolution_wind_angle"""
        return self._resolution_wind_angle.copy()

    @property
    def wind_speeds(self):
        """Returns a read only version of self._resolution_wind_speed"""
        return self._resolution_wind_speed.copy()

    @property
    def boat_speeds(self):
        """Returns a read only version of self._data"""
        return self._data.copy()

    # V: Soweit in Ordnung
    def to_csv(self, csv_path):
        """Creates a .csv-file with the following format
            PolarDiagramTable
            Wind speed resolution:
            self.wind_speeds
            Wind angle resolution:
            self.wind_angles
            Boat speeds:
            self.boat_speeds

        and delimiter ','

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

    # V: In Arbeit.
    def change_entries(self, data, wind_speeds=None, wind_angles=None, tws=True, twa=True):
        """Changes specified entries in self._data"""
        data = np.array(data)
        wind_dict = convert_wind({"wind_speed": wind_speeds,
                                  "wind_angle": wind_angles}, tws, twa)
        wind_speeds = wind_dict["wind_speed"]
        wind_angles = wind_dict["wind_angle"]
        if wind_speeds is not None:
            speed_ind = get_indices(wind_speeds, self.wind_speeds)
            if wind_angles is not None:
                angle_ind = get_indices(wind_angles, self.wind_angles)
                mask = np.zeros(self.boat_speeds.shape, dtype=bool)
                for i in angle_ind:
                    for j in speed_ind:
                        mask[i, j] = True
                try:
                    self._data[mask] = data.flat
                except ValueError:
                    raise PolarDiagramException("Wrong shape", (len(angle_ind), len(speed_ind)),
                                                data.shape)
            else:
                try:
                    self._data[:, speed_ind] = data.reshape(-1, len(speed_ind))
                except ValueError:
                    raise PolarDiagramException("Wrong shape", (len(self.wind_angles), len(speed_ind)),
                                                data.shape)
        elif wind_angles is not None:
            angle_ind = get_indices(wind_angles, self.wind_angles)
            try:
                self._data[angle_ind, :] = data.reshape(len(angle_ind), -1)
            except ValueError:
                raise PolarDiagramException("Wrong shape", (len(angle_ind), len(self.wind_speeds)),
                                            data.shape)
        else:
            if data.shape != (len(self.wind_angles), len(self.wind_speeds)):
                raise PolarDiagramException("Wrong shape", (len(self.wind_angles), len(self.wind_speeds)),
                                            data.shape)
            self._data = data

    # V: Soweit in Ordnung
    def _get_slice_data(self, wind_speed):
        try:
            column = list(self.wind_speeds).index(wind_speed)
            return list(self.boat_speeds[:, column])
        except ValueError:
            raise PolarDiagramException("Not in resolution", wind_speed, self.wind_speeds)

    # V: Soweit in Ordnung
    def polar_plot_slice(self, wind_speed, ax=None, **kwargs):
        """"""
        wind_angles = list(np.deg2rad(self.wind_angles))
        boat_speeds = self._get_slice_data(wind_speed)
        return plot_polar(wind_angles, boat_speeds, ax, **kwargs)

    # V: Soweit in Ordnung
    def flat_plot_slice(self, wind_speed, ax=None, **kwargs):
        """"""
        wind_angles = self.wind_angles
        boat_speeds = self._get_slice_data(wind_speed)
        return plot_flat(wind_angles, boat_speeds, ax, **kwargs)

    # V: In Arbeit
    def polar_plot(self, wind_speed_range=None, ax=None, min_color=np.array([1, 0, 0]),
                   max_color=np.array([0, 1, 0]), **kwargs):
        """"""
        if wind_speed_range is None:
            wind_speed_range = self.wind_speeds

        boat_speeds_list = []
        for wind_speed in wind_speed_range:
            boat_speeds_list.append(self._get_slice_data(wind_speed))
        wind_angles_list = [list(np.deg2rad(self.wind_angles))] * len(boat_speeds_list)
        return plot_polar_range(wind_speed_range, wind_angles_list, boat_speeds_list,
                                ax, min_color, max_color, **kwargs)

    def flat_plot(self, wind_speed_range=None, ax=None, min_color=np.array([1, 0, 0]),
                  max_color=np.array([0, 1, 0]), **kwargs):
        """"""
        if wind_speed_range is None:
            wind_speed_range = self.wind_speeds

        boat_speeds_list = []
        for wind_speed in wind_speed_range:
            boat_speeds_list.append(self._get_slice_data(wind_speed))
        wind_angles_list = [list(self.wind_angles)]*len(boat_speeds_list)
        return flat_plot_range(wind_speed_range, wind_angles_list, boat_speeds_list,
                               ax, min_color, max_color, **kwargs)

    # V: In Arbeit
    def plot_3d(self):
        """"""
        pass

    # V: In Arbeit
    def plot_color_gradient(self, ax=None, min_color=np.array([1, 0, 0]),
                            max_color=np.array([0, 1, 0])):
        """"""
        wind_speeds, wind_angles = np.meshgrid(self.wind_speeds, self.wind_angles)
        wind_speeds = wind_speeds.reshape(-1,)
        wind_angles = wind_angles.reshape(-1,)
        boat_speeds = self.boat_speeds.reshape(-1,)
        return plot_color(wind_speeds, wind_angles, boat_speeds, ax, min_color, max_color)

    # V: Soweit in Ordnung
    def plot_convex_hull_slice(self, wind_speed, ax=None, **kwargs):
        """"""
        angles = list(np.deg2rad(self.wind_angles))
        speeds = self._get_slice_data(wind_speed)
        return plot_convex_hull(angles, speeds, **kwargs)

    # V: In Arbeit
    def plot_convex_hull_3d(self, **kwargs):
        """"""
        pass


class PolarDiagramCurve(PolarDiagram):
    """
    A class to represent, visualize and work with a polar performance diagram
    given by a fitted curve.

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
        Returns a string representation of the PolarDiagramCurve instance
    __call__(wind_speed, wind_angle):
    curve:
        Returns a read only version of self._f
    parameters:
        Returns a read only version of self._params
    to_csv(csv_path):
        Writes object to a .csv-file
    """

    # V: Noch nicht getestet
    def __init__(self, f, *params):
        """Initializes a PolarDiagramCurve object"""
        self._f = f
        self._params = params

    # V: In Arbeit
    def __str__(self):
        """"""
        return f"Function: {self.curve}\n Optimal parameters: {self.parameters}"

    # V: Noch nicht getestet
    def __repr__(self):
        """Returns a string representation of the PolarDiagramCurve instance"""
        return f"PolarDiagramCurve(f={self.curve}, params={self.parameters})"

    # V: Noch nicht getestet
    def __call__(self, wind_speed, wind_angle):
        return self.curve(np.column_stack((wind_speed, wind_angle)), self.parameters)

    @property
    def curve(self):
        """Returns a read only version of self._f"""
        return self._f

    @property
    def parameters(self):
        """Returns a read only version of self._params"""
        return self._params

    # V: Dummy property
    @property
    def wind_angles(self):
        return

    # V: Dummy property
    @property
    def wind_speeds(self):
        return

    # V: Dummy property
    @property
    def boat_speeds(self):
        return

    # V: Noch nicht getestet
    def to_csv(self, csv_path):
        """Creates a .csv-file with the following format
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

    # V: Noch nicht getestet
    def polar_plot_slice(self, wind_speed, ax=None, **kwargs):
        """"""
        wind_angles = list(np.deg2rad(np.linspace(0, 360, 1000)))
        boat_speeds = [self(wind_speed, wind_angle) for wind_angle in wind_angles]
        return plot_polar(wind_angles, boat_speeds, ax, **kwargs)

    # V: Noch nicht getestet
    def flat_plot_slice(self, wind_speed, ax=None, **kwargs):
        """"""
        wind_angles = list(np.linspace(0, 360, 1000))
        boat_speeds = [self(wind_speed, wind_angle) for wind_angle in wind_angles]
        return plot_polar(wind_angles, boat_speeds, ax, **kwargs)

    # V: In Arbeit
    def polar_plot(self, wind_speed_range=(0, 20), ax=None, min_color=np.array([1, 0, 0]),
                   max_color=np.array([0, 1, 0]), **kwargs):
        """"""
        wind_speed_lower, wind_speed_upper = wind_speed_range
        wind_speed_list = list(np.linspace(wind_speed_lower, wind_speed_upper, 50))
        boat_speed_list = []
        wind_angle_list = [np.linspace(0, 360, 1000)]

        return plot_polar_range(wind_speed_list, wind_angle_list, boat_speed_list, ax,
                                min_color, max_color, **kwargs)

    def flat_plot(self, wind_speed_range=(0, 20), ax=None, min_color=np.array([1, 0, 0]),
                  max_color=np.array([0, 1, 0]), **kwargs):
        """"""
        wind_speed_lower, wind_speed_upper = wind_speed_range
        wind_speed_list = list(np.linspace(wind_speed_lower, wind_speed_upper, 50))
        boat_speed_list = []
        wind_angle_list = []

        return flat_plot_range(wind_speed_list, wind_angle_list, boat_speed_list, ax,
                               min_color, max_color, **kwargs)

    # V: In Arbeit
    def plot_3d(self):
        """"""
        pass

    # V: In Arbeit
    def plot_color_gradient(self, ax=None, min_color=np.array([1, 0, 0]),
                            max_color=np.array([0, 1, 0])):
        """"""
        pass

    # V: Noch nicht getestet
    def plot_convex_hull_slice(self, wind_speed, ax=None, **kwargs):
        """"""
        wind_angles = list(np.deg2rad(np.linspace(0, 360, 1000)))
        boat_speeds = [self(wind_speed, wind_angle) for wind_angle in wind_angles]
        return plot_convex_hull(wind_angles, boat_speeds, ax, **kwargs)

    # V: In Arbeit
    def plot_convex_hull_3d(self):
        """"""
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
    __init__(points):
        Initializes a PolarDiagramPointcloud object
    __str__():
        Returns a table of all points in the point cloud"
    __repr__():
        Returns a string representation of the PolarDiagramPointcloud instance
    __getitem__(wind_tup):
    wind_speeds:
        Returns a list of all occuring wind speeds
    wind_angles:
        Returns a list of all occuring wind angles
    points:
        Returns a read only version of self.data
    to_csv:
        Writes instance to a .csv-file
    add_points(new_points):
        Appends given points to self._data
    """
    # V: In Arbeit
    def __init__(self, points=None, tws=True, twa=True):
        """Initializes a PolarDiagramPointcloud object"""
        if points is not None:
            if len(points[0]) != 3:
                raise PolarDiagramException("Wrong shape", (len(points[0]), "x"), (3, "x"))
            points = np.array(points)
            wind_dict = create_wind_dict(points)
            wind_dict = convert_wind(wind_dict, tws, twa)
            points = np.column_stack((wind_dict["wind_speed"], wind_dict["wind_angle"], points[:, 2]))
            self._data = points
        else:
            self._data = np.array([])

    # V: In Arbeit
    def __str__(self):
        """Returns a table of all points in the point cloud"""
        return tabulate(self.points, headers=["TWS", "TWA", "BSP"])

    # V: Soweit in Ordnung
    def __repr__(self):
        """Returns a string representation of the PolarDiagramPointcloud instance"""
        return f"PolarDiagramPointcloud(data={self.points})"

    # V: In Arbeit
    def __getitem__(self, wind_tup):
        """"""
        wind_speed, wind_angle = wind_tup
        return self.points[self.points[:, :1] == [wind_speed, wind_angle]]

    @property
    def wind_speeds(self):
        """Returns a list of all occuring wind speeds"""
        return list(dict.fromkeys(self.points[:, 0]))

    @property
    def wind_angles(self):
        """Returns a list of all occuring wind angles"""
        return list(dict.fromkeys(self.points[:, 1]))

    # V: Dummy property
    # V: Noch in Arbeit
    @property
    def boat_speeds(self):
        return self._data.copy()

    @property
    def points(self):
        """Returns a read only version of self._data"""
        return self._data.copy()

    # V: Soweit in Ordnung
    def to_csv(self, csv_path):
        """Creates a .csv-file with the following format
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

    # V: In Arbeit
    def add_points(self, new_points, tws=True, twa=True):
        """"""
        if len(new_points[0]) != 3:
            raise PolarDiagramException("Wrong shape", (len(new_points[0]), "x"), (3, "x"))

        wind_dict = create_wind_dict(new_points)
        wind_dict = convert_wind(wind_dict, tws, twa)
        new_points = np.column_stack(wind_dict["wind_speed"], wind_dict["wind_angle"], new_points[:, 2])
        self._data = np.row_stack((self.points, new_points))

    # V: In Arbeit
    def change_points(self):
        """"""
        pass

    # V: Soweit in Ordnung
    def polar_plot_slice(self, wind_speed, ax=None, **kwargs):
        """"""
        points = self.points[self.points[:, 0] == wind_speed][:, 1:]
        try:
            wind_angles = list(np.deg2rad(points[:, 0]))
            boat_speeds = points[:, 1]
        except (ValueError, IndexError):
            raise PolarDiagramException("No points found", wind_speed)
        return plot_polar(wind_angles, boat_speeds, ax, **kwargs)

    # V: Soweit in Ordnung
    def flat_plot_slice(self, wind_speed, ax=None, **kwargs):
        """"""
        points = self.points[self.points[:, 0] == wind_speed][:, 1:]
        try:
            wind_angles = points[:, 0]
            boat_speeds = points[:, 1]
        except (ValueError, IndexError):
            raise PolarDiagramException("No points found", wind_speed)
        return plot_flat(wind_angles, boat_speeds, ax, **kwargs)

    # V: In Arbeit
    def polar_plot(self, wind_speed_range=(0, np.inf), ax=None, min_color=np.array([1, 0, 0]),
                   max_color=np.array([0, 1, 0]), **kwargs):
        """"""
        wind_speed_list = []
        wind_angle_list = []
        boat_speed_list = []
        wind_speed_lower, wind_speed_upper = wind_speed_range
        for wind_speed in self.wind_speeds:
            if wind_speed_lower <= wind_speed <= wind_speed_upper:
                wind_speed_list.append(wind_speed)
                points = self.points[self.points[:, 0] == wind_speed][:, 1:]
                wind_angle_list.append(np.deg2rad(points[:, 0]))
                boat_speed_list.append(points[:, 1])

        return plot_polar_range(wind_speed_list, wind_angle_list, boat_speed_list,
                                ax, min_color, max_color, **kwargs)

    def flat_plot(self, wind_speed_range=(0, np.inf), ax=None, min_color=np.array([1, 0, 0]),
                  max_color=np.array([0, 1, 0]), **kwargs):
        """"""
        wind_speed_list = []
        wind_angle_list = []
        boat_speed_list = []
        wind_speed_lower, wind_speed_upper = wind_speed_range
        for wind_speed in self.wind_speeds:
            if wind_speed_lower <= wind_speed <= wind_speed_upper:
                wind_speed_list.append(wind_speed)
                points = self.points[self.points[:, 0] == wind_speed][:, 1:]
                wind_angle_list.append(points[:, 0])
                boat_speed_list.append(points[:, 1])

        return flat_plot_range(wind_speed_list, wind_angle_list, boat_speed_list,
                               ax, min_color, max_color, **kwargs)

    # V: In Arbeit
    def plot_3d(self):
        """"""
        pass

    # V: In Arbeit
    def plot_color_gradient(self, ax=None, min_color=np.array([1, 0, 0]), max_color=np.array([0, 1, 0])):
        """"""
        wind_speeds, wind_angles, boat_speeds = np.hsplit(self.points, 3)
        return plot_color(wind_speeds, wind_angles, boat_speeds, ax, min_color, max_color)

    # V: Soweit in Ordnung
    def plot_convex_hull_slice(self, wind_speed, ax=None, **kwargs):
        """"""
        points = self.points[self.points[:, 0] == wind_speed][:, 1:]
        try:
            angles = list(np.deg2rad(points[:, 0]))
            speeds = points[:, 1]
        except (ValueError, IndexError):
            raise PolarDiagramException("No points found", wind_speed)
        return plot_convex_hull(angles, speeds, ax, **kwargs)

    # V: In Arbeit
    def plot_convex_hull_3d(self):
        """"""
        pass
