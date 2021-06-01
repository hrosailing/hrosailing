# import logging
# import logging.handlers
import pickle
# import sys
from abc import ABC, abstractmethod
from tabulate import tabulate
from _utils import *
from _filereading import *
from _exceptions import *
from _plotting_functions import *


# logging.basicConfig(
# format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
# LOG_FILE = "PolarDiagram.log"

# polardiagram_logger = logging.getLogger(__name__)
# console_handler = logging.StreamHandler(sys.stdout)
# file_handler = logging.handlers.TimedRotatingFileHandler(
# LOG_FILE, when='midnight')
# polardiagram_logger.addHandler(console_handler)
# polardiagram_logger.setLevel(logging.DEBUG)


def to_csv(csv_path, obj):
    """Writes a PolarDiagram object to a .csv-file"""
    obj.to_csv(csv_path)


def from_csv(csv_path, fmt='hro', tw=True):
    """Creates a PolarDiagram instance from a .csv-file"""

    FMTS = ('hro', 'orc', 'array', 'opencpn')
    if fmt not in FMTS:
        raise PolarDiagramException(
            f"csv-format {fmt} not yet implemented")

    if fmt == 'hro':
        with open(csv_path, 'r', newline='') as file:
            csv_reader = csv.reader(file, delimiter=',', quotechar='"')
            first_row = next(csv_reader)[0]
            if (first_row not in
               ("PolarDiagramTable", "PolarDiagramPointcloud")):
                raise PolarDiagramException(
                    f"hro-format for {first_row} not yet implemented")

            if first_row == "PolarDiagramTable":
                ws_res, wa_res, data = read_table(csv_reader)
                return PolarDiagramTable(
                    wind_speed_resolution=ws_res,
                    wind_angle_resolution=wa_res,
                    data=data, tw=tw)

            data = read_pointcloud(csv_reader)
            return PolarDiagramPointcloud(
                points=data, tw=tw)

    ws_res, wa_res, data = read_extern_format(csv_path, fmt)
    return PolarDiagramTable(
        wind_speed_resolution=ws_res,
        wind_angle_resolution=wa_res,
        data=data, tw=tw)


def pickling(pkl_path, obj):
    """Writes a PolarDiagram object to a .pkl-file"""
    obj.pickling(pkl_path)


def depickling(pkl_path):
    """Creates a PolarDiagram object from a .pkl-file"""
    with open(pkl_path, 'rb') as file:
        return pickle.load(file)


# V: In Arbeit
# V: Noch nicht verwenden!
def convert(obj, convert_type):
    """Converts a PolarDiagram object to another type
    of PolarDiagram object"""
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
                point = boat_speeds[
                    boat_speeds[:1] == [wind_speeds[j], wind_angles[i]]][2]
                row.append(point)
            data.append(row)

        return PolarDiagramTable(
            wind_speed_resolution=wind_speeds,
            wind_angle_resolution=wind_angles,
            data=data)

    if convert_type is PolarDiagramPointcloud:
        for i in range(rows):
            for j in range(cols):
                data.append(
                    [wind_speeds[j],
                     wind_angles[i],
                     boat_speeds[i][j]])

        return PolarDiagramPointcloud(points=data)


def symmetric_polar_diagram(obj):
    """"""
    if not isinstance(obj, (PolarDiagramTable, PolarDiagramPointcloud)):
        raise PolarDiagramException(
            "functionality for obj not yet implemented")

    if isinstance(obj, PolarDiagramPointcloud):
        sym_points = obj.points
        sym_points[:, 1] = 360 - sym_points[:, 1]
        points = np.row_stack((obj.points, sym_points))
        return PolarDiagramPointcloud(points=points)

    wa_res = np.concatenate(
        [obj.wind_angles, 360-np.flip(obj.wind_angles)])
    data = np.row_stack(
        (obj.boat_speeds, np.flip(obj.boat_speeds, axis=0)))

    if 180 in obj.wind_angles:
        wa_res = list(wa_res)
        h = int(len(wa_res)/2)
        del wa_res[h]
        data = np.row_stack((data[:h, :], data[h+1:, :]))
    if 0 in obj.wind_angles:
        data = data[:-1, :]
        wa_res = wa_res[:-1]
    return PolarDiagramTable(
        wind_speed_resolution=obj.wind_speeds,
        wind_angle_resolution=wa_res,
        data=data, tw=True)


class PolarDiagram(ABC):
    """
    Methods
    ------
    pickling(pkl_path):
        Writes a PolarDiagram object to a .pkl-file

    Abstract Methods
    ----------------
    to_csv(csv_path)
    polar_plot_slice(wind_speed, ax=None, **kwargs)
    flat_plot_slice(wind_speed, ax=None, **kwargs)
    polar_plot(wind_speed_range, ax=None, min_color='g',
               max_color='r', **kwargs)
    plot_color_gradient(ax=None, min_color='g', max_color='r')
    plot_convex_hull_slice(wind_speed, ax=None, **kwargs)
    """
    def pickling(self, pkl_path):
        """Writes a PolarDiagram object to a .pkl-file"""
        with open(pkl_path, 'wb') as file:
            pickle.dump(self, file)

    @abstractmethod
    def to_csv(self, csv_path):
        pass

    @abstractmethod
    def polar_plot_slice(self, ws, ax=None, **plot_kw):
        pass

    @abstractmethod
    def flat_plot_slice(self, ws, ax=None, **plot_kw):
        pass

    @abstractmethod
    def polar_plot(self, wind_speed_range, ax=None,
                   colors=('green', 'red'), show_legend=True,
                   legend_kw=None, **plot_kw):
        pass

    @abstractmethod
    def flat_plot(self, wind_speed_range, ax=None,
                  colors=('green', 'red'), show_legend=True,
                  legend_kw=None, **plot_kw):
        pass

    @abstractmethod
    def plot_3d(self):
        pass

    @abstractmethod
    def plot_color_gradient(self, ax=None, colors=('green', 'red'),
                            marker=None, show_legend=True, **legend_kw):
        pass

    @abstractmethod
    def plot_convex_hull_slice(self, ws, ax=None, **plot_kw):
        pass

    @abstractmethod
    def plot_convex_hull_3d(self):
        pass


class PolarDiagramTable(PolarDiagram):
    """
    A class to represent, visualize and work with a polar
    performance diagram in form of a table.

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
    __getitem__(wind_tup):
    wind_angles:
        Returns a read only verion of self._wind_angle_resolution
    wind_speeds:
        Returns a read only version of self._wind_speed_resolution
    boat_speeds:
        Returns a read only version of self._data
    to_csv(csv_path):
        Writes object to a .csv-file
    change_entries(data, wind_speeds=None, wind_angles=None,
                   tws=True, twa=True):
        Changes entries in table
    """
    def __init__(self, wind_speed_resolution=None, wind_angle_resolution=None,
                 data=None, tw=True):

        w_dict = convert_wind(
            {"wind_speed": wind_speed_resolution,
             "wind_angle": wind_angle_resolution,
             "boat_speed": data},
            tw)

        self._resolution_wind_speed = speed_resolution(
            w_dict.get("wind_speed"))
        self._resolution_wind_angle = angle_resolution(
            w_dict.get("wind_angle"))

        rows = len(self._resolution_wind_angle)
        cols = len(self._resolution_wind_speed)
        if data is None:
            data = np.zeros(rows, cols)
        data = np.array(data)
        if data.ndim != 2:
            raise PolarDiagramException(
                "Expecting 2 dimensional array to be viewed as "
                "a Polar Diagram Tableau,")
        if data.shape != (rows, cols):
            try:
                data = data.reshape(rows, cols)
            except ValueError:
                raise PolarDiagramException(
                    "data couldn't be broadcasted to an array of" +
                    f"shape {(rows, cols)}")
        self._data = data

    def __str__(self):
        """Returns a tabulate of the polar diagram table"""

        if len(self.wind_speeds) <= 15:
            table = np.column_stack(
                (self.wind_angles, self.boat_speeds))
            headers = ["TWA \\ TWS"] + list(self.wind_speeds)
            return tabulate(table, headers=headers)

        length = len(self.wind_angles)
        table = np.column_stack(
            (self.wind_angles,
             self.boat_speeds[:, :5],
             np.array(["..."]*length),
             self.boat_speeds[:, -5:]))
        headers = (["TWA \\ TWS"]
                   + list(self.wind_speeds)[:5]
                   + ["..."]
                   + list(self.wind_speeds)[-5:])
        return tabulate(table, headers=headers)

    def __repr__(self):
        """"""
        return f"PolarDiagramTable(" \
               f"wind_speed_resolution={self.wind_speeds}, "\
               f"wind_angle_resolution={self.wind_angles}, " \
               f"data={self.boat_speeds})"

    def __getitem__(self, key):
        """"""
        ws, wa = key
        col = get_indices(ws, self.wind_speeds)
        row = get_indices(wa, self.wind_angles)
        return self.boat_speeds[row, col]

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

    def change_entries(self, new_data, ws=None, wa=None,
                       tw=True):
        """Changes specified entries in self._data"""
        new_data = np.array(new_data)
        w_dict = convert_wind(
            {"wind_speed": ws,
             "wind_angle": wa,
             "boat_speed": new_data},
            tw)
        ws = w_dict.get("wind_speed")
        wa = w_dict.get("wind_angle")
        ws_ind = get_indices(ws, self.wind_speeds)
        wa_ind = get_indices(wa, self.wind_angles)
        mask = np.zeros(self.boat_speeds.shape, dtype=bool)
        for i in wa_ind:
            for j in ws_ind:
                mask[i, j] = True
        try:
            new_data = new_data.reshape(len(wa_ind), len(ws_ind))
        except ValueError:
            raise PolarDiagramException(
                "new_data couldn't be broadcasted to an"
                f"array of shape {(len(wa_ind), len(ws_ind))}")
        self._data[mask] = new_data.flat

    def _get_slice_data(self, ws):
        ws_ind = get_indices(ws, self.wind_speeds)
        return self.boat_speeds[:, ws_ind]

    def _get_radians(self):
        return np.deg2rad(self.wind_angles)

    def polar_plot_slice(self, ws, ax=None, **plot_kw):
        """"""
        wa = list(self._get_radians())
        bsp = self._get_slice_data(ws)
        return plot_polar(wa, bsp, ax, **plot_kw)

    def flat_plot_slice(self, ws, ax=None, **plot_kw):
        """"""
        bsp = self._get_slice_data(ws)
        return plot_flat(self.wind_angles, bsp, ax, **plot_kw)

    def polar_plot(self, ws_range=None, ax=None,
                   colors=('green', 'red'), show_legend=True,
                   legend_kw=None, **plot_kw):
        """"""
        if ws_range is None:
            ws_range = self.wind_speeds

        bsp_list = list(self._get_slice_data(ws=ws_range).T)
        wa_list = [list(self._get_radians())] * len(bsp_list)

        return plot_polar_range(
            ws_range, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def flat_plot(self, ws_range=None, ax=None,
                  colors=('green', 'red'), show_legend=True,
                  legend_kw=None, **plot_kw):
        """"""
        if ws_range is None:
            ws_range = self.wind_speeds

        bsp_list = list(self._get_slice_data(ws=ws_range).T)
        wa_list = [list(self.wind_angles)] * len(bsp_list)
        return plot_flat_range(
            ws_range, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def plot_3d(self, ax=None, colors=('blue', 'blue')):
        """"""
        ws, wa = np.meshgrid(self.wind_speeds,
                             self._get_radians())
        bsp = self.boat_speeds
        bsp, wa = bsp * np.cos(wa), bsp * np.sin(wa)
        return plot_surface(ws, wa, bsp, ax, colors)

    def plot_color_gradient(
            self, ax=None, colors=('green', 'red'),
            marker=None, show_legend=True, **legend_kw):
        """"""
        ws, wa = np.meshgrid(self.wind_speeds,
                             self.wind_angles)
        ws = ws.reshape(-1,)
        wa = wa.reshape(-1,)
        bsp = self.boat_speeds.reshape(-1,)
        return plot_color(
            ws, wa, bsp,
            ax, colors, marker, show_legend, **legend_kw)

    def plot_convex_hull_slice(self, ws, ax=None,
                               **plot_kw):
        """"""
        wa = list(self._get_radians())
        bsp = self._get_slice_data(ws)
        return plot_convex_hull(wa, bsp, ax, **plot_kw)

    def plot_convex_hull_3d(self):
        """"""
        pass


class PolarDiagramCurve(PolarDiagram):
    """
    A class to represent, visualize and work with a
    polar performance diagram given by a fitted curve.

    ...

    Attributes
    ----------
    _f : ``function``

    _params : ``list``

    Methods
    -------
    __init__(f, *params):
        Initializes a PolarDiagramCurve object
    __repr__():
        Returns a string representation of the PolarDiagramCurve instance
    __call__(wind_speed, wind_angle):
        Returns self.curve([wind_speed, wind_angle], self.parameters)
    curve:
        Returns a read only version of self._f
    parameters:
        Returns a read only version of self._params
    to_csv(csv_path):
        Writes object to a .csv-file
    """

    def __init__(self, f, radians=False, *params):
        self._f = f
        self._params = params
        self._rad = radians

    def __repr__(self):
        """"""
        return f"PolarDiagramCurve(f={self.curve.__name__}, " \
               f"                  radians={self.radians}, " \
               f"                  {self.parameters})"

    def __call__(self, ws, wa):
        """Returns self.curve([wind_speed,wind_angle] self.parameters)"""
        return self.curve(np.column_stack((ws, wa)), *self.parameters)

    @property
    def curve(self):
        """Returns a read only version of self._f"""
        return self._f

    @property
    def parameters(self):
        """Returns a read only version of self._params"""
        return self._params

    @property
    def radians(self):
        """Returns a read only version of self._rad"""
        return self._rad

    # V: In Arbeit
    def to_csv(self, csv_path):
        """Creates a .csv-file with the following format
            PolarDiagramCurve
            Function: self.curve
            Radians: self.rad
            Parameters: self.parameters

        :param csv_path:
            Path to .csv-file
        :type csv_path: ``str``
        """
        with open(csv_path, 'w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=':', quotechar='"')
            csv_writer.writerow(["PolarDiagramCurve"])
            csv_writer.writerow(["Function"] + [self.curve.__name__])
            csv_writer.writerow(["Radians"] + [str(self.radians)])
            csv_writer.writerow(["Parameters"] + list(self.parameters))

    def _get_wind_angles(self):
        wa = np.linspace(0, 360, 1000)
        if self.radians:
            wa = np.deg2rad(wa)
        return wa

    def polar_plot_slice(self, ws, ax=None, **plot_kw):
        """"""
        wa = self._get_wind_angles()
        bsp = self(np.array([ws] * 1000), wa)
        return plot_polar(
            np.deg2rad(np.linspace(0, 360, 1000)), bsp,
            ax, **plot_kw)

    def flat_plot_slice(self, ws, ax=None, **plot_kw):
        """"""
        wa = self._get_wind_angles()
        bsp = self(np.array([ws] * 1000), wa)
        return plot_flat(
            np.linspace(0, 360, 1000), bsp, ax,
            **plot_kw)

    def polar_plot(self, ws_range=(0, 20, 5), ax=None,
                   colors=('green', 'red'), show_legend=True,
                   legend_kw=None, **plot_kw):
        """"""
        ws_lower, ws_upper, ws_step = ws_range
        ws_list = list(np.linspace(ws_lower, ws_upper, ws_step))
        wa = np.linspace(0, 360, 1000)
        wa_list = [np.deg2rad(wa)] * len(ws_list)
        if self.radians:
            wa = wa_list[0]

        bsp_list = []
        for ws in ws_list:
            bsp_list.append(self(np.array([ws] * 1000), wa))

        return plot_polar_range(
            ws_list, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def flat_plot(self, ws_range=(0, 20, 5), ax=None,
                  colors=('green', 'red'), show_legend=True,
                  legend_kw=None, **plot_kw):
        """"""
        ws_lower, ws_upper, ws_step = ws_range
        ws_list = list(np.linspace(ws_lower, ws_upper, ws_step))
        wa = np.linspace(0, 360, 1000)
        wa_list = [wa] * len(ws_list)
        if self.radians:
            wa = np.deg2rad(wa)

        bsp_list = []
        for ws in ws_list:
            bsp_list.append(self(np.array([ws] * 1000), wa))

        return plot_flat_range(
                    ws_list, wa_list, bsp_list,
                    ax, colors, show_legend, legend_kw, **plot_kw)

    def plot_3d(self, ws_range=(0, 20, 100), ax=None,
                colors=('blue', 'blue')):
        """"""
        ws_lower, ws_upper, ws_step = ws_range
        ws = np.linspace(ws_lower, ws_upper, ws_step)
        wa = self._get_wind_angles()

        ws_arr, wa_arr = np.meshgrid(ws, wa)
        bsp_arr = self(np.array([ws[0] * 1000]), wa)
        for wind_speed in ws[1:]:
            np.column_stack(
                (bsp_arr,
                 self(np.array([wind_speed]*1000), wa).reshape(-1, 1)))

        if self.radians:
            bsp, wa = bsp_arr * np.cos(wa_arr), bsp_arr * np.sin(wa_arr)
        else:
            wa_arr = np.deg2rad(wa_arr)
            bsp, wa = bsp_arr * np.cos(wa_arr), bsp_arr * np.sin(wa_arr)
        return plot_surface(ws_arr, wa, bsp, ax, colors)

    def plot_color_gradient(
            self, ws_range=(0, 20, 100), ax=None,
            colors=('green', 'red'), marker=None,
            show_legend=True, **legend_kw):
        """"""
        ws_lower, ws_upper, ws_step = ws_range
        ws, wa = np.meshgrid(
            np.linspace(ws_lower, ws_upper, ws_step),
            np.linspace(0, 360, 1000))
        ws = ws.reshape(-1,)
        wa = wa.reshape(-1,)

        if self.radians:
            bsp = self(ws, np.deg2rad(wa)).reshape(-1,)
        else:
            bsp = self(ws, wa).reshape(-1,)

        return plot_color(
            ws, wa, bsp,
            ax, colors, marker, show_legend, **legend_kw)

    def plot_convex_hull_slice(self, ws, ax=None, **plot_kw):
        """"""
        wa = self._get_wind_angles()
        bsp = self(np.array([ws] * 1000), wa)
        return plot_convex_hull(
            np.deg2rad(np.linspace(0, 360, 1000)), bsp,
            ax, **plot_kw)

    def plot_convex_hull_3d(self):
        """"""
        pass


class PolarDiagramPointcloud(PolarDiagram):
    """
    A class to represent, visualize and work with a
    polar performance diagram given by a point cloud

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

    def __init__(self, points=None, tw=True):
        if points is not None:
            points = np.array(points)
            if len(points[0]) != 3:
                try:
                    points = points.reshape(-1, 3)
                except ValueError:
                    raise PolarDiagramException(
                        "points could not be broadcasted "
                        "to an array of shape (n,3)")

            w_dict = convert_wind(
                {"wind_speed": points[:, 0],
                 "wind_angle": points[:, 1],
                 "boat_speed": points[:, 2]},
                tw)
            points = np.column_stack(
                (w_dict.get("wind_speed"),
                 w_dict.get("wind_angle"),
                 points[:, 2]))
            self._data = points
        else:
            self._data = np.array([])

    def __str__(self):
        return tabulate(self.points, headers=["TWS", "TWA", "BSP"])

    def __repr__(self):
        return f"PolarDiagramPointcloud(data={self.points})"

    @property
    def wind_speeds(self):
        return list(dict.fromkeys(self.points[:, 0]))

    @property
    def wind_angles(self):
        return list(dict.fromkeys(self.points[:, 1]))

    @property
    def points(self):
        return self._data.copy()

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
            csv_writer.writerow([
                "True wind speed ",
                "True wind angle ",
                "Boat speed "])
            csv_writer.writerows(self.points)

    def add_points(self, new_points, tw=True):
        """"""
        new_points = np.array(new_points)
        if len(new_points[0]) != 3:
            try:
                new_points = new_points.reshape(-1, 3)
            except ValueError:
                raise PolarDiagramException(
                    "new_points could not be broadcasted "
                    "to an array of shape (n,3)")

        w_dict = convert_wind(
            {"wind_speed": new_points[:, 0],
             "wind_angle": new_points[:, 1],
             "boat_speed": new_points[:, 2]},
            tw)
        new_points = np.column_stack(
            (w_dict.get("wind_speed"),
             w_dict.get("wind_angle"),
             new_points[:, 2]))

        if self.points == np.array([]):
            self._data = new_points
            return

        self._data = np.row_stack((self.points, new_points))

    def change_points(self):
        """"""
        pass

    def _get_slice_data(self, ws):
        points = self.points[self.points[:, 0] == ws][:, 1:]
        if points.size == 0:
            raise PolarDiagramException(
                f"No points with wind speed = {ws} found")
        return points[:, 0], points[:, 1]

    def polar_plot_slice(self, ws, ax=None, **plot_kw):
        """"""
        wa, bsp = self._get_slice_data(ws)
        wa = list(np.deg2rad(wa))
        return plot_polar(wa, bsp, ax, **plot_kw)

    def flat_plot_slice(self, ws, ax=None, **plot_kw):
        """"""
        wa, bsp = self._get_slice_data(ws)
        return plot_flat(wa, bsp, ax, **plot_kw)

    def _get_slices(self, ws_range):
        ws_list, wa_list, bsp_list = [], [], []
        ws_lower, ws_upper = ws_range
        for ws in sorted(self.wind_speeds):
            if ws > ws_upper:
                break

            if ws_lower <= ws:
                ws_list.append(ws)
                points = self.points[self.points[:, 0] == ws][:, 1:]
                wa_list.append(points[:, 0])
                bsp_list.append(points[:, 1])

        return ws_list, wa_list, bsp_list

    def polar_plot(self, ws_range=(0, np.inf), ax=None,
                   colors=('green', 'red'), show_legend=True,
                   legend_kw=None, **plot_kw):
        """"""
        ws_list, wa_list, bsp_list = \
            self._get_slices(ws_range)

        wa_list = [np.deg2rad(wa) for wa in wa_list]
        return plot_polar_range(
            ws_list, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def flat_plot(self, ws_range=(0, np.inf), ax=None,
                  colors=('green', 'red'), show_legend=True,
                  legend_kw=None, **plot_kw):
        """"""
        ws_list, wa_list, bsp_list = \
            self._get_slices(ws_range)

        return plot_flat_range(
            ws_list, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def plot_3d(self, ax=None, **plot_kw):
        """"""
        ws, wa, bsp = np.hsplit(self.points, 3)
        wa = np.deg2rad(wa)
        bsp, wa = bsp * np.cos(wa), bsp * np.sin(wa)
        return plot3d(ws, wa, bsp, ax, **plot_kw)

    def plot_color_gradient(self, ax=None, colors=('green', 'red'),
                            marker=None, show_legend=True, **legend_kw):
        """"""
        ws, wa, bsp = np.hsplit(self.points, 3)
        return plot_color(
            ws, wa, bsp,
            ax, colors, marker, show_legend, **legend_kw)

    def plot_convex_hull_slice(self, ws, ax=None, **plot_kw):
        """"""
        wa, bsp = self._get_slice_data(ws)
        wa = list(np.deg2rad(wa))
        return plot_convex_hull(wa, bsp, ax, **plot_kw)

    def plot_convex_hull_3d(self):
        """"""
        pass
