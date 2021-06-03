import logging
import logging.handlers
import pickle
import sys
from abc import ABC, abstractmethod
from tabulate import tabulate
from utils import *
from filereading import *
from exceptions import *
from plotting import *


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO)
LOG_FILE = "polardiagram.log"

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when='midnight')
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)


def to_csv(csv_path, obj):
    """Writes a PolarDiagram object to a .csv-file"""
    logger.debug(f"Function 'to_csv({csv_path}, {obj.__name__})' called")
    obj.to_csv(csv_path)


def from_csv(csv_path, fmt='hro', tw=True):
    """Creates a PolarDiagram instance from a .csv-file"""
    logger.debug(f"Function 'from_csv({csv_path}, {fmt}, {tw})' called")
    FMTS = ('hro', 'orc', 'array', 'opencpn')
    if fmt not in FMTS:
        logger.info(f"Error occured, when requesting format={fmt}")
        raise PolarDiagramException(
            f"csv-format {fmt} not yet implemented")

    if fmt == 'hro':
        try:
            with open(csv_path, 'r', newline='') as file:
                csv_reader = csv.reader(file, delimiter=',', quotechar='"')
                first_row = next(csv_reader)[0]
                if (first_row not in
                        ("PolarDiagramTable", "PolarDiagramPointcloud")):
                    logger.info(f"Error occured when reading {first_row}")
                    raise PolarDiagramException(
                        f"hro-format for {first_row} not yet implemented")

                if first_row == "PolarDiagramTable":
                    logger.debug("""Internal function
                                 'read_table(csv_reader)' called""")
                    ws_res, wa_res, data = read_table(csv_reader)
                    return PolarDiagramTable(
                        wind_speed_resolution=ws_res,
                        wind_angle_resolution=wa_res,
                        data=data, tw=tw)

                logger.debug("""Internal function 
                             'read_pointcloud(csv_reader)' called""")
                data = read_pointcloud(csv_reader)
                return PolarDiagramPointcloud(
                    points=data, tw=tw)
        except OSError:
            logger.info(f"Error occured when accessing file {csv_path}")
            raise FileReadingException(f"can't find/open/read {csv_path}")

    logger.debug(f"""Internal function 
                 'read_extern_format({csv_path}, {fmt})' called""")
    ws_res, wa_res, data = read_extern_format(csv_path, fmt)
    return PolarDiagramTable(
        wind_speed_resolution=ws_res,
        wind_angle_resolution=wa_res,
        data=data, tw=tw)


def pickling(pkl_path, obj):
    """Writes a PolarDiagram object to a .pkl-file"""
    logger.debug(f"Function 'pickling({pkl_path}, {obj.__name__})' called")
    try:
        obj.pickling(pkl_path)
    except OSError:
        logger.info(f"Error occured when writing to file {pkl_path}")
        raise FileReadingException(f"Can't write to {pkl_path}")


def depickling(pkl_path):
    """Creates a PolarDiagram object from a .pkl-file"""
    logger.debug(f"Function 'depickling({pkl_path})' called")
    try:
        with open(pkl_path, 'rb') as file:
            return pickle.load(file)
    except OSError:
        logger.info(f"Error occured when accessing file {pkl_path}")
        raise FileReadingException(f"Can't find/open/read {pkl_path}")


# V: In Arbeit
# V: Noch nicht verwenden!
def convert(obj, convert_type):
    pass


def symmetric_polar_diagram(obj):
    """"""
    logger.debug(f"Function 'symmetric_polar_diagram({obj.__name__}' called")
    if not isinstance(obj, (PolarDiagramTable, PolarDiagramPointcloud)):
        logger.info(f"Error occured when trying to symmetrize {obj.__name__}")
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
        logger.debug(f"Method 'PolarDiagram.pickling({pkl_path})' called")

        try:
            pickle.dump(self, pkl_path)
        except OSError:
            logger.info(f"Error occured when writing to {pkl_path}")
            raise FileReadingException(f"Can't write to {pkl_path}")

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
        logger.debug(f"""Dunder-Method
                     'PolarDiagramTable.__init__(wind_speed_resolution,
                     wind_angle_resolution, data, tw={tw}' called""")

        logger.debug(f"""Internal function
                     'utils.convert_wind(w_dict, tw={tw})' called""")
        w_dict = convert_wind(
            {"wind_speed": wind_speed_resolution,
             "wind_angle": wind_angle_resolution,
             "boat_speed": data},
            tw)

        logger.debug("""Internal function 
                     'utils.speed_resolution(ws_res)' called""")
        self._resolution_wind_speed = speed_resolution(
            w_dict.get("wind_speed"))
        logger.debug("""Internal function
                     'utils.angle_resolution(wa_res)' called""")
        self._resolution_wind_angle = angle_resolution(
            w_dict.get("wind_angle"))

        rows = len(self._resolution_wind_angle)
        cols = len(self._resolution_wind_speed)
        if data is None:
            data = np.zeros(rows, cols)
        data = np.asarray(data)
        if data.ndim != 2:
            logger.info(f"Error occured when checking data.ndim ")
            raise PolarDiagramException(
                "Expecting 2 dimensional array to be viewed as "
                "a Polar Diagram Tableau,")
        if data.shape != (rows, cols):
            try:
                data = data.reshape(rows, cols)
            except ValueError:
                logger.info(f"""Error occured when trying to broadcast
                            data to shape {(rows, cols)} """)
                raise PolarDiagramException(
                    "data couldn't be broadcasted to an array of" +
                    f"shape {(rows, cols)}")
        self._data = data

    def __str__(self):
        """Returns a tabulate of the polar diagram table"""
        logger.debug("""Dunder-method
                     'PolarDiagramTable.__str__()' called""")
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
        logger.debug("""Dunder-method
                     'PolarDiagramTable.__repr__()' called""")
        return f"PolarDiagramTable(" \
               f"wind_speed_resolution={self.wind_speeds}, "\
               f"wind_angle_resolution={self.wind_angles}, " \
               f"data={self.boat_speeds})"

    def __getitem__(self, key):
        """"""
        logger.debug(f"""Dunder-method
                     'PolarDiagramTable.__getitem__({key})' called""")
        ws, wa = key
        logger.debug("""Internal function
                     'utils.get_indices(ws, self.wind_speeds)' called""")
        col = get_indices(ws, self.wind_speeds)
        logger.debug("""Internal function
                     'utils.get_indices(wa, self.wind_angles)' called""")
        row = get_indices(wa, self.wind_angles)
        return self.boat_speeds[row, col]

    @property
    def wind_angles(self):
        """Returns a read only version of self._resolution_wind_angle"""
        logger.debug("Property 'PolarDiagramTable.wind_angles' called")
        return self._resolution_wind_angle.copy()

    @property
    def wind_speeds(self):
        """Returns a read only version of self._resolution_wind_speed"""
        logger.debug("Property 'PolarDiagramTable.wind_speeds' called")
        return self._resolution_wind_speed.copy()

    @property
    def boat_speeds(self):
        """Returns a read only version of self._data"""
        logger.debug("Property 'PolarDiagramTable.boat_speeds' called")
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
        logger.debug(f"Method 'PolarDiagramTable.to_csv({csv_path}' called")
        try:
            with open(csv_path, 'w', newline='') as file:
                csv_writer = csv.writer(file, delimiter=',', quotechar='"')
                csv_writer.writerow(["PolarDiagramTable"])
                csv_writer.writerow(["Wind speed resolution:"])
                csv_writer.writerow(self.wind_speeds)
                csv_writer.writerow(["Wind angle resolution:"])
                csv_writer.writerow(self.wind_angles)
                csv_writer.writerow(["Boat speeds:"])
                csv_writer.writerows(self.boat_speeds)
        except OSError:
            logger.info(f"Error occured when accessing file {csv_path}")
            raise FileReadingException(f"Can't write to {csv_path}")

    def change_entries(self, new_data, ws=None, wa=None,
                       tw=True):
        """Changes specified entries in self._data"""
        logger.debug(f"""Method 'PolarDiagramTable.change_entries(new_data, 
                     ws={ws}, wa={wa}, tw={tw}) called""")
        new_data = np.asarray(new_data)

        logger.debug(f"""Internal function
                     'utils.convert_wind(w_dict, tw={tw})' called""")
        w_dict = convert_wind(
            {"wind_speed": ws,
             "wind_angle": wa,
             "boat_speed": new_data},
            tw)
        ws = w_dict.get("wind_speed")
        wa = w_dict.get("wind_angle")
        logger.debug("""Internal function
                     'utils.get_indices(ws, self.wind_speeds)' called""")
        ws_ind = get_indices(ws, self.wind_speeds)
        logger.debug("""Internal function
                     'utils.get_indices(wa, self.wind_angles)' called""")
        wa_ind = get_indices(wa, self.wind_angles)
        mask = np.zeros(self.boat_speeds.shape, dtype=bool)
        for i in wa_ind:
            for j in ws_ind:
                mask[i, j] = True
        try:
            new_data = new_data.reshape(len(wa_ind), len(ws_ind))
        except ValueError:
            logger.info(f"""Error occured when trying to broadcast new_data
                        to shape {(len(wa_ind), len(ws_ind))}""")
            raise PolarDiagramException(
                "new_data couldn't be broadcasted to an"
                f"array of shape {(len(wa_ind), len(ws_ind))}")
        self._data[mask] = new_data.flat

    def _get_slice_data(self, ws):
        logger.debug(f"""Method 'PolarDiagramTable._get_slice_data(ws={ws})'
                     called""")

        logger.debug("""Internal function 
                     'utils.get_indices(ws, self.wind_speeds called""")
        ws_ind = get_indices(ws, self.wind_speeds)

        return self.boat_speeds[:, ws_ind]

    def _get_radians(self):
        logger.debug("Method 'PolarDiagramTable._get_radians()' called")

        return np.deg2rad(self.wind_angles)

    def polar_plot_slice(self, ws, ax=None, **plot_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramTable.polar_plot_slice(
                     ws={ws}, ax={ax}, plot_kw={plot_kw}' called""")

        wa = list(self._get_radians())
        bsp = self._get_slice_data(ws)

        logger.debug(f"""Internal function
                     'plotting.plot_polar(wa, bsp, ax, **plot_kw)' called""")
        return plot_polar(wa, bsp, ax, **plot_kw)

    def flat_plot_slice(self, ws, ax=None, **plot_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramTable.flat_plot_slice(
                     ws={ws}, ax={ax}, plot_kw={plot_kw}' called""")

        bsp = self._get_slice_data(ws)

        logger.debug(f"""Internal function
                     'plotting.plot_flat(self.wind_angles, bsp, ax, **plot_kw)'
                     called""")
        return plot_flat(self.wind_angles, bsp, ax, **plot_kw)

    def polar_plot(self, ws_range=None, ax=None,
                   colors=('green', 'red'), show_legend=True,
                   legend_kw=None, **plot_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramTablepolar_plot(
                     ws_range={ws_range}, ax={ax}, colors={colors}, 
                     show_legend={show_legend},legend_kw={legend_kw}, 
                     plot_kw={plot_kw}' called""")

        if ws_range is None:
            ws_range = self.wind_speeds

        bsp_list = list(self._get_slice_data(ws=ws_range).T)
        wa_list = [list(self._get_radians())] * len(bsp_list)

        logger.debug(f"""Internal function 
                     'plotting.plot_polar_range(ws_range, wa_list, bsp_list,
                     colors, show_legend, legend_kw, **plot_kw)' called""")
        return plot_polar_range(
            ws_range, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def flat_plot(self, ws_range=None, ax=None,
                  colors=('green', 'red'), show_legend=True,
                  legend_kw=None, **plot_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramTable.flat_plot(
                     ws_range={ws_range}, ax={ax}, colors={colors}, 
                     show_legend={show_legend}, legen_kw={legend_kw}, 
                     plot_kw={plot_kw})' called""")

        if ws_range is None:
            ws_range = self.wind_speeds

        bsp_list = list(self._get_slice_data(ws=ws_range).T)
        wa_list = [list(self.wind_angles)] * len(bsp_list)

        logger.debug(f"""Internal function 
                     'plotting.plot_flat_range(ws_range, wa_list, bsp_list,
                     colors, show_legend, legend_kw, **plot_kw) called'""")
        return plot_flat_range(
            ws_range, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def plot_3d(self, ax=None, colors=('blue', 'blue')):
        """"""
        logger.debug(f"""Method 'PolarDiagramTable.plot_3d(ax={ax}, 
                     colors={colors})' called""")

        ws, wa = np.meshgrid(self.wind_speeds,
                             self._get_radians())
        bsp = self.boat_speeds
        bsp, wa = bsp * np.cos(wa), bsp * np.sin(wa)

        logger.debug("""Internal function 
                     'plotting.plot_surface(ws, wa, bsp, ax, colors)'
                     called""")
        return plot_surface(ws, wa, bsp, ax, colors)

    def plot_color_gradient(
            self, ax=None, colors=('green', 'red'),
            marker=None, show_legend=True, **legend_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramTable.plot_color_gradient(
                     ax={ax}, colors={colors}, marker={marker}, 
                     show_legend={show_legend}, legend_kw={legend_kw})' 
                     called""")

        ws, wa = np.meshgrid(self.wind_speeds,
                             self.wind_angles)
        ws = ws.reshape(-1,)
        wa = wa.reshape(-1,)
        bsp = self.boat_speeds.reshape(-1,)

        logger.debug("""Internal function
                     'plotting.plot_color(ws, wa, bsp, ax, colors, 
                     marker, show_legend, **legend_kw)' called""")
        return plot_color(
            ws, wa, bsp,
            ax, colors, marker, show_legend, **legend_kw)

    def plot_convex_hull_slice(self, ws, ax=None,
                               **plot_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramTable.plot_convex_hull_slice(
                     ws={ws}, ax={ax}, plot_kw={plot_kw})' called""")

        wa = list(self._get_radians())
        bsp = self._get_slice_data(ws)

        logger.debug("""Internal function
                     'plotting.plot_convex_hull(wa, bsp, ax, **plot_kw)'
                     called""")
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
        logger.debug(f"""Dunder-Method 'PolarDiagramCurve.__init__(
                     f={f.__name__}, radians={radians}, *params={params})'
                     called""")
        self._f = f
        self._params = params
        self._rad = radians

    def __repr__(self):
        """"""
        logger.debug("Dunder-Method 'PolarDiagramCurve.__repr__()' called")
        return f"PolarDiagramCurve(f={self.curve.__name__}, " \
               f"                  radians={self.radians}, " \
               f"                  {self.parameters})"

    def __call__(self, ws, wa):
        """Returns self.curve([wind_speed,wind_angle] self.parameters)"""
        logger.debug(f"""Dunder-Method 'PolarDiagramCurve.__call__(
                     ws, wa)' called""")
        return self.curve(np.column_stack((ws, wa)), *self.parameters)

    @property
    def curve(self):
        """Returns a read only version of self._f"""
        logger.debug("""Property 'PolarDiagramCurve.curve' called""")
        return self._f

    @property
    def parameters(self):
        """Returns a read only version of self._params"""
        logger.debug("""Property 'PolarDiagramCurve.parameters' called""")
        return self._params

    @property
    def radians(self):
        """Returns a read only version of self._rad"""
        logger.debug("""Property 'PolarDiagramCurve.radians' called""")
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
        logger.debug(f"Method 'PolarDiagramCurve.to_csv({csv_path})' called")

        try:
            with open(csv_path, 'w', newline='') as file:
                csv_writer = csv.writer(file, delimiter=':', quotechar='"')
                csv_writer.writerow(["PolarDiagramCurve"])
                csv_writer.writerow(["Function"] + [self.curve.__name__])
                csv_writer.writerow(["Radians"] + [str(self.radians)])
                csv_writer.writerow(["Parameters"] + list(self.parameters))
        except OSError:
            logger.info(f"Error occured when accessing file {csv_path}")
            raise FileReadingException(f"Can't write to {csv_path}")

    def _get_wind_angles(self):
        logger.debug("Method 'PolarDiagramCurve._get_wind_angles()' called")

        wa = np.linspace(0, 360, 1000)
        if self.radians:
            wa = np.deg2rad(wa)
        return wa

    def polar_plot_slice(self, ws, ax=None, **plot_kw):
        """"""
        logger.debug(f"Method 'PolarDiagramCurve.polar_plot_slice(ws={ws},"
                     f"ax={ax}, **plot_kw={plot_kw})' called")

        wa = self._get_wind_angles()
        bsp = self(np.array([ws] * 1000), wa)

        logger.debug("""Internal function 'plotting.plot_polar(
                     np.deg2rad(np.linspace(0, 360, 1000), bsp, ax, 
                     **plot_kw)' called""")
        return plot_polar(
            np.deg2rad(np.linspace(0, 360, 1000)), bsp,
            ax, **plot_kw)

    def flat_plot_slice(self, ws, ax=None, **plot_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramCurve.flat_plot_slice(ws={ws},
                     ax={ax}, **plot_kw={plot_kw})' called""")

        wa = self._get_wind_angles()
        bsp = self(np.array([ws] * 1000), wa)

        logger.debug("""Internal function 'plotting.plot_flat(
                     np.linspace(0, 360, 1000), bsp, ax, **plot_kw)' 
                     called""")
        return plot_flat(
            np.linspace(0, 360, 1000), bsp, ax,
            **plot_kw)

    def polar_plot(self, ws_range=(0, 20, 5), ax=None,
                   colors=('green', 'red'), show_legend=True,
                   legend_kw=None, **plot_kw):
        """"""
        logger.debug(f"""Method PolarDiagramCurve.polar_plot(
                     ws_range={ws_range}, ax={ax}, colors={colors},
                     show_legend={show_legend}, legend_kw={legend_kw},
                     **plot_kw={plot_kw})' called""")

        ws_lower, ws_upper, ws_step = ws_range
        ws_list = list(np.linspace(ws_lower, ws_upper, ws_step))
        wa = np.linspace(0, 360, 1000)
        wa_list = [np.deg2rad(wa)] * len(ws_list)
        if self.radians:
            wa = wa_list[0]

        bsp_list = []
        for ws in ws_list:
            bsp_list.append(self(np.array([ws] * 1000), wa))

        logger.debug("""Internal function 'plotting.plot_polar_range(
                     ws_list, wa_list, bsp_list, ax, colors, show_legend,
                     legend_kw, **plot_kw)' called""")
        return plot_polar_range(
            ws_list, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def flat_plot(self, ws_range=(0, 20, 5), ax=None,
                  colors=('green', 'red'), show_legend=True,
                  legend_kw=None, **plot_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramCurve.flat_plot(
                     ws_range={ws_range}, ax={ax}, colors={colors},
                     show_legend={show_legend}, legend_kw={legend_kw},
                     **plot_kw={plot_kw})' called""")

        ws_lower, ws_upper, ws_step = ws_range
        ws_list = list(np.linspace(ws_lower, ws_upper, ws_step))
        wa = np.linspace(0, 360, 1000)
        wa_list = [wa] * len(ws_list)
        if self.radians:
            wa = np.deg2rad(wa)

        bsp_list = []
        for ws in ws_list:
            bsp_list.append(self(np.array([ws] * 1000), wa))

        logger.debug("""Internal function 'plotting.plot_flat_range(
                     ws_list, wa_list, bsp_list, ax, colors, show_legend,
                     legend_kw, **plot_kw)' called""")
        return plot_flat_range(
            ws_list, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def plot_3d(self, ws_range=(0, 20, 100), ax=None,
                colors=('blue', 'blue')):
        """"""
        logging.debug(f"""Method 'PolarDiagramCurve.plot_3d(
                      ws_range={ws_range}, ax={ax}, colors={colors})'
                      called""")

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

        logging.debug("""Internal function 'plotting.plot_surface(
                      ws_arr, wa, bsp, ax, colors)' called""")
        return plot_surface(ws_arr, wa, bsp, ax, colors)

    def plot_color_gradient(
            self, ws_range=(0, 20, 100), ax=None,
            colors=('green', 'red'), marker=None,
            show_legend=True, **legend_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramCurve.plot_color_gradient(
                     ws_range={ws_range}, ax={ax}, colors={colors}, 
                     marker={marker}, show_legend={show_legend},
                     **legend_kw={legend_kw})' called""")

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

        logger.debug("""Internal function 'plotting.plot_color(
                     ws, wa, bsp, ax, colors, marker, show_legend,
                     **legend_kw)' called""")
        return plot_color(
            ws, wa, bsp,
            ax, colors, marker, show_legend, **legend_kw)

    def plot_convex_hull_slice(self, ws, ax=None, **plot_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramCurve.plot_convex_hull_slice(
                     ws={ws}, ax={ax}, **plot_kw={plot_kw})' called""")

        wa = self._get_wind_angles()
        bsp = self(np.array([ws] * 1000), wa)

        logger.debug("""Internal function 'plotting.plot_convex_hull(
                     np.deg2rad(np.linspace(0, 360, 1000), bsp, ax
                     **plot_kw)' called""")
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
        logger.debug(f"""Dunder-Method 'PolarDiagramPointcloud.__init__(
                     points, tw={tw})' called""")

        if points is not None:
            points = np.asarray(points)
            if len(points[0]) != 3:
                try:
                    points = points.reshape(-1, 3)
                except ValueError:
                    logger.info("Error occured when trying to broadcast"
                                "points to shape (_, 3)")
                    raise PolarDiagramException(
                        "points could not be broadcasted "
                        "to an array of shape (n,3)")

            logger.debug("""Internal function 'utils.convert_wind(
                         w_dict, tw)' called""")
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
        logger.debug("""Dunder-Method 'PolarDiagramPointcloud.__str__()'
                     called""")

        return tabulate(self.points, headers=["TWS", "TWA", "BSP"])

    def __repr__(self):
        logger.debug("""Dunder-Method 'PolarDiagramPointcloud.__repr__()'
                     called""")

        return f"PolarDiagramPointcloud(data={self.points})"

    @property
    def wind_speeds(self):
        logger.debug("Property 'PolarDiagramPointcloud.wind_speeds' called")

        return list(dict.fromkeys(self.points[:, 0]))

    @property
    def wind_angles(self):
        logger.debug("Property 'PolarDiagramPointcloud.wind_angles' called")

        return list(dict.fromkeys(self.points[:, 1]))

    @property
    def points(self):
        logger.debug("Property 'PolarDiagramPointcloud.points' called")

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
        logger.debug(f"""Method 'PolarDiagramPointcloud.to_csv({csv_path})'
                     called""")
        try:
            with open(csv_path, 'w', newline='') as file:
                csv_writer = csv.writer(file, delimiter=',', quotechar='"')
                csv_writer.writerow(["PolarDiagramPointcloud"])
                csv_writer.writerow([
                    "True wind speed ",
                    "True wind angle ",
                    "Boat speed "])
                csv_writer.writerows(self.points)
        except OSError:
            logger.info(f"Error occured when accessing file {csv_path}")
            raise FileReadingException(f"Can't write to {csv_path}")

    def add_points(self, new_points, tw=True):
        """"""
        logger.debug(f"""Method 'PolarDiagramPointcloud.add_points(
                     new_points, tw={tw})' called""")

        new_points = np.asarray(new_points)
        if len(new_points[0]) != 3:
            try:
                new_points = new_points.reshape(-1, 3)
            except ValueError:
                logger.info("""Error occured when trying to broadcast 
                            new_points to shape (_, 3)""")
                raise PolarDiagramException(
                    "new_points could not be broadcasted "
                    "to an array of shape (n,3)")

        logger.debug(f"""Internal function 'utils.convert_wind(
                     w_dict, tw={tw})' called""")
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
        logger.debug(f"""Method 'PolarDiagramPointcloud._get_slice_data(
                     ws={ws})' called""")

        points = self.points[self.points[:, 0] == ws][:, 1:]
        if points.size == 0:
            logger.info("Error occured, when trying to get slice data")
            raise PolarDiagramException(
                f"No points with wind speed={ws} found")

        return points[:, 0], points[:, 1]

    def polar_plot_slice(self, ws, ax=None, **plot_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramPointcloud.polar_plot_slice(
                     ws={ws}, ax={ax}, **plot_kw={plot_kw})' called""")

        wa, bsp = self._get_slice_data(ws)
        wa = list(np.deg2rad(wa))

        logger.debug("""Internal function 'plotting.plot_polar(
                     wa, bsp, ax, **plot_kw)' called""")
        return plot_polar(wa, bsp, ax, **plot_kw)

    def flat_plot_slice(self, ws, ax=None, **plot_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramPointcloud.flat_plot_slice(
                     ws={ws}, ax={ax}, **plot_kw={plot_kw})' called""")

        wa, bsp = self._get_slice_data(ws)

        logger.debug("""Internal function 'plotting.plot_flat(
                     wa, bsp, ax, **plot_kw)' called""")
        return plot_flat(wa, bsp, ax, **plot_kw)

    def _get_slices(self, ws_range):
        logger.debug(f"""Method 'PolarDiagramPointcloud._get_slices(
                     ws_range={ws_range})' called""")

        ws_list, wa_list, bsp_list = [], [], []
        ws_lower, ws_upper = ws_range
        for ws in sorted(self.wind_speeds):
            if ws > ws_upper:
                break

            if ws_lower <= ws:
                ws_list.append(ws)
                points = self._get_slice_data(ws)
                wa_list.append(points[0])
                bsp_list.append(points[1])

        return ws_list, wa_list, bsp_list

    def polar_plot(self, ws_range=(0, np.inf), ax=None,
                   colors=('green', 'red'), show_legend=True,
                   legend_kw=None, **plot_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramPointcloud.polar_plot(
                     ws_range={ws_range}, ax={ax}, colors={colors},
                     show_legend={show_legend}, legend_kw={legend_kw},
                     **plot_kw={plot_kw})' called""")

        ws_list, wa_list, bsp_list = \
            self._get_slices(ws_range)
        wa_list = [np.deg2rad(wa) for wa in wa_list]

        logger.debug("""Internal function 'plotting.plot_polar_range(
                     ws_list, wa_list, bsp_list, ax, colors,
                     show_legend, legend_kw, **plot_kw)' called""")
        return plot_polar_range(
            ws_list, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def flat_plot(self, ws_range=(0, np.inf), ax=None,
                  colors=('green', 'red'), show_legend=True,
                  legend_kw=None, **plot_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramPointcloud.flat_plot(
                     ws_range={ws_range}, ax={ax}, colors={colors},
                     show_legend={show_legend}, legend_kw={legend_kw},
                     **plot_kw={plot_kw})' called""")

        ws_list, wa_list, bsp_list = \
            self._get_slices(ws_range)

        logger.debug("""Internal function 'plotting.plot_flat_range(
                     ws_list, wa_list, bsp_list, ax, colors,
                     show_legend, legend_kw, **plot_kw)' called""")
        return plot_flat_range(
            ws_list, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def plot_3d(self, ax=None, **plot_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramPointcloud.plot_3d(
                     ax={ax}, **plot_kw={plot_kw})' called""")

        ws, wa, bsp = np.hsplit(self.points, 3)
        wa = np.deg2rad(wa)
        bsp, wa = bsp * np.cos(wa), bsp * np.sin(wa)

        logger.debug("""Internal function 'plotting.plot3d(
                     ws, wa, bsp, ax, **plot_kw)' called""")
        return plot3d(ws, wa, bsp, ax, **plot_kw)

    def plot_color_gradient(self, ax=None, colors=('green', 'red'),
                            marker=None, show_legend=True, **legend_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramPointcloud.plot_color_gradient(
                     ax={ax}, colors={colors}, marker={marker}, 
                     show_legend={show_legend}, **legend_kw={legend_kw})'
                     called""")

        ws, wa, bsp = np.hsplit(self.points, 3)

        logger.debug("""Internal function 'plotting.plot_color(
                     ws, wa, bsp, ax, colors, marker, show_legend,
                     **legend_kw)' called""")
        return plot_color(
            ws, wa, bsp,
            ax, colors, marker, show_legend, **legend_kw)

    def plot_convex_hull_slice(self, ws, ax=None, **plot_kw):
        """"""
        logger.debug(f"""Method 'PolarDiagramPointcloud.plot_convex_hull_slice(
                     ws={ws}, ax={ax}, **plot_kw={plot_kw})' called""")

        wa, bsp = self._get_slice_data(ws)
        wa = list(np.deg2rad(wa))

        logger.debug("""Internal function 'plotting.plot_convex_hull(
                     wa, bsp, ax, **plot_kw)' called""")
        return plot_convex_hull(wa, bsp, ax, **plot_kw)

    def plot_convex_hull_3d(self):
        """"""
        pass
