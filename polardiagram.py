"""
Classes to represent polar diagrams in various different forms
as well as small functions to save / load PolarDiagram-objects to files
in different forms and functions to manipulate PolarDiagram-objects
"""

# Author: Valentin F. Dannenberg / Ente

import pickle

from abc import ABC, abstractmethod
from tabulate import tabulate

from exceptions import *
from filereading import *
from plotting import *
from utils import *

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='logging/polardiagram.log')
LOG_FILE = "logging/polardiagram.log"

logger = logging.getLogger(__name__)
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when='midnight')
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


def to_csv(csv_path, obj):
    """Calls the to_csv-method of
    the PolarDiagram instance

    Parameters
    ----------
    csv_path : string
        Path where a .csv-file is located or
        where a new .csv file will be created
    obj : PolarDiagram
        PolarDiagram instance which will be
        written to .csv file

    Function raises an exception
    if file can't be written to
    """
    logger.info(f"Function 'to_csv({csv_path}, {obj.__name__})' called")
    obj.to_csv(csv_path)


def from_csv(csv_path, fmt='hro', tw=True):
    """Reads a .csv file and
    returns the PolarDiagram
    instance contained in it

    Parameters
    ----------
    csv_path : string
        Path to a .csv file
        which will be read
    fmt : string
        The "format" of the .csv file.
        Supported inputs are:
                'hro' -> format created by
                the to_csv-method of the
                PolarDiagram class
                'orc' -> format found at
                `ORC <https://jieter.github.io/orc-data/site/>`_
                'opencpn' -> format created by
                `OpenCPN Polar Plugin
                <https://opencpn.org/OpenCPN/plugins/polar.html>`_
                'array'
    tw : bool
        Specifies if wind
        data in file should
        be viewed as true wind

        Defaults to True

    Returns
    -------
    polar_diagram : PolarDiagram
        PolarDiagram instance
        saved in .csv file

    Function raises an exception
    if an unknown format was
    specified, or file can't be
    read
    """
    logger.info(f"Function 'from_csv({csv_path}, {fmt}, {tw})' called")
    FMTS = ('hro', 'orc', 'array', 'opencpn')
    if fmt not in FMTS:
        logger.info(f"Error occured, when requesting fmt")
        raise PolarDiagramException(
            f"csv-format {fmt} not yet implemented")

    if fmt == 'hro':
        try:
            with open(csv_path, 'r', newline='') as file:
                csv_reader = csv.reader(file, delimiter=',', quotechar='"')
                first_row = next(csv_reader)[0]
                if (first_row not in
                        ("PolarDiagramTable", "PolarDiagramPointcloud")):
                    logger.error(f"Error occured when reading {first_row}")
                    raise PolarDiagramException(
                        f"hro-format for {first_row} not yet implemented")

                if first_row == "PolarDiagramTable":
                    logger.info("""Internal function
                                 'read_table(csv_reader)' called""")
                    ws_res, wa_res, data = read_table(csv_reader)
                    return PolarDiagramTable(
                        ws_res=ws_res, wa_res=wa_res,
                        data=data, tw=tw)

                logger.info("""Internal function 
                             'read_pointcloud(csv_reader)' called""")
                data = read_pointcloud(csv_reader)
                return PolarDiagramPointcloud(
                    points=data, tw=tw)
        except OSError:
            logger.info(f"Error occured when accessing file {csv_path}")
            raise FileReadingException(f"can't find/open/read {csv_path}")

    logger.info(f"""Internal function 
                 'read_extern_format({csv_path}, {fmt})' called""")
    ws_res, wa_res, data = read_extern_format(csv_path, fmt)
    return PolarDiagramTable(
        ws_res=ws_res, wa_res=wa_res,
        data=data, tw=tw)


def pickling(pkl_path, obj):
    """Calls the pickling-method
    of the PolarDiagram instance

    Parameters
    ----------
    pkl_path : string
        Path where a .pkl file is
        located or where a new
        .pkl file will be created
    obj : PolarDiagram
        PolarDiagram instance which
        will be written to .csv file

    Function raises an exception
    if file can't be written to
    """
    logger.info(f"Function 'pickling({pkl_path}, {obj.__name__})' called")
    obj.pickling(pkl_path)


def depickling(pkl_path):
    """Reads a .pkl file and
    returns the PolarDiagram
    instance contained in it.

    Parameters
    ----------
    pkl_path : string
        Path a .pkl file
        which will be read

    Returns
    -------
    _ : PolarDiagram
        PolarDiagram instance
        saved in .pkl file

    Function raises an exception
    if file can't be read
    """
    logger.info(f"Function 'depickling({pkl_path})' called")
    try:
        with open(pkl_path, 'rb') as file:
            return pickle.load(file)
    except OSError:
        logger.info(f"Error occured when accessing file {pkl_path}")
        raise FileReadingException(f"Can't find/open/read {pkl_path}")


# V: In Arbeit
def convert(obj, convert_type):
    pass


def symmetric_polar_diagram(obj):
    """Symmetrize a PolarDiagram
    instance, meaning for a
    datapoint with wind speed w,
    wind angle phi and
    boat speed s,  a new
    datapoint with wind speed w,
    wind angle 360 - phi and
    boat speed s will be added

    Parameters
    ----------
    obj : PolarDiagram
        PolarDiagram instance
        which will be
        symmetrized

    Returns
    -------
    _ : PolarDiagram
        "symmetrized" version
        of input

    Function raises an exception
    if obj is not of type
    PolarDiagramTable or
    PolarDiagramPointcloud
    """
    logger.info(f"Function 'symmetric_polar_diagram({obj.__name__}' called")
    if not isinstance(obj, (PolarDiagramTable, PolarDiagramPointcloud)):
        logger.error(f"Error occured when checking type of {obj.__name__}")
        raise PolarDiagramException(
            "functionality for obj not yet implemented")

    if isinstance(obj, PolarDiagramPointcloud):
        sym_points = obj.points
        if not sym_points.size:
            logger.error(f"Error occured when trying to"
                         f"symmetrize {obj.__name__}")
            raise PolarDiagramException(f"{obj.__name__} doesn't"
                                        f"contain any points")

        sym_points[:, 1] = 360 - sym_points[:, 1]
        points = np.row_stack((obj.points, sym_points))
        return PolarDiagramPointcloud(points=points)

    wa_res = np.concatenate(
        [obj.wind_angles, 360 - np.flip(obj.wind_angles)])
    data = np.row_stack(
        (obj.boat_speeds, np.flip(obj.boat_speeds, axis=0)))

    # deleting multiple 180° and 0°
    # occurences in the table
    if 180 in obj.wind_angles:
        wa_res = list(wa_res)
        h = int(len(wa_res) / 2)
        del wa_res[h]
        data = np.row_stack((data[:h, :], data[h + 1:, :]))
    if 0 in obj.wind_angles:
        data = data[:-1, :]
        wa_res = wa_res[:-1]

    return PolarDiagramTable(
        ws_res=obj.wind_speeds,
        wa_res=wa_res,
        data=data, tw=True)


class PolarDiagram(ABC):
    """Abstract Base Class
    for all polardiagram classes

    Methods
    -------
    pickling(pkl_path)
        Writes PolarDiagram
        instance to a .pkl file

    Abstract Methods
    ----------------
    to_csv(self, csv_path)
    polar_plot_slice(self, ws,
                     ax=None,
                     **plot_kw)
    flat_plot_slice(self, ws,
                    ax=None,
                    **plot_kw)
    polar_plot(self, ws_range,
               ax=None,
               colors=('green', 'red'),
               show_legend=True,
               legend_kw=None,
               **plot_kw)
    flat_plot(self, ws_range,
              ax=None,
              colors=('green', 'red'),
              show_legend=True,
              legend_kw=None,
              **plot_kw)
    plot_3d(self, ax=None,
            **plot_kw)
    plot_color_gradient(self, ax=None,
                        colors=('green', 'red'),
                        marker=None,
                        show_legend=True,
                        **legend_kw)
    plot_convex_hull_slice(self, ws,
                           ax=None,
                           **plot_kw)
    """

    def pickling(self, pkl_path):
        """Writes PolarDiagram
        instance to a .pkl file

        Parameters
        ----------
        pkl_path: string
            Path where a .pkl file is
            located or where a new
            .pkl file will be created

        Function raises an exception
        if file can't be written to
        """
        logger.info(f"Method 'PolarDiagram.pickling({pkl_path})' called")

        try:
            pickle.dump(self, pkl_path)
        except OSError:
            logger.error(f"Error occured when writing to {pkl_path}")
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
    A class to represent,
    visualize and work with
    a polar diagram in the
    form of a table.

    Parameters
    ----------
    ws_res : Iterable or int or float, optional
        Wind speeds that will
        correspond to the
        columns of the table.

        Can either be a sequence
        of length cdim or a
        number

        If a number num is passed,
        numpy.arange(num, 40, num)
        will be assigned to ws_res

        If nothing is passed,
        it will default to
        numpy.arange(2, 42, 2)

    wa_res : Iterable or int or float, optional
        Wind angles that will
        correspond to the
        columns of the table.

        Can either be sequence
        of length rdim or a
        number

        If a number num is passed,
        numpy.arange(num, 360, num)
        will be assigned to wa_res

        If nothing is passed,
        it will default to
        numpy.arange(0, 360, 5)

    data : array_like, optional
        Sequence of corresponding
        boat speeds, should be
        broadcastable to the
        shape (rdim, cdim)

        If nothing is passed
        it will default to
        numpy.zeros((rdim, cdim))

    tw : bool, optional
        Specifies if the
        given wind data should
        be viewed as true wind

        If False, wind data
        will be converted
        to true wind

        Defaults to True

    Raises an exception if
    data can't be broadcasted
    to a fitting shape or
    is of a wrong dimension

    Methods
    -------
    wind_speeds
        Returns a read only version
        of self._resolution_wind_speed
    wind_angles
        Returns a read only version
        of self._resolution_wind_angle
    boat_speeds
        Returns a read only version
        of self._data
    to_csv(csv_path)
        Creates a .csv-file with
        delimiter ',' and the
        following format:
            PolarDiagramTable
            Wind speed resolution:
            self.wind_speeds
            Wind angle resolution:
            self.wind_angles
            Boat speeds:
            self.boat_speeds
    change_entries(data, ws=None,
                   wa=None, tw=True)
        Changes specified entries
        in the table
    polar_plot_slice(ws, ax=None,
                     **plot_kw)
        Creates a polar plot of a
        given slice (column) of the
        polar diagram
    flat_plot_slice(ws, ax=None,
                    **plot_kw)
        Creates a cartesian plot
        of a given slice (column)
        of the polar diagram
    polar_plot(ws_range=None, ax=None,
               colors=('green', 'red'),
               show_legend=True,
               legend_kw=None,
               **plot_kw)
        Creates a polar plot
        of multiple slices (columns)
        of the polar diagram
    flat_plot(ws_range=None, ax=None,
              colors=('green', 'red'),
              show_legend=True,
              legend_kw=None,
              **plot_kw)
        Creates a cartesian plot
        of multiple slices (columns)
        of the polar diagram
    plot_3d(ax=None,
            colors=('blue', 'blue'))
        Creates a 3d plot
        of the polar diagram
    plot_color_gradient(ax=None,
                        colors=('green', 'red'),
                        marker=None,
                        show_legend=True,
                        **legend_kw)
        Creates a 'wind speed
        vs. wind angle' color gradient
        plot of the polar diagram
        with respect to the
        respective boat speeds
    plot_convex_hull_slice(ws, ax=None,
                           **plot_kw)
        Computes the convex
        hull of a slice (column)
        of the polar diagram
        and creates a polar plot
        of it
    """

    def __init__(self, ws_res=None, wa_res=None,
                 data=None, tw=True):
        logger.info(f"""Class 'PolarDiagramTable(ws_res,
                     wa_res, data, tw={tw})' called""")

        logger.info(f"""Internal function
                     'utils.convert_wind(w_dict, tw={tw})' called""")
        w_dict = convert_wind(
            {"wind_speed": ws_res,
             "wind_angle": wa_res,
             "boat_speed": data},
            tw)

        logger.info("""Internal function 
                     'utils.speed_resolution(ws_res)' called""")
        self._resolution_wind_speed = speed_resolution(
            w_dict.get("wind_speed"))
        logger.info("""Internal function
                     'utils.angle_resolution(wa_res)' called""")
        self._resolution_wind_angle = angle_resolution(
            w_dict.get("wind_angle"))

        rows = len(self._resolution_wind_angle)
        cols = len(self._resolution_wind_speed)
        if data is None:
            data = np.zeros(rows, cols)
        data = np.asarray(data)
        if data.ndim != 2:
            logger.error(f"Error occured when checking data.ndim ")
            raise PolarDiagramException(
                "Expecting 2 dimensional array to be viewed as "
                "a Polar Diagram Tableau,")
        if data.shape != (rows, cols):
            try:
                data = data.reshape(rows, cols)
            except ValueError:
                logger.error(f"""Error occured when trying to broadcast
                            data to shape {(rows, cols)} """)
                raise PolarDiagramException(
                    "data couldn't be broadcasted to an array of" +
                    f"shape {(rows, cols)}")
        self._data = data

    def __str__(self):
        logger.info("""Dunder-method
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
             np.array(["..."] * length),
             self.boat_speeds[:, -5:]))
        headers = (["TWA \\ TWS"]
                   + list(self.wind_speeds)[:5]
                   + ["..."]
                   + list(self.wind_speeds)[-5:])
        return tabulate(table, headers=headers)

    def __repr__(self):
        logger.info("""Dunder-method
                     'PolarDiagramTable.__repr__()' called""")
        return f"PolarDiagramTable(" \
               f"wind_speed_resolution={self.wind_speeds}, " \
               f"wind_angle_resolution={self.wind_angles}, " \
               f"data={self.boat_speeds})"

    def __getitem__(self, key):
        logger.info(f"""Dunder-method
                     'PolarDiagramTable.__getitem__({key})' called""")
        ws, wa = key
        logger.info("""Internal function
                     'utils.get_indices(ws, self.wind_speeds)' called""")
        col = get_indices(ws, self.wind_speeds)
        logger.info("""Internal function
                     'utils.get_indices(wa, self.wind_angles)' called""")
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
        """Creates a .csv file with
        delimiter ',' and the
        following format:
            PolarDiagramTable
            Wind speed resolution:
            self.wind_speeds
            Wind angle resolution:
            self.wind_angles
            Boat speeds:
            self.boat_speeds

        Parameters
        ----------
        csv_path : string
            Path where a .csv file is
            located or where a new
            .csv file will be created

        Function raises an exception
        if file can't be written to
        """
        logger.info(f"Method 'PolarDiagramTable.to_csv({csv_path}' called")
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
            logger.error(f"Error occured when accessing file {csv_path}")
            raise FileReadingException(f"Can't write to {csv_path}")

    def change_entries(self, new_data, ws=None, wa=None):
        """Changes specified entries in the table

        Parameters
        ----------
        new_data: array_like
            Sequence containing the
            new data to be inserted
            in the specified entries

        ws: Iterable or int or float, optional
            Element(s) of self.wind_speeds,
            specifying the columns, where
            new data will be inserted

            If nothing is passed it will
            default to self.wind_speeds

        wa: Iterable or int or float, optional
            Element(s) of self.wind_angles,
            specifiying the rows, where
            new data will be inserted

            If nothing is passed it will
            default to self.wind_angles


        Function raises an exception:
            If ws is not contained
            in self.wind_speeds

            If wa is not contained
            in self.wind_angles

            If new_data can't be
            broadcasted to a
            fitting shape
        """
        logger.info(f"""Method 'PolarDiagramTable.change_entries(new_data, 
                     ws, wa) called""")
        new_data = np.asarray(new_data)

        logger.info("""Internal function
                     'utils.get_indices(ws, self.wind_speeds)' called""")
        ws_ind = get_indices(ws, self.wind_speeds)
        logger.info("""Internal function
                     'utils.get_indices(wa, self.wind_angles)' called""")
        wa_ind = get_indices(wa, self.wind_angles)
        mask = np.zeros(self.boat_speeds.shape, dtype=bool)
        for i in wa_ind:
            for j in ws_ind:
                mask[i, j] = True
        try:
            new_data = new_data.reshape(len(wa_ind), len(ws_ind))
        except ValueError:
            logger.error(f"""Error occured when trying 
                        to broadcast new_data to shape 
                        {(len(wa_ind), len(ws_ind))}""")
            raise PolarDiagramException(
                f"""new_data couldn't be broadcasted to an
                array of shape {(len(wa_ind), len(ws_ind))}""")
        self._data[mask] = new_data.flat

    def _get_slice_data(self, ws):
        ws_ind = get_indices(ws, self.wind_speeds)

        return self.boat_speeds[:, ws_ind]

    def _get_radians(self):
        return np.deg2rad(self.wind_angles)

    def polar_plot_slice(self, ws, ax=None, **plot_kw):
        """Creates a polar plot of a
        given slice (column) of the
        polar diagram

        Parameters
        ----------
        ws : int or float
            Slice (column) of the polar
            diagram, given as an element
            of self.wind_speeds

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot


        Function raises an exception
        if ws is not an element
        of self.wind_speeds
        """
        logger.info(f"""Method 'PolarDiagramTable.polar_plot_slice(
                     ws={ws}, ax={ax}, plot_kw={plot_kw}' called""")

        wa = list(self._get_radians())
        bsp = self._get_slice_data(ws)

        logger.info(f"""Internal function
                     'plotting.plot_polar(wa, bsp, ax, **plot_kw)' called""")
        return plot_polar(wa, bsp, ax, **plot_kw)

    def flat_plot_slice(self, ws, ax=None, **plot_kw):
        """Creates a cartesian plot
        of a given slice (column)
        of the polar diagram

        Parameters
        ----------
        ws : int or float
            Slice (column) of the polar
            diagram, given as an element
            of self.wind_speeds

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot

        Function raises an exception
        if ws is not in the wind
        speed resolution.
        """
        logger.info(f"""Method 'PolarDiagramTable.flat_plot_slice(
                     ws={ws}, ax={ax}, plot_kw={plot_kw}' called""")

        bsp = self._get_slice_data(ws)
        wa = self.wind_angles

        logger.info(f"""Internal function
                     'plotting.plot_flat(self.wind_angles, 
                     bsp, ax, **plot_kw)' called""")
        return plot_flat(wa, bsp, ax, **plot_kw)

    def polar_plot(self, ws_range=None, ax=None,
                   colors=('green', 'red'), show_legend=True,
                   legend_kw=None, **plot_kw):
        """Creates a polar plot
        of multiple slices (columns)
        of the polar diagram

        Parameters
        ----------
        ws_range : Iterable, optional
            Slices (columns) of the
            polar diagram table,
            given as an Iterable
            of elements of
            self.wind_speeds.

            If nothing it passed,
            it will default to
            self.Wind_speeds

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        colors : tuple, optional
            Specifies the colors to
            be used for the different
            slices. There are four
            options
                If as many or more
                colors as slices
                are passed,
                each slice will
                be plotted in the
                specified color

                Otherwise if
                exactly 2 colors
                are passed, the
                slices will be
                plotted with a
                color gradient
                consiting of the
                two colors

                If more than 2
                colors are passed,
                either the first
                n_color slices will
                be plotted in the
                specified colors,
                and the rest will
                be plotted in the
                default color 'blue',
                or one can specify
                certain slices to be
                plotted in a certain
                color by passing a
                tuple of (ws, color)
                pairs

                Defaults to the tuple
                ('green', 'red')

        show_legend : bool, optional
            Specifies wether or not
            a legend will be shown
            next to the plot

            The type of legend depends
            on the color options:
            If the slices are plotted
            with a color gradient,
            a matplotlib.colorbar.Colorbar
            object will be created
            and assigned to ax

            Otherwise a
            matplotlib.legend.Legend
            will be created and
            assigned to ax

            Default to 'True'


        legend_kw : dict, optional
            Keyword arguments to be
            passed to either the
            matplotlib.colorbar.Colorbar
            or matplotlib.legend.Legend
            classes to change position
            and appearence of the legend.

            Will only be used if
            'show_legend=True'

            Defaults to an empty
            dictionary

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot

        Function raises an exception
        if at least one element
        of ws_range is not in
        the wind speed resolution
        """
        logger.info(f"""Method 'PolarDiagramTablepolar_plot(
                     ws_range={ws_range}, ax={ax}, colors={colors}, 
                     show_legend={show_legend},legend_kw={legend_kw}, 
                     plot_kw={plot_kw}' called""")

        if ws_range is None:
            ws_range = self.wind_speeds

        if isinstance(ws_range, np.ndarray):
            if not ws_range.size:
                logger.error("")
                raise PolarDiagramException("")
        elif not ws_range:
            logger.error("")
            raise PolarDiagramException("")

        bsp_list = list(self._get_slice_data(ws=ws_range).T)
        wa_list = [list(self._get_radians())] * len(bsp_list)

        logger.info(f"""Internal function 
                     'plotting.plot_polar_range(ws_range, 
                     wa_list, bsp_list, colors, show_legend, 
                     legend_kw, **plot_kw)' called""")
        return plot_polar_range(
            ws_range, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def flat_plot(self, ws_range=None, ax=None,
                  colors=('green', 'red'), show_legend=True,
                  legend_kw=None, **plot_kw):
        """Creates a cartesian plot
        of multiple slices (columns)
        of the polar diagram

        Parameters
        ----------
        ws_range : Iterable, optional
            Slices (columns) of the
            polar diagram table,
            given as an Iterable
            of elements of
            self.wind_speeds.
            If nothing it passed,
            it will default to
            self.Wind_speeds

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        colors : tuple, optional
            Specifies the colors to
            be used for the different
            slices. There are four
            options:
                If as many or more
                colors as slices
                are passed,
                each slice will
                be plotted in the
                specified color

                Otherwise if
                exactly 2 colors
                are passed, the
                slices will be
                plotted with a
                color gradient
                consiting of the
                two colors

                If more than 2
                colors are passed,
                either the first
                n_color slices will
                be plotted in the
                specified colors,
                and the rest will
                be plotted in the
                default color 'blue',
                or one can specify
                certain slices to be
                plotted in a certain
                color by passing a
                tuple of (ws, color)
                pairs

                Defaults to the tuple
                ('green', 'red')

        show_legend : bool, optional
            Specifies wether or not
            a legend will be shown
            next to the plot

            The type of legend depends
            on the color options:
            If the slices are plotted
            with a color gradient,
            a matplotlib.colorbar.Colorbar
            object will be created
            and assigned to ax

            Otherwise a
            matplotlib.legend.Legend
            will be created and
            assigned to ax

            Default to 'True'

        legend_kw : dict, optional
            Keyword arguments to be
            passed to either the
            matplotlib.colorbar.Colorbar
            or matplotlib.legend.Legend
            classes to change position
            and appearence of the legend

            Will only be used if
            'show_legend=True'

            Defaults to an empty
            dictionary

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot

        Function raises an exception
        if at least one element
        of ws_range is not in
        the wind speed resolution
        """
        logger.info(f"""Method 'PolarDiagramTable.flat_plot(
                     ws_range={ws_range}, ax={ax}, colors={colors}, 
                     show_legend={show_legend}, legen_kw={legend_kw}, 
                     plot_kw={plot_kw})' called""")

        if ws_range is None:
            ws_range = self.wind_speeds

        if isinstance(ws_range, np.ndarray):
            if not ws_range.size:
                logger.error("")
                raise PolarDiagramException("")
        elif not ws_range:
            logger.error("")
            raise PolarDiagramException("")

        bsp_list = list(self._get_slice_data(ws=ws_range).T)
        wa_list = [list(self.wind_angles)] * len(bsp_list)

        logger.info(f"""Internal function 
                     'plotting.plot_flat_range(ws_range, 
                     wa_list, bsp_list, colors, show_legend, 
                     legend_kw, **plot_kw) called'""")
        return plot_flat_range(
            ws_range, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def plot_3d(self, ax=None, colors=('blue', 'blue')):
        """Creates a 3d plot
        of the polar diagram

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        colors : tuple of length 2, optional
            Colors which specify
            the color gradient with
            which the polar diagram
            will be plotted.

            If no color gradient is
            desired, set both elements
            to the same color

            Defaults to
            ('blue', 'blue')
        """
        logger.info(f"""Method 'PolarDiagramTable.plot_3d(ax={ax}, 
                     colors={colors})' called""")

        ws, wa = np.meshgrid(self.wind_speeds,
                             self._get_radians())
        bsp = self.boat_speeds
        bsp, wa = bsp * np.cos(wa), bsp * np.sin(wa)

        logger.info("""Internal function 'plotting.plot_surface(
                     ws, wa, bsp, ax, colors)' called""")
        return plot_surface(ws, wa, bsp, ax, colors)

    def plot_color_gradient(
            self, ax=None, colors=('green', 'red'),
            marker=None, show_legend=True, **legend_kw):
        """Creates a 'wind speed
        vs. wind angle' color gradient
        plot of the polar diagram
        with respect to the
        respective boat speeds

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        colors : tuple of length 2, optional
            Colors which specify
            the color gradient with
            which the polar diagram
            will be plotted.

            Defaults to
            ('green', 'red')

        marker : matplotlib.markers.Markerstyleor equivalent, optional
            Markerstyle for the
            created scatter plot

            If nothing is passed,
            it will default to 'o'

        show_legend : bool, optional
            Specifies wether or not
            a legend will be shown
            next to the plot

            Legend will be a
            matplotlib.colorbar.Colorbar
            object.

            Defaults to 'True'

        legend_kw : Keyword arguments
            Keyword arguments to be
            passed to the
            matplotlib.colorbar.Colorbar
            class to change position
            and appearence of the legend.

            Will only be used if
            'show_legend=True'
        """
        logger.info(f"""Method 'PolarDiagramTable.plot_color_gradient(
                     ax={ax}, colors={colors}, marker={marker}, 
                     show_legend={show_legend}, legend_kw={legend_kw})' 
                     called""")

        ws, wa = np.meshgrid(self.wind_speeds,
                             self.wind_angles)
        ws = ws.reshape(-1, )
        wa = wa.reshape(-1, )
        bsp = self.boat_speeds.reshape(-1, )

        logger.info("""Internal function
                     'plotting.plot_color(ws, wa, bsp, 
                     ax, colors, marker, show_legend, 
                     **legend_kw)' called""")
        return plot_color(
            ws, wa, bsp,
            ax, colors, marker, show_legend, **legend_kw)

    def plot_convex_hull_slice(self, ws, ax=None,
                               **plot_kw):
        """Computes the convex
        hull of a slice (column)
        of the polar diagram
        and creates a polar plot
        of it

        Parameters
        ----------
        ws : int or float
            Slice (column) of the polar
            diagram, given as an element
            of self.wind_speeds

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot


        Function raises an exception
        if ws is not in the wind
        speed resolution.

        """
        logger.info(f"""Method 'PolarDiagramTable.plot_convex_hull_slice(
                     ws={ws}, ax={ax}, plot_kw={plot_kw})' called""")

        wa = list(self._get_radians())
        bsp = self._get_slice_data(ws)

        logger.info("""Internal function
                     'plotting.plot_convex_hull(wa, bsp, 
                     ax, **plot_kw)' called""")
        return plot_convex_hull(wa, bsp, ax, **plot_kw)

    def plot_convex_hull_3d(self):
        """"""
        pass


class PolarDiagramCurve(PolarDiagram):
    """
    A class to represent,
    visualize and work
    with a polar diagram
    given by a fitted curve/surface.

    Parameters
    ----------
    f : function
        Curve/surface that describes
        the polar diagram, given as
        a function, which takes a
        numpy.ndarray with two columns,
        corresponding to (wind speed, wind angle)
        pairs as well as some additional
        parameters

    radians : bool, optional
        Specifies if f takes the
        wind angles to be in
        radians or degrees

        Defaults to 'False'

    *params : Arguments
        Additional optimized
        parameters that
        f takes


    Methods
    -------
    curve
        Returns a read only version
        of self._f
    parameters
        Returns a read only version
        of self._params
    radians
        Returns a read only version
        of self._rad
    to_csv(csv_path)
        Creates a .csv-file with
        delimiter ':' and the
        following format:
            PolarDiagramCurve
            Function: self.curve
            Radians: self.rad
            Parameters: self.parameters
    polar_plot_slice(ws, ax=None,
                     **plot_kw)
        Creates a polar plot
        of a given slice of
        the polar diagram
    flat_plot_slice(ws, ax=None,
                    **plot_kw)
        Creates a cartesian
        plot of a slice of the
        polar diagram
    polar_plot(ws_range=(0, 20, 5), ax=None,
               colors=('green', 'red'),
               show_legend=True,
               legend_kw=None, **plot_kw)
        Creates a polar plot
        of multiple slices of
        the polar diagram
    flat_plot(ws_range=(0, 20, 5), ax=None,
              colors=('green', 'red'),
              show_legend=True,
              legend_kw=None, **plot_kw)
        Creates a cartesian
        plot of multiple slices
        of the polar diagram
    plot_3d(ws_range=(0, 20, 100), ax=None,
            colors=('blue', 'blue'))
        Creates a 3d plot
        of a part of the
        polar diagram
    plot_color_gradient(ws_range=(0, 20, 100),
                        ax=None,
                        colors=('green', 'red'),
                        marker=None,
                        show_legend=True,
                        **legend_kw)
        Creates a 'wind speed
        vs. wind angle' color gradient
        plot of a part of the
        polar diagram with
        respect to the
        respective boat speeds
    plot_convex_hull_slice(ws, ax=None,
                           **plot_kw)
        Computes the convex
        hull of a slice (column)
        of the polar diagram
        and creates a polar plot
        of it
    """

    def __init__(self, f, radians=False, *params):
        logger.info(f"""Class 'PolarDiagramCurve(
                     f={f.__name__}, radians={radians}, *params={params})'
                     called""")
        # TODO Errorchecks
        self._f = f
        self._params = params
        self._rad = radians

    def __repr__(self):
        logger.info("Dunder-Method 'PolarDiagramCurve.__repr__()' called")
        return f"PolarDiagramCurve(f={self.curve.__name__}, " \
               f"                  radians={self.radians}, " \
               f"                  {self.parameters})"

    def __call__(self, ws, wa):
        logger.info(f"""Dunder-Method 'PolarDiagramCurve.__call__(
                     ws, wa)' called""")
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

    def to_csv(self, csv_path):
        """Creates a .csv file with
        delimiter ':' and the
        following format:
            PolarDiagramCurve
            Function: self.curve.__name__
            Radians: self.rad
            Parameters: self.parameters

        Parameters
        ----------
        csv_path : string
            Path where a .csv file is
            located or where a new
            .csv file will be created

        Function raises an exception
        if file can't be written to
        """
        logger.info(f"Method 'PolarDiagramCurve.to_csv({csv_path})' called")

        try:
            with open(csv_path, 'w', newline='') as file:
                csv_writer = csv.writer(file, delimiter=':', quotechar='"')
                csv_writer.writerow(["PolarDiagramCurve"])
                csv_writer.writerow(["Function"] + [self.curve.__name__])
                csv_writer.writerow(["Radians"] + [str(self.radians)])
                csv_writer.writerow(["Parameters"] + list(self.parameters))
        except OSError:
            logger.error(f"Error occured when accessing file {csv_path}")
            raise FileReadingException(f"Can't write to {csv_path}")

    def _get_wind_angles(self):
        wa = np.linspace(0, 360, 1000)
        if self.radians:
            wa = np.deg2rad(wa)
        return wa

    def polar_plot_slice(self, ws, ax=None, **plot_kw):
        """Creates a polar plot
        of a given slice of
        the polar diagram

        Parameters
        ----------
        ws : int or float
            Slice of the polar diagram,
            given as a single wind speed

            Slice then equals
            self(ws, wa), where wa will
            go through several wind angles

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot


        """
        logger.info(f"""Method 'PolarDiagramCurve.polar_plot_slice(ws={ws},
                    ax={ax}, **plot_kw={plot_kw})' called""")

        wa = self._get_wind_angles()
        bsp = self(np.array([ws] * 1000), wa)

        logger.info("""Internal function 'plotting.plot_polar(
                     np.deg2rad(np.linspace(0, 360, 1000), bsp, ax, 
                     **plot_kw)' called""")
        return plot_polar(
            np.deg2rad(np.linspace(0, 360, 1000)), bsp,
            ax, **plot_kw)

    def flat_plot_slice(self, ws, ax=None, **plot_kw):
        """Creates a cartesian
        plot of a slice of the
        polar diagram

        Parameters
        ----------
        ws : int or float
            Slice of the polar diagram,
            given as a single wind speed

            Slice then equals
            self(ws, wa), where wa will
            go through several wind angles

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot


        """
        logger.info(f"""Method 'PolarDiagramCurve.flat_plot_slice(ws={ws},
                     ax={ax}, **plot_kw={plot_kw})' called""")

        wa = self._get_wind_angles()
        bsp = self(np.array([ws] * 1000), wa)

        logger.info("""Internal function 'plotting.plot_flat(
                     np.linspace(0, 360, 1000), bsp, ax, **plot_kw)' 
                     called""")
        return plot_flat(
            np.linspace(0, 360, 1000), bsp, ax,
            **plot_kw)

    def polar_plot(self, ws_range=(0, 20, 5), ax=None,
                   colors=('green', 'red'), show_legend=True,
                   legend_kw=None, **plot_kw):
        """Creates a polar plot
        of multiple slices of
        the polar diagram

        Parameters
        ----------
        ws_range : tuple of length 3 or list, optional
            Slices of the polar diagram
            given either as a
            tuple of three values, which
            will be interpreted as a
            start and end point of an
            interval aswell as a number of
            slices, which will be evenly
            spaces in the given interval,
            or a list of specific wind speed
            values

            Defaults to (0, 20, 5)

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        colors : tuple, optional
            Specifies the colors to
            be used for the different
            slices. There are four
            options:
                If as many or more
                colors as slices
                are passed,
                each slice will
                be plotted in the
                specified color

                Otherwise if
                exactly 2 colors
                are passed, the
                slices will be
                plotted with a
                color gradient
                consiting of the
                two colors

                If more than 2
                colors are passed,
                either the first
                n_color slices will
                be plotted in the
                specified colors,
                and the rest will
                be plotted in the
                default color 'blue',
                or one can specify
                certain slices to be
                plotted in a certain
                color by passing a
                tuple of (ws, color)
                pairs

                Defaults to the tuple
                ('green', 'red')

        show_legend : bool, optional
            Specifies wether or not
            a legend will be shown
            next to the plot

            The type of legend depends
            on the color options:
            If the slices are plotted
            with a color gradient,
            a matplotlib.colorbar.Colorbar
            object will be created
            and assigned to ax.

            Otherwise a
            matplotlib.legend.Legend
            will be created and
            assigned to ax.

            Default to 'True'

        legend_kw : dict, optional
            Keyword arguments to be
            passed to either the
            matplotlib.colorbar.Colorbar
            or matplotlib.legend.Legend
            classes to change position
            and appearence of the legend.

            Will only be used if
            'show_legend=True'

            Defaults to an empty
            dictionary

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot

        """
        logger.info(f"""Method PolarDiagramCurve.polar_plot(
                     ws_range={ws_range}, ax={ax}, colors={colors},
                     show_legend={show_legend}, legend_kw={legend_kw},
                     **plot_kw={plot_kw})' called""")

        if isinstance(ws_range, tuple):
            ws_lower, ws_upper, ws_step = ws_range
            ws_range = list(np.linspace(ws_lower, ws_upper, ws_step))

        wa = np.linspace(0, 360, 1000)
        wa_list = [np.deg2rad(wa)] * len(ws_range)
        if self.radians:
            wa = wa_list[0]

        bsp_list = []
        for ws in ws_range:
            bsp_list.append(self(np.array([ws] * 1000), wa))

        logger.info("""Internal function 'plotting.plot_polar_range(
                     ws_list, wa_list, bsp_list, ax, colors, show_legend,
                     legend_kw, **plot_kw)' called""")
        return plot_polar_range(
            ws_range, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def flat_plot(self, ws_range=(0, 20, 5), ax=None,
                  colors=('green', 'red'), show_legend=True,
                  legend_kw=None, **plot_kw):
        """Creates a cartesian
        plot of multiple slices
        of the polar diagram

        Parameters
        ----------
        ws_range : tuple of length 3 or list, optional
            Slices of the polar diagram
            given either as a
            tuple of three values, which
            will be interpreted as a
            start and end point of an
            interval aswell as a number of
            slices, which will be evenly
            spaces in the given interval,
            or a list of specific wind speed
            values

            Defaults to (0, 20, 5)

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        colors : tuple, optional
            Specifies the colors to
            be used for the different
            slices. There are four
            options:
                If as many or more
                colors as slices
                are passed,
                each slice will
                be plotted in the
                specified color

                Otherwise if
                exactly 2 colors
                are passed, the
                slices will be
                plotted with a
                color gradient
                consiting of the
                two colors

                If more than 2
                colors are passed,
                either the first
                n_color slices will
                be plotted in the
                specified colors,
                and the rest will
                be plotted in the
                default color 'blue',
                or one can specify
                certain slices to be
                plotted in a certain
                color by passing a
                tuple of (ws, color)
                pairs

                Defaults to the tuple
                ('green', 'red')

        show_legend : bool, optional
            Specifies wether or not
            a legend will be shown
            next to the plot

            The type of legend depends
            on the color options:
            If the slices are plotted
            with a color gradient,
            a matplotlib.colorbar.Colorbar
            object will be created
            and assigned to ax.

            Otherwise a
            matplotlib.legend.Legend
            will be created and
            assigned to ax.

            Default to 'True'

        legend_kw : dict, optional
            Keyword arguments to be
            passed to either the
            matplotlib.colorbar.Colorbar
            or matplotlib.legend.Legend
            classes to change position
            and appearence of the legend.

            Will only be used if
            'show_legend=True'

            Defaults to an empty
            dictionary

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot

        """
        logger.info(f"""Method 'PolarDiagramCurve.flat_plot(
                     ws_range={ws_range}, ax={ax}, colors={colors},
                     show_legend={show_legend}, legend_kw={legend_kw},
                     **plot_kw={plot_kw})' called""")

        if isinstance(ws_range, tuple):
            ws_lower, ws_upper, ws_step = ws_range
            ws_range = list(np.linspace(ws_lower, ws_upper, ws_step))

        wa = np.linspace(0, 360, 1000)
        wa_list = [wa] * len(ws_range)
        if self.radians:
            wa = np.deg2rad(wa)

        bsp_list = []
        for ws in ws_range:
            bsp_list.append(self(np.array([ws] * 1000), wa))

        logger.info("""Internal function 'plotting.plot_flat_range(
                     ws_list, wa_list, bsp_list, ax, colors, show_legend,
                     legend_kw, **plot_kw)' called""")
        return plot_flat_range(
            ws_range, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def plot_3d(self, ws_range=(0, 20, 100), ax=None,
                colors=('blue', 'blue')):
        """Creates a 3d plot
        of a part of the
        polar diagram

        Parameters
        ----------
        ws_range : tuple of length 3, optional
            A region of the polar
            diagram given as a
            tuple of three values,
            which will be interpreted
            as a start and end point
            of an interval aswell as
            a number of samples in
            this interval. The more
            samples there are, the
            "smoother" the resulting
            plot will be

            Defaults to (0, 20, 100)

        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        colors : tuple of length 2, optional
            Colors which specify
            the color gradient with
            which the polar diagram
            will be plotted.

            If no color gradient is
            desired, set both elements
            as the same color

            Defaults to
            ('blue', 'blue')

        """
        logging.info(f"""Method 'PolarDiagramCurve.plot_3d(
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
                 self(np.array([wind_speed] * 1000), wa).reshape(-1, 1)))

        if self.radians:
            bsp, wa = bsp_arr * np.cos(wa_arr), bsp_arr * np.sin(wa_arr)
        else:
            wa_arr = np.deg2rad(wa_arr)
            bsp, wa = bsp_arr * np.cos(wa_arr), bsp_arr * np.sin(wa_arr)

        logging.info("""Internal function 'plotting.plot_surface(
                      ws_arr, wa, bsp, ax, colors)' called""")
        return plot_surface(ws_arr, wa, bsp, ax, colors)

    def plot_color_gradient(
            self, ws_range=(0, 20, 100), ax=None,
            colors=('green', 'red'), marker=None,
            show_legend=True, **legend_kw):
        """Creates a 'wind speed
        vs. wind angle' color gradient
        plot of a part of the
        polar diagram with respect
        to the respective boat speeds

        Parameters
        ----------
        ws_range :  tuple of length 3, optional
            A region of the polar
            diagram given as a
            tuple of three values,
            which will be interpreted
            as a start and end point
            of an interval aswell as
            a number of samples in
            this interval.

            Defaults to (0, 20, 100)

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        colors : tuple of length 2, optional
            Colors which specify
            the color gradient with
            which the polar diagram
            will be plotted.

            Defaults to
            ('green', 'red')

        marker : matplotlib.markers.Markerstyleor equivalent, optional
            Markerstyle for the
            created scatter plot

            If nothing is passed,
            function uses the
            default 'o'

        show_legend : bool, optional
            Specifies wether or not
            a legend will be shown
            next to the plot

            Legend will be a
            matplotlib.colorbar.Colorbar
            object.

            Defaults to true

        legend_kw :
            Keyword arguments to be
            passed to the
            matplotlib.colorbar.Colorbar
            class to change position
            and appearence of the legend.

            Will only be used if
            'show_legend=True'

        """
        logger.info(f"""Method 'PolarDiagramCurve.plot_color_gradient(
                     ws_range={ws_range}, ax={ax}, colors={colors}, 
                     marker={marker}, show_legend={show_legend},
                     **legend_kw={legend_kw})' called""")

        ws_lower, ws_upper, ws_step = ws_range
        ws, wa = np.meshgrid(
            np.linspace(ws_lower, ws_upper, ws_step),
            np.linspace(0, 360, 1000))
        ws = ws.reshape(-1, )
        wa = wa.reshape(-1, )

        if self.radians:
            bsp = self(ws, np.deg2rad(wa)).reshape(-1, )
        else:
            bsp = self(ws, wa).reshape(-1, )

        logger.info("""Internal function 'plotting.plot_color(
                     ws, wa, bsp, ax, colors, marker, show_legend,
                     **legend_kw)' called""")
        return plot_color(
            ws, wa, bsp,
            ax, colors, marker, show_legend, **legend_kw)

    def plot_convex_hull_slice(self, ws, ax=None, **plot_kw):
        """Computes the convex
        hull of a slice (column)
        of the polar diagram
        and creates a polar plot
        of it

        Parameters
        ----------
        ws : int or float
            Slice of the polar diagram,
            given as a single wind speed

            Slice then equals
            self(ws, wa), where wa will
            go through several wind angles

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot


        """
        logger.info(f"""Method 'PolarDiagramCurve.plot_convex_hull_slice(
                     ws={ws}, ax={ax}, **plot_kw={plot_kw})' called""")

        wa = self._get_wind_angles()
        bsp = self(np.array([ws] * 1000), wa)

        logger.info("""Internal function 'plotting.plot_convex_hull(
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
    A class to represent,
    visualize and work
    with a polar diagram
    given by a point cloud

    Parameters
    ----------
    points : array_like, optional
        Initial points of the
        point cloud, given
        as a sequence of points
        consisting of wind speed,
        wind angle and boat speed

        If nothing is passed,
        point cloud will be
        initialized with an
        empty array

    tw : bool, optional
        Specifies if the
        given wind data should
        be viewed as true wind

        If False, wind data
        will be converted
        to true wind

        Defaults to True

    Raises an exception
    if points can't be
    broadcasted to a
    fitting shape

    Methods
    -------
    wind_speeds
        Returns a list of all the
        different wind speeds
        in the point cloud
    wind_angles
        Returns a list of all the
        different wind angles
        in the point cloud
    points
        Returns a read only version
        of self._data
    to_csv(csv_path)
        Creates a .csv-file with
        delimiter ',' and the
        following format:
            PolarDiagramPointcloud
            True wind speed: , True wind angle: , Boat speed:
            self.get_points
    add_points(new_points)
        Adds additional points
        to the point cloud
    polar_plot_slice(ws, ax=None,
                     **plot_kw)
        Creates a polar plot
        of a slice of the
        polar diagram
    flat_plot_slice(ws, ax=None,
                    **plot_kw)
        Creates a cartesian
        plot of a slice of
        the polar diagram
    polar_plot(ws_range=(0, np.inf), ax=None,
               colors=('green', 'red'),
               show_legend=True,
               legend_kw=None, **plot_kw)
        Creates a polar plot
        of multiple slices of
        the polar diagram
    flat_plot(ws_range=(0, np.inf), ax=None,
              colors=('green', 'red'),
              show_legend=True,
              legend_kw=None, **plot_kw)
        Creates a cartesian
        plot of multiple slices
        of the polar diagram
    plot_3d(ax=None, **plot_kw)
        Creates a 3d plot
        of the polar diagram
    plot_color_gradient(ax=None,
                        colors=('green', 'red'),
                        marker=None,
                        show_legend=True,
                        **legend_kw)
        Creates a 'wind speed
        vs. wind angle' color gradient
        plot of the polar diagram
        with respect to the
        respective boat speeds
    plot_convex_hull_slice(ws, ax=None,
                           **plot_kw)
        Computes the convex
        hull of a slice (column)
        of the polar diagram
        and creates a polar plot
        of it
    """

    def __init__(self, points=None, tw=True):
        logger.info(f"""Class 'PolarDiagramPointcloud(
                     points, tw={tw})' called""")

        if points is not None:
            points = np.asarray(points)
            if len(points[0]) != 3:
                try:
                    points = points.reshape(-1, 3)
                except ValueError:
                    logger.error("""Error occured when trying to broadcast
                                 points to shape (_, 3)""")
                    raise PolarDiagramException(
                        "points could not be broadcasted "
                        "to an array of shape (n,3)")

            logger.info("""Internal function 'utils.convert_wind(
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
        logger.info("""Dunder-Method 'PolarDiagramPointcloud.__str__()'
                     called""")

        return tabulate(self.points, headers=["TWS", "TWA", "BSP"])

    def __repr__(self):
        logger.info("""Dunder-Method 'PolarDiagramPointcloud.__repr__()'
                     called""")

        return f"PolarDiagramPointcloud(data={self.points})"

    @property
    def wind_speeds(self):
        """Returns a list of all the different wind speeds
        in the point cloud"""
        return list(dict.fromkeys(self.points[:, 0]))

    @property
    def wind_angles(self):
        """Returns a list of all the different wind angles
        in the point cloud"""
        return list(dict.fromkeys(self.points[:, 1]))

    @property
    def points(self):
        """Returns a read only version of self._data"""
        return self._data.copy()

    def to_csv(self, csv_path):
        """Creates a .csv file with
        delimiter ',' and the
        following format:
            PolarDiagramPointcloud
            True wind speed ,True wind angle ,Boat speed
            self.get_points

        Parameters
        ----------
        csv_path : string
            Path where a .csv-file is
            located or where a new
            .csv file will be created

        Function raises an exception
        if file can't be written to
        """
        logger.info(f"""Method 'PolarDiagramPointcloud.to_csv({csv_path})'
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
            logger.error(f"Error occured when accessing file {csv_path}")
            raise FileReadingException(f"Can't write to {csv_path}")

    def add_points(self, new_points, tw=True):
        """Adds additional
        points to the point cloud

        Parameters
        ----------
        new_points: array_like
            New points to be added to
            the point cloud given as
            a sequence of points
            consisting of wind speed,
            wind angel and boat speed

        tw : bool, optional
            Specifies if the
            given wind data should
            be viewed as true wind

            If False, wind data
            will be converted
            to true wind

            Defaults to True

        Function raises an
        exception if new_points
        can't be broadcasted
        to a fitting shape
        """
        logger.info(f"""Method 'PolarDiagramPointcloud.add_points(
                     new_points, tw={tw})' called""")

        new_points = np.asarray(new_points)
        if len(new_points[0]) != 3:
            try:
                new_points = new_points.reshape(-1, 3)
            except ValueError:
                logger.error("""Error occured when trying to broadcast 
                            new_points to shape (_, 3)""")
                raise PolarDiagramException(
                    "new_points could not be broadcasted "
                    "to an array of shape (n,3)")

        logger.info(f"""Internal function 'utils.convert_wind(
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
        points = self.points[self.points[:, 0] == ws][:, 1:]
        if points.size == 0:
            logger.error("Error occured, when trying to get slice data")
            raise PolarDiagramException(
                f"No points with wind speed={ws} found")

        return points[:, 0], points[:, 1]

    def polar_plot_slice(self, ws, ax=None, **plot_kw):
        """Creates a polar plot
        of a slice of the
        polar diagram

        Parameters
        ----------
        ws : int or float
            Slice of the polar diagram
            given by a single wind speed

            Slice then consists of all
            the points in the point
            cloud with wind speed ws

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot

        Function raises an
        exception if there
        are no points in
        the given slice in
        the point cloud
        """
        logger.info(f"""Method 'PolarDiagramPointcloud.polar_plot_slice(
                     ws={ws}, ax={ax}, **plot_kw={plot_kw})' called""")

        wa, bsp = self._get_slice_data(ws)
        wa = list(np.deg2rad(wa))

        logger.info("""Internal function 'plotting.plot_polar(
                     wa, bsp, ax, **plot_kw)' called""")
        return plot_polar(wa, bsp, ax, **plot_kw)

    def flat_plot_slice(self, ws, ax=None, **plot_kw):
        """Creates a cartesian
        plot of a slice of
        the polar diagram

        Parameters
        ----------
        ws : int or float
            Slice of the polar diagram
            given by a single wind speed

            Slice then consists of all
            the points in the point
            cloud with wind speed ws

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot

        Function raises an
        exception if there
        are no points in
        the given slice in
        the point cloud
        """
        logger.info(f"""Method 'PolarDiagramPointcloud.flat_plot_slice(
                     ws={ws}, ax={ax}, **plot_kw={plot_kw})' called""")

        wa, bsp = self._get_slice_data(ws)

        logger.info("""Internal function 'plotting.plot_flat(
                     wa, bsp, ax, **plot_kw)' called""")
        return plot_flat(wa, bsp, ax, **plot_kw)

    def _get_slices(self, ws_range):
        wa_list, bsp_list = [], []

        if not ws_range:
            logger.error("")
            raise PolarDiagramException("")

        if isinstance(ws_range, tuple):
            ws_lower, ws_upper = ws_range
            ws_range = [ws for ws in self.wind_speeds
                        if ws_lower <= ws <= ws_upper]

        for ws in ws_range:
            points = self._get_slice_data(ws)
            wa_list.append(points[0])
            bsp_list.append(points[1])

        return ws_range, wa_list, bsp_list

    def polar_plot(self, ws_range=(0, np.inf), ax=None,
                   colors=('green', 'red'), show_legend=True,
                   legend_kw=None, **plot_kw):
        """Creates a polar plot
        of multiple slices of
        the polar diagram

        Parameters
        ----------
        ws_range : tuple of length 2 or list, optional
            Slices of the polar diagram
            given as either a tuple of
            two values which will be
            interpreted as a lower
            and upper bound of the
            wind speed, such that all
            slices with a wind speed
            that fits within these
            bounds will be plotted,
            or a list of specific
            wind speed values / slices
            which will be plotted

            Defaults to (0, np.inf)

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        colors : tuple, optional
            Specifies the colors to
            be used for the different
            slices. There are four
            options:
                If as many or more
                colors as slices
                are passed,
                each slice will
                be plotted in the
                specified color

                Otherwise if
                exactly 2 colors
                are passed, the
                slices will be
                plotted with a
                color gradient
                consiting of the
                two colors

                If more than 2
                colors are passed,
                either the first
                n_color slices will
                be plotted in the
                specified colors,
                and the rest will
                be plotted in the
                default color 'blue',
                or one can specify
                certain slices to be
                plotted in a certain
                color by passing a
                tuple of (ws, color)
                pairs

                Defaults to the tuple
                ('green', 'red')

        show_legend : bool, optional
            Specifies wether or not
            a legend will be shown
            next to the plot

            The type of legend depends
            on the color options:
            If the slices are plotted
            with a color gradient,
            a matplotlib.colorbar.Colorbar
            object will be created
            and assigned to ax.

            Otherwise a
            matplotlib.legend.Legend
            will be created and
            assigned to ax.

            Default to 'True'

        legend_kw : dict, optional
            Keyword arguments to be
            passed to either the
            matplotlib.colorbar.Colorbar
            or matplotlib.legend.Legend
            classes to change position
            and appearence of the legend.

            Will only be used if
            'show_legend=True'

            Defaults to an empty
            dictionary

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot

        Raises an exception
        if ws_range is
        a list and there is
        a wind_speed in ws_range
        such that there are no
        points in the given slice
        in the point cloud
        """
        logger.info(f"""Method 'PolarDiagramPointcloud.polar_plot(
                     ws_range={ws_range}, ax={ax}, colors={colors},
                     show_legend={show_legend}, legend_kw={legend_kw},
                     **plot_kw={plot_kw})' called""")

        ws_list, wa_list, bsp_list = \
            self._get_slices(ws_range)
        wa_list = [np.deg2rad(wa) for wa in wa_list]

        logger.info("""Internal function 'plotting.plot_polar_range(
                     ws_list, wa_list, bsp_list, ax, colors,
                     show_legend, legend_kw, **plot_kw)' called""")
        return plot_polar_range(
            ws_list, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def flat_plot(self, ws_range=(0, np.inf), ax=None,
                  colors=('green', 'red'), show_legend=True,
                  legend_kw=None, **plot_kw):
        """Creates a cartesian
        plot of multiple slices
        of the polar diagram

        Parameters
        ----------
        ws_range : tuple of length 2 or list, optional
            Slices of the polar diagram
            given as either a tuple of
            two values which will be
            interpreted as a lower
            and upper bound of the
            wind speed, such that all
            slices with a wind speed
            that fits within these
            bounds will be plotted,
            or a list of specific
            wind speed values / slices
            which will be plotted

            Defaults to (0, np.inf)

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        colors : tuple, optional
            Specifies the colors to
            be used for the different
            slices. There are four
            options:
                If as many or more
                colors as slices
                are passed,
                each slice will
                be plotted in the
                specified color

                Otherwise if
                exactly 2 colors
                are passed, the
                slices will be
                plotted with a
                color gradient
                consiting of the
                two colors

                If more than 2
                colors are passed,
                either the first
                n_color slices will
                be plotted in the
                specified colors,
                and the rest will
                be plotted in the
                default color 'blue',
                or one can specify
                certain slices to be
                plotted in a certain
                color by passing a
                tuple of (ws, color)
                pairs

                Defaults to the tuple
                ('green', 'red')

        show_legend : bool, optional
            Specifies wether or not
            a legend will be shown
            next to the plot

            The type of legend depends
            on the color options:
            If the slices are plotted
            with a color gradient,
            a matplotlib.colorbar.Colorbar
            object will be created
            and assigned to ax.

            Otherwise a
            matplotlib.legend.Legend
            will be created and
            assigned to ax.

            Default to 'True'

        legend_kw : dict, optional
            Keyword arguments to be
            passed to either the
            matplotlib.colorbar.Colorbar
            or matplotlib.legend.Legend
            classes to change position
            and appearence of the legend.

            Will only be used if
            'show_legend=True'

            Defaults to an empty
            dictionary

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot

        Raises an exception
        if ws_range is
        a list and there is
        a wind_speed in ws_range
        such that there are no
        points in the given slice
        in the point cloud
        """
        logger.info(f"""Method 'PolarDiagramPointcloud.flat_plot(
                     ws_range={ws_range}, ax={ax}, colors={colors},
                     show_legend={show_legend}, legend_kw={legend_kw},
                     **plot_kw={plot_kw})' called""")

        ws_list, wa_list, bsp_list = \
            self._get_slices(ws_range)

        logger.info("""Internal function 'plotting.plot_flat_range(
                     ws_list, wa_list, bsp_list, ax, colors,
                     show_legend, legend_kw, **plot_kw)' called""")
        return plot_flat_range(
            ws_list, wa_list, bsp_list,
            ax, colors, show_legend, legend_kw, **plot_kw)

    def plot_3d(self, ax=None, **plot_kw):
        """Creates a 3d plot
        of the polar diagram

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot

        """
        logger.info(f"""Method 'PolarDiagramPointcloud.plot_3d(
                     ax={ax}, **plot_kw={plot_kw})' called""")

        ws, wa, bsp = (self.points[:, 0],
                       self.points[:, 1],
                       self.points[:, 2])
        wa = np.deg2rad(wa)
        bsp, wa = bsp * np.cos(wa), bsp * np.sin(wa)

        logger.info("""Internal function 'plotting.plot3d(
                     ws, wa, bsp, ax, **plot_kw)' called""")
        return plot3d(ws, wa, bsp, ax, **plot_kw)

    def plot_color_gradient(self, ax=None, colors=('green', 'red'),
                            marker=None, show_legend=True, **legend_kw):
        """Creates a 'wind speed
        vs. wind angle' color gradient
        plot of the polar diagram
        with respect to the
        respective boat speeds

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        colors : tuple of length 2, optional
            Colors which specify
            the color gradient with
            which the polar diagram
            will be plotted.

            Defaults to
            ('green', 'red')

        marker : matplotlib.markers.Markerstyleor equivalent, optional
            Markerstyle for the
            created scatter plot

            If nothing is passed,
            function uses the
            default 'o'

        show_legend : bool, optional
            Specifies wether or not
            a legend will be shown
            next to the plot

            Legend will be a
            matplotlib.colorbar.Colorbar
            object.

            Defaults to true

        legend_kw :
            Keyword arguments to be
            passed to the
            matplotlib.colorbar.Colorbar
            class to change position
            and appearence of the legend.

            Will only be used if
            'show_legend=True'

        """
        logger.info(f"""Method 'PolarDiagramPointcloud.plot_color_gradient(
                     ax={ax}, colors={colors}, marker={marker}, 
                     show_legend={show_legend}, **legend_kw={legend_kw})'
                     called""")

        ws, wa, bsp = np.hsplit(self.points, 3)

        logger.info("""Internal function 'plotting.plot_color(
                     ws, wa, bsp, ax, colors, marker, show_legend,
                     **legend_kw)' called""")
        return plot_color(
            ws, wa, bsp,
            ax, colors, marker, show_legend, **legend_kw)

    def plot_convex_hull_slice(self, ws, ax=None, **plot_kw):
        """Computes the convex
        hull of a slice of
        the polar diagram and
        creates a polar plot
        of it

        Parameters
        ----------
        ws : int or float
            Slice of the polar diagram
            given by a single wind speed

            Slice then consists of all
            the points in the point
            cloud with wind speed ws

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot
            will be created. If nothing
            is passed, the function will
            create a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will
            be passed to the
            matplotlib.axes.Axes.plot
            function, to change certain
            appearences of the plot

        Function raises an
        exception if there
        are no points in
        the given slice in
        the point cloud
        """
        logger.info(f"""Method 'PolarDiagramPointcloud.plot_convex_hull_slice(
                     ws={ws}, ax={ax}, **plot_kw={plot_kw})' called""")

        wa, bsp = self._get_slice_data(ws)
        wa = list(np.deg2rad(wa))

        logger.info("""Internal function 'plotting.plot_convex_hull(
                     wa, bsp, ax, **plot_kw)' called""")
        return plot_convex_hull(wa, bsp, ax, **plot_kw)

    def plot_convex_hull_3d(self):
        """"""
        pass
