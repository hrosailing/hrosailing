"""
Classes to represent polar diagrams in various different forms
as well as small functions to save / load PolarDiagram-objects to files
in different forms and functions to manipulate PolarDiagram-objects
"""

# Author: Valentin F. Dannenberg / Ente

import csv
import logging.handlers
import pickle

from abc import ABC, abstractmethod

from hrosailing.polardiagram.plotting import *

from hrosailing.wind import (
    apparent_wind_to_true,
    speed_resolution,
    angle_resolution,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    filename="hrosailing/logging/polardiagram.log",
)
LOG_FILE = "hrosailing/logging/polardiagram.log"

logger = logging.getLogger(__name__)
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when="midnight"
)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


class PolarDiagramException(Exception):
    pass


class FileReadingException(Exception):
    pass


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

    obj.to_csv(csv_path)


# TODO: Make it cleaner!
def from_csv(csv_path, fmt="hro", tw=True):
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
    out : PolarDiagram
        PolarDiagram instance
        saved in .csv file

    Function raises an exception:
        if an unknown format was
        specified

        file can't be found,
        opened, or read
    """

    FMTS = ("hro", "orc", "array", "opencpn")
    if fmt not in FMTS:
        raise FileReadingException(f"Format {fmt} not yet implemented")
    try:
        with open(csv_path, "r", newline="") as file:
            if fmt == "hro":
                csv_reader = csv.reader(file, delimiter=",")
                first_row = next(csv_reader)[0]
                if first_row not in (
                    "PolarDiagramTable",
                    "PolarDiagramPointcloud",
                ):
                    raise FileReadingException(
                        f"hro-format for {first_row} not yet implemented"
                    )
                if first_row == "PolarDiagramTable":
                    ws_res, wa_res, bsps = _read_table(csv_reader)
                    return PolarDiagramTable(
                        ws_res=ws_res, wa_res=wa_res, bsps=bsps, tw=tw
                    )

                pts = _read_pointcloud(csv_reader)
                return PolarDiagramPointcloud(pts=pts, tw=tw)

            ws_res, wa_res, bsps = _read_extern_format(csv_path, fmt)
            return PolarDiagramTable(
                ws_res=ws_res, wa_res=wa_res, bsps=bsps, tw=tw
            )
    except OSError:
        raise FileReadingException(f"can't find/open/read {csv_path}")


def _read_table(csv_reader):
    next(csv_reader)
    ws_res = [eval(ws) for ws in next(csv_reader)]
    next(csv_reader)
    wa_res = [eval(wa) for wa in next(csv_reader)]
    next(csv_reader)
    bsps = []
    for row in csv_reader:
        bsps.append([eval(bsp) for bsp in row])

    return ws_res, wa_res, bsps


def _read_pointcloud(csv_reader):
    points = []
    next(csv_reader)
    for row in csv_reader:
        points.append([eval(entry) for entry in row])

    return np.array(points)


def _read_extern_format(csv_path, fmt):
    if fmt == "array":
        return _read_array_csv(csv_path)
    if fmt == "orc":
        delimiter = ";"
    else:
        delimiter = ","

    return _read_sail_csv(csv_path, delimiter)


def _read_sail_csv(csv_path, delimiter):
    with open(csv_path, "r", newline="") as file:
        csv_reader = csv.reader(file, delimiter=delimiter)
        ws_res = [eval(ws) for ws in next(csv_reader)[1:]]
        wa_res, bsps = [], []
        next(csv_reader)
        for row in csv_reader:
            wa_res.append(eval(row[0]))
            bsps.append([eval(bsp) if bsp != "" else 0 for bsp in row[1:]])

        return ws_res, wa_res, bsps


def _read_array_csv(csv_path):
    file_data = np.genfromtxt(csv_path, delimiter="\t")
    return file_data[0, 1:], file_data[1:, 0], file_data[1:, 1:]


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
    out : PolarDiagram
        PolarDiagram instance
        saved in .pkl file

    Function raises an exception
    if file can't be found,
    opened, or read
    """

    try:
        with open(pkl_path, "rb") as file:
            return pickle.load(file)
    except OSError:
        raise FileReadingException(f"Can't find/open/read {pkl_path}")


# TODO: Make it cleaner!
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
    out : PolarDiagram
        "symmetrized" version
        of input

    Function raises an exception
    if obj is not of type
    PolarDiagramTable or
    PolarDiagramPointcloud

    """
    if not isinstance(obj, (PolarDiagramTable, PolarDiagramPointcloud)):
        raise PolarDiagramException(
            f"Functionality for Type {type(obj)} not yet implemented"
        )

    if isinstance(obj, PolarDiagramPointcloud):
        sym_pts = obj.points

        if not sym_pts.size:
            return obj

        sym_pts[:, 1] = 360 - sym_pts[:, 1]
        pts = np.row_stack((obj.points, sym_pts))
        return PolarDiagramPointcloud(pts=pts)

    wa_res = np.concatenate([obj.wind_angles, 360 - np.flip(obj.wind_angles)])
    bsps = np.row_stack((obj.boat_speeds, np.flip(obj.boat_speeds, axis=0)))

    # deleting multiple 180° and 0°
    # occurences in the table
    if 180 in obj.wind_angles:
        wa_res = list(wa_res)
        h = int(len(wa_res) / 2)
        del wa_res[h]
        bsps = np.row_stack((bsps[:h, :], bsps[h + 1 :, :]))
    if 0 in obj.wind_angles:
        bsps = bsps[:-1, :]
        wa_res = wa_res[:-1]

    return PolarDiagramTable(
        ws_res=obj.wind_speeds, wa_res=wa_res, bsps=bsps, tw=True
    )


class PolarDiagram(ABC):
    """Base class for all
    polardiagram classes

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

        try:
            pickle.dump(self, pkl_path)
        except OSError:
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
    def polar_plot(
        self,
        ws_range,
        ax=None,
        colors=("green", "red"),
        show_legend=True,
        legend_kw=None,
        **plot_kw,
    ):
        pass

    @abstractmethod
    def flat_plot(
        self,
        ws_range,
        ax=None,
        colors=("green", "red"),
        show_legend=True,
        legend_kw=None,
        **plot_kw,
    ):
        pass

    @abstractmethod
    def plot_3d(self):
        pass

    @abstractmethod
    def plot_color_gradient(
        self,
        ax=None,
        colors=("green", "red"),
        marker=None,
        show_legend=True,
        **legend_kw,
    ):
        pass

    @abstractmethod
    def plot_convex_hull_slice(self, ws, ax=None, **plot_kw):
        pass

    @abstractmethod
    def plot_convex_hull_3d(self, ax=None, colors="blue"):
        pass


def _get_indices(ws, res):
    if ws is None:
        return range(len(res))

    if isinstance(ws, (int, float)):
        try:
            return [list(res).index(ws)]
        except ValueError:
            raise PolarDiagramException(f"{ws} is not contained in {res}")

    ws = set(ws)
    if not ws:
        raise PolarDiagramException("Empty slice-list was passed")

    if not ws.issubset(set(res)):
        raise PolarDiagramException(f"{ws} is not a subset of {res}")

    return [i for i, w in enumerate(res) if w in ws]


def _convert_wind(wind_arr, tw):
    if tw:
        return wind_arr

    return apparent_wind_to_true(wind_arr)


class PolarDiagramTable(PolarDiagram):
    """
    A class to represent,
    visualize and work with
    a polar diagram in the
    form of a table.

    Parameters
    ----------
    ws_res : array_like or int/float, optional
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

    wa_res : array_like or int/float, optional
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

    bsps : array_like, optional
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

    # TODO: Make it cleaner
    def __init__(self, ws_res=None, wa_res=None, bsps=None, tw=True):
        logger.info(
            f"Class 'PolarDiagramTable("
            f"ws_res={ws_res}, "
            f"wa_res={wa_res}, "
            f"bsps={bsps}, "
            f"tw={tw})' called"
        )

        ws_res = speed_resolution(ws_res)
        wa_res = angle_resolution(wa_res)

        rows = len(wa_res)
        cols = len(ws_res)
        if bsps is None:
            bsps = np.zeros((rows, cols))
        bsps = np.asarray(bsps)
        if not bsps.size:
            raise PolarDiagramException("")
        if bsps.ndim != 2:
            raise PolarDiagramException("")
        try:
            bsps = bsps.reshape(rows, cols)
        except ValueError:
            raise PolarDiagramException(
                f"bsps couldn't be broadcasted "
                f"to an array of shape "
                f"{(rows, cols)}"
            )

        ws_res, wa_res = np.meshgrid(ws_res, wa_res)
        ws_res = np.ravel(ws_res)
        wa_res = np.ravel(wa_res)
        bsps = np.ravel(bsps)
        wind_arr = _convert_wind(np.column_stack((ws_res, wa_res, bsps)), tw)

        self._res_wind_speed = np.array(sorted(list(set(wind_arr[:, 0]))))
        self._res_wind_angle = np.array(sorted(list(set(wind_arr[:, 1]))))
        self._boat_speeds = bsps.reshape(rows, cols)

    def __str__(self):
        table = "  TWA \\ TWS"
        bsps = self.boat_speeds
        if len(self.wind_speeds) <= 15:
            wind = self.wind_speeds
            for ws in wind:
                table += f"    {float(ws):.1f}"
            table += "\n-----------"
            for ws in wind:
                le = len(f"{float(ws):.1f}")
                table += "  ".ljust(le + 4, "-")
            table += "\n"
            for i, wa in enumerate(self.wind_angles):
                angle = f"{float(wa):.1f}"
                table += angle.ljust(11)
                for j, ws in enumerate(wind):
                    entry = f"{bsps[i][j]:.2f}"
                    le = len(str(ws))
                    table += entry.rjust(4 + le)
                table += "\n"
            return table

        wind = []
        wind.extend(self.wind_speeds[:5])
        wind.extend(self.wind_speeds[-5:])
        for i, ws in enumerate(wind):
            if i == 5:
                table += "  ..."
            table += f"    {float(ws):.1f}"
        table += "\n-----------"
        for i, ws in enumerate(wind):
            if i == 5:
                table += "  ---"
            le = len(f"{float(ws):.1f}")
            table += "  ".ljust(le + 4, "-")
        table += "\n"
        for i, wa in enumerate(self.wind_angles):
            angle = f"{float(wa):.1f}"
            table += angle.rjust(11)
            for j, ws in enumerate(wind):
                if j == 5:
                    table += "  ..."
                entry = f"{bsps[i][j]:.2f}"
                le = len(str(ws))
                table += entry.rjust(4 + le)
            table += "\n"
        return table

    def __repr__(self):
        return (
            f"PolarDiagramTable("
            f"wind_speed_resolution={self.wind_speeds}, "
            f"wind_angle_resolution={self.wind_angles}, "
            f"data={self.boat_speeds})"
        )

    def __getitem__(self, key):
        ws, wa = key
        col = _get_indices(ws, self.wind_speeds)
        row = _get_indices(wa, self.wind_angles)
        return self.boat_speeds[row, col]

    @property
    def wind_angles(self):
        """Returns a read only version of
        self._resolution_wind_angle"""
        return self._res_wind_angle.copy()

    @property
    def wind_speeds(self):
        """Returns a read only version of
        self._resolution_wind_speed"""
        return self._res_wind_speed.copy()

    @property
    def boat_speeds(self):
        """Returns a read only version of
        self._data"""
        return self._boat_speeds.copy()

    def to_csv(self, csv_path, fmt="hro"):
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

        fmt : string

        Function raises an exception
        if file can't be written to
        """
        logger.info(f"Method '.to_csv({csv_path})' called")

        try:
            with open(csv_path, "w", newline="") as file:
                csv_writer = csv.writer(file, delimiter=",")
                if fmt == "opencpn":
                    csv_writer.writerow(["TWA\\TWS"] + self.wind_speeds)
                    rows = np.column_stack(
                        (self.wind_angles, self.boat_speeds)
                    )
                    csv_writer.writerows(rows)

                csv_writer.writerow(["PolarDiagramTable"])
                csv_writer.writerow(["Wind speed resolution:"])
                csv_writer.writerow(self.wind_speeds)
                csv_writer.writerow(["Wind angle resolution:"])
                csv_writer.writerow(self.wind_angles)
                csv_writer.writerow(["Boat speeds:"])
                csv_writer.writerows(self.boat_speeds)
        except OSError:
            raise FileReadingException(f"Can't write to {csv_path}")

    def change_entries(self, new_bsps, ws=None, wa=None):
        """Changes specified entries
         in the table

        Parameters
        ----------
        new_bsps: array_like
            Sequence containing the
            new boat speeds to be inserted
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
        logger.info(
            f"Method "
            f"'PolarDiagramTable.change_entries("
            f"new_bsps={new_bsps},"
            f"ws={ws}, wa={wa}) called"
        )

        new_bsps = np.asarray(new_bsps)
        if not new_bsps.size:
            raise PolarDiagramException(f"No new data was passed")

        ws_ind = _get_indices(ws, self.wind_speeds)
        wa_ind = _get_indices(wa, self.wind_angles)

        mask = np.zeros(self.boat_speeds.shape, dtype=bool)
        for i in wa_ind:
            for j in ws_ind:
                mask[i, j] = True
        try:
            new_bsps = new_bsps.reshape(len(wa_ind), len(ws_ind))
        except ValueError:
            raise PolarDiagramException(
                f"{new_bsps} couldn't be "
                f"broadcasted to an "
                f"array of shape "
                f"{(len(wa_ind), len(ws_ind))}"
            )

        self._boat_speeds[mask] = new_bsps.flat

    def _get_slice_data(self, ws):
        ind = _get_indices(ws, self.wind_speeds)
        return self.boat_speeds[:, ind]

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
        logger.info(
            f"Method 'polar_plot_slice("
            f"ws={ws}, ax={ax}, "
            f"plot_kw={plot_kw})' called"
        )

        wa = self._get_radians()
        bsp = self._get_slice_data(ws)
        plot_polar(wa, bsp, ax, **plot_kw)

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
        logger.info(
            f"Method 'flat_plot_slice("
            f"ws={ws}, ax={ax}, "
            f"plot_kw={plot_kw})' called"
        )

        bsp = self._get_slice_data(ws)
        wa = self.wind_angles
        plot_flat(wa, bsp, ax, **plot_kw)

    def polar_plot(
        self,
        ws_range=None,
        ax=None,
        colors=("green", "red"),
        show_legend=True,
        legend_kw=None,
        **plot_kw,
    ):
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
        logger.info(
            f"Method 'polar_plot("
            f"ws_range={ws_range}, "
            f"ax={ax}, colors={colors}, "
            f"show_legend={show_legend},"
            f"legend_kw={legend_kw}, "
            f"plot_kw={plot_kw})' called"
        )

        if ws_range is None:
            ws_range = self.wind_speeds

        # TODO Better way?
        if isinstance(ws_range, np.ndarray):
            if not ws_range.size:
                raise PolarDiagramException(
                    "ws_range doesn't contain any slices"
                )
        elif not ws_range:
            raise PolarDiagramException("ws_range doesn't contain any slices")

        bsp_list = list(self._get_slice_data(ws_range).T)
        wa_list = [list(self._get_radians())] * len(bsp_list)
        plot_polar_range(
            ws_range,
            wa_list,
            bsp_list,
            ax,
            colors,
            show_legend,
            legend_kw,
            **plot_kw,
        )

    def flat_plot(
        self,
        ws_range=None,
        ax=None,
        colors=("green", "red"),
        show_legend=True,
        legend_kw=None,
        **plot_kw,
    ):
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
        logger.info(
            f"Method 'flat_plot("
            f"ws_range={ws_range}, "
            f"ax={ax}, colors={colors}, "
            f"show_legend={show_legend},"
            f"legend_kw={legend_kw}, "
            f"plot_kw={plot_kw})' called"
        )

        if ws_range is None:
            ws_range = self.wind_speeds

        # TODO Better way
        if isinstance(ws_range, np.ndarray):
            if not ws_range.size:
                raise PolarDiagramException(
                    "ws_range doesn't contain any slices"
                )
        elif not ws_range:
            raise PolarDiagramException("ws_range doesn't contain any slices")

        bsp_list = list(self._get_slice_data(ws=ws_range).T)
        wa_list = [list(self.wind_angles)] * len(bsp_list)

        plot_flat_range(
            ws_range,
            wa_list,
            bsp_list,
            ax,
            colors,
            show_legend,
            legend_kw,
            **plot_kw,
        )

    def plot_3d(self, ax=None, colors=("blue", "blue")):
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
        logger.info(f"Method 'plot_3d(ax={ax}, colors={colors})' called")

        ws, wa = np.meshgrid(self.wind_speeds, self._get_radians())
        bsp = self.boat_speeds
        bsp, wa = bsp * np.cos(wa), bsp * np.sin(wa)
        plot_surface(ws, wa, bsp, ax, colors)

    def plot_color_gradient(
        self,
        ax=None,
        colors=("green", "red"),
        marker=None,
        show_legend=True,
        **legend_kw,
    ):
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
        logger.info(
            f"Method 'plot_color_gradient("
            f"ax={ax}, colors={colors}, "
            f"marker={marker}, "
            f"show_legend={show_legend}, "
            f"legend_kw={legend_kw})' "
            f"called"
        )

        ws, wa = np.meshgrid(self.wind_speeds, self.wind_angles)
        ws = np.ravel(ws)
        wa = np.ravel(wa)
        bsp = np.ravel(self.boat_speeds)

        plot_color(ws, wa, bsp, ax, colors, marker, show_legend, **legend_kw)

    def plot_convex_hull_slice(self, ws, ax=None, **plot_kw):
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
        logger.info(
            f"Method "
            f"'plot_convex_hull_slice("
            f"ws={ws}, ax={ax}, "
            f"plot_kw={plot_kw})' called"
        )

        wa = list(self._get_radians())
        bsp = self._get_slice_data(ws)
        plot_convex_hull(wa, bsp, ax, **plot_kw)

    # Still very much in development
    # Don't use
    def plot_convex_hull_3d(self, ax=None, color="blue"):
        """"""
        logger.info(
            f"Method "
            f"'plot_convex_hull_3d("
            f"ax={ax}, "
            f"color={color})' "
            f"called"
        )

        ws, wa = np.meshgrid(self.wind_speeds, self._get_radians())
        bsp = self.boat_speeds
        bsp, wa = bsp * np.cos(wa), bsp * np.sin(wa)

        plot_convex_surface(ws, wa, bsp, ax, color)


class PolarDiagramMultiSails(PolarDiagram):
    def __init__(self, polar_tables=None):
        if polar_tables is None:
            polar_tables = [PolarDiagramTable()]

        self._sails = polar_tables

        self._res_wind_speed = polar_tables[0].wind_speeds
        self._res_wind_angle = [pt.wind_angles for pt in polar_tables]

    @property
    def wind_speeds(self):
        return self._res_wind_speed.copy()

    @property
    def wind_angles(self):
        return [wa.copy() for wa in self._res_wind_angle]

    @property
    def boat_speeds(self):
        return [pt.boat_speed for pt in self._sails]

    def to_csv(self, csv_path):
        pass

    def _get_radians(self):
        return [np.deg2rad(wa) for wa in self.wind_angles]

    def _get_slice_data(self, ws):
        ind = _get_indices(ws, self.wind_speeds)
        return [bsp[:, ind].ravel() for bsp in self.boat_speeds]

    def polar_plot_slice(self, ws, ax=None, **plot_kw):
        wa = self._get_radians()
        bsp = self._get_slice_data(ws)

        plot_polar(wa, bsp, ax, **plot_kw)

    def flat_plot_slice(self, ws, ax=None, **plot_kw):
        wa = self.wind_angles
        bsp = self._get_slice_data(ws)

        plot_flat(wa, bsp, ax, **plot_kw)

    # TODO: Implementation
    #       Problems: How to handle legend?
    #                 Multiple legends?
    def polar_plot(
        self,
        ws_range,
        ax=None,
        colors=("green", "red"),
        show_legend=True,
        legend_kw=None,
        **plot_kw,
    ):
        pass

    # TODO: Implementation
    #       Problems: How to handle legend?
    #                 Multiple legends?
    def flat_plot(
        self,
        ws_range,
        ax=None,
        colors=("green", "red"),
        show_legend=True,
        legend_kw=None,
        **plot_kw,
    ):
        pass

    def plot_3d(self, ax=None, colors=None):
        if colors is None:
            colors = [("blue", "blue"), ("blue", "blue")]

        self._sail_1.plot_3d(ax, colors=colors[0])
        self._sail_2.plot_3d(ax, colors=colors[1])

    # TODO: Implementation
    #       Problems: Can we make two color gradients
    #                 for the two sails? How?
    #                 How to handle legend?
    #                 Multiple legends?
    #                 Change helper function?
    def plot_color_gradient(
        self,
        ax=None,
        colors=("green", "red"),
        marker=None,
        show_legend=True,
        **legend_kw,
    ):
        pass

    # TODO: Works but it would be nice,
    #       to see which parts of the
    #       convex hull belong to the
    #       convex hull of one sail,
    #       which belong to the convex
    #       hull of the other, and
    #       which are a combination
    #       of those...
    #       Mutliple colors and change
    #       helper function, or create
    #       seperate helper function?
    def plot_convex_hull_slice(self, ws, ax=None, **plot_kw):
        wa = np.concatenate(self._get_radians())
        bsp = np.concatenate(self._get_slice_data(ws))

        plot_convex_hull(wa, bsp, ax, **plot_kw)

    def plot_convex_hull_3d(self, ax=None, colors=("blue",)):
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

    def __init__(self, f, params, radians=False):
        if not callable(f):
            raise PolarDiagramException(f"{f.__name__} is not callable")

        logger.info(
            f"Class 'PolarDiagramCurve("
            f"f={f.__name__}, {params}, "
            f"radians = {radians})'"
            f"called"
        )

        self._f = f
        self._params = params
        self._rad = radians

    def __repr__(self):
        return (
            f"PolarDiagramCurve("
            f"f={self._f.__name__},"
            f"{self._params},"
            f"radians={self._rad}"
        )

    def __call__(self, ws, wa):
        return self.curve(np.column_stack((ws, wa)), *self.parameters)

    @property
    def curve(self):
        """Returns a read only version
        of self._f"""
        return self._f

    @property
    def parameters(self):
        """Returns a read only version
        of self._params"""
        return self._params.copy()

    @property
    def radians(self):
        """Returns a read only version
        of self._rad"""
        return self._rad.copy()

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
        logger.info(f"Method '.to_csv({csv_path})' called")

        try:
            with open(csv_path, "w", newline="") as file:
                csv_writer = csv.writer(file, delimiter=":")
                csv_writer.writerow(["PolarDiagramCurve"])
                csv_writer.writerow(["Function"] + [self.curve.__name__])
                csv_writer.writerow(["Radians"] + [str(self.radians)])
                csv_writer.writerow(["Parameters"] + list(self.parameters))
        except OSError:
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
        logger.info(
            f"Method 'polar_plot_slice("
            f"ws={ws}, ax={ax}, "
            f"plot_kw={plot_kw})' called"
        )

        wa = self._get_wind_angles()
        bsp = self(np.array([ws] * 1000), wa)

        if not self.radians:
            wa = np.deg2rad(wa)

        plot_polar(wa, bsp, ax, **plot_kw)

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
        logger.info(
            f"Method 'flat_plot_slice("
            f"ws={ws}, ax={ax}, "
            f"plot_kw={plot_kw})' called"
        )

        wa = self._get_wind_angles()
        bsp = self(np.array([ws] * 1000), wa)

        if self.radians:
            wa = np.rad2deg(wa)

        plot_flat(wa, bsp, ax, **plot_kw)

    def polar_plot(
        self,
        ws_range=(0, 20, 5),
        ax=None,
        colors=("green", "red"),
        show_legend=True,
        legend_kw=None,
        **plot_kw,
    ):
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
        logger.info(
            f"Method 'polar_plot("
            f"ws_range={ws_range}, "
            f"ax={ax}, colors={colors},"
            f"show_legend={show_legend}, "
            f"legend_kw={legend_kw},"
            f"**plot_kw={plot_kw})' called"
        )

        if isinstance(ws_range, tuple):
            ws_lower, ws_upper, ws_step = ws_range
            ws_range = list(np.linspace(ws_lower, ws_upper, ws_step))

        wa = self._get_wind_angles()
        if self.radians:
            wa_list = [wa] * len(ws_range)
        else:
            wa_list = [np.deg2rad(wa)] * len(ws_range)
        bsp_list = []
        for ws in ws_range:
            bsp_list.append(self(np.array([ws] * 1000), wa))

        plot_polar_range(
            ws_range,
            wa_list,
            bsp_list,
            ax,
            colors,
            show_legend,
            legend_kw,
            **plot_kw,
        )

    def flat_plot(
        self,
        ws_range=(0, 20, 5),
        ax=None,
        colors=("green", "red"),
        show_legend=True,
        legend_kw=None,
        **plot_kw,
    ):
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
        logger.info(
            f"Method 'polar_plot("
            f"ws_range={ws_range}, "
            f"ax={ax}, colors={colors},"
            f"show_legend={show_legend}, "
            f"legend_kw={legend_kw},"
            f"**plot_kw={plot_kw})' called"
        )

        if isinstance(ws_range, tuple):
            ws_lower, ws_upper, ws_step = ws_range
            ws_range = list(np.linspace(ws_lower, ws_upper, ws_step))

        wa = self._get_wind_angles()
        if self.radians:
            wa_list = [np.rad2deg(wa)] * len(ws_range)
        else:
            wa_list = [wa] * len(ws_range)

        bsp_list = []
        for ws in ws_range:
            bsp_list.append(self(np.array([ws] * 1000), wa))

        plot_flat_range(
            ws_range,
            wa_list,
            bsp_list,
            ax,
            colors,
            show_legend,
            legend_kw,
            **plot_kw,
        )

    def plot_3d(self, ws_range=(0, 20, 100), ax=None, colors=("blue", "blue")):
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
        logging.info(
            f"Method 'plot_3d("
            f"ws_range={ws_range}, "
            f"ax={ax}, "
            f"colors={colors})' "
            f"called"
        )

        ws_lower, ws_upper, ws_step = ws_range
        ws = np.linspace(ws_lower, ws_upper, ws_step)
        wa = self._get_wind_angles()

        ws_arr, wa_arr = np.meshgrid(ws, wa)
        bsp_arr = []
        for w in ws:
            bsp_arr.append(self(np.array([w] * 1000), wa))
        bsp_arr = np.asarray(bsp_arr)

        if self.radians:
            bsp, wa = (bsp_arr * np.cos(wa_arr), bsp_arr * np.sin(wa_arr))
        else:
            wa_arr = np.deg2rad(wa_arr)
            bsp, wa = (bsp_arr * np.cos(wa_arr), bsp_arr * np.sin(wa_arr))

        plot_surface(ws_arr, wa, bsp, ax, colors)

    def plot_color_gradient(
        self,
        ws_range=(0, 20, 100),
        ax=None,
        colors=("green", "red"),
        marker=None,
        show_legend=True,
        **legend_kw,
    ):
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

        marker : matplotlib.markers.Markerstyle or equivalent, optional
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
        logger.info(
            f"Method 'plot_color_gradient("
            f"ws_range={ws_range}, ax={ax}, "
            f"colors={colors}, "
            f"marker={marker}, "
            f"show_legend={show_legend}, "
            f"**legend_kw={legend_kw})' "
            f"called"
        )

        ws_lower, ws_upper, ws_step = ws_range
        ws, wa = np.meshgrid(
            np.linspace(ws_lower, ws_upper, ws_step), np.linspace(0, 360, 1000)
        )
        ws = np.ravel(ws)
        wa = np.ravel(wa)

        if self.radians:
            bsp = np.ravel(self(ws, np.deg2rad(wa)))
        else:
            bsp = np.ravel(self(ws, wa))

        plot_color(ws, wa, bsp, ax, colors, marker, show_legend, **legend_kw)

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
        logger.info(
            f"Method 'plot_convex_hull_slice("
            f"ws={ws}, ax={ax}, "
            f"**plot_kw={plot_kw})' called"
        )

        wa = self._get_wind_angles()
        bsp = self(np.array([ws] * 1000), wa)

        if not self.radians:
            wa = np.deg2rad(wa)

        plot_convex_hull(wa, bsp, ax, **plot_kw)

    def plot_convex_hull_3d(self, ax=None, colors=None):
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
    pts : array_like, optional
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
    add_points(new_pts)
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

    def __init__(self, pts=None, tw=True):
        logger.info(
            f"Class 'PolarDiagramPointcloud(pts={pts}, tw={tw})' called"
        )

        if pts is None:
            self._pts = np.array([])
            return

        pts = np.asarray(pts)
        if not pts.size:
            self._pts = np.array([])
            return

        try:
            pts = pts.reshape(-1, 3)
        except ValueError:
            raise PolarDiagramException(
                f"{pts} could not be "
                f"broadcasted to an "
                f"array of shape (n,3)"
            )

        self._pts = _convert_wind(pts, tw)

    def __str__(self):
        table = "   TWS      TWA     BSP\n"
        table += "------  -------  ------\n"
        for point in self.points:
            for i in range(3):
                entry = f"{float(point[i]):.2f}"
                if i == 1:
                    table += entry.rjust(7)
                    table += "  "
                    continue

                table += entry.rjust(6)
                table += "  "
            table += "\n"
        return table

    def __repr__(self):
        return f"PolarDiagramPointcloud(pts={self.points})"

    @property
    def wind_speeds(self):
        """Returns a list of all the different
        wind speeds in the point cloud"""
        return sorted(list(set(self.points[:, 0])))

    @property
    def wind_angles(self):
        """Returns a list of all the different
        wind angles in the point cloud"""
        return sorted(list(set(self.points[:, 1])))

    @property
    def points(self):
        """Returns a read only version
        of self._pts"""
        return self._pts.copy()

    def to_csv(self, csv_path):
        """Creates a .csv file with
        delimiter ',' and the
        following format:
            PolarDiagramPointcloud
            True wind speed ,True wind angle ,Boat speed
            self.points

        Parameters
        ----------
        csv_path : string
            Path where a .csv-file is
            located or where a new
            .csv file will be created

        Function raises an exception
        if file can't be written to
        """
        logger.info(f"Method '.to_csv({csv_path})' called")

        try:
            with open(csv_path, "w", newline="") as file:
                csv_writer = csv.writer(file, delimiter=",")
                csv_writer.writerow(["PolarDiagramPointcloud"])
                csv_writer.writerow(
                    ["True wind speed ", "True wind angle ", "Boat speed "]
                )
                csv_writer.writerows(self.points)
        except OSError:
            raise FileReadingException(f"Can't write to {csv_path}")

    def add_points(self, new_pts, tw=True):
        """Adds additional
        points to the point cloud

        Parameters
        ----------
        new_pts: array_like
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
        exception if new_pts
        can't be broadcasted
        to a fitting shape
        """
        logger.info(f"Method 'add_points(new_pts{new_pts}, tw={tw})' called")

        new_pts = np.asarray(new_pts)
        if not new_pts.size:
            raise PolarDiagramException(f"new_pts is an empty array")
        try:
            new_pts = new_pts.reshape(-1, 3)
        except ValueError:
            raise PolarDiagramException(
                f"{new_pts} could not be "
                f"broadcasted to an array "
                f"of shape (n,3)"
            )

        new_pts = _convert_wind(new_pts, tw)

        if not self.points.size:
            self._pts = new_pts
            return

        self._pts = np.row_stack((self.points, new_pts))

    def _get_slice_data(self, ws):
        pts = self.points[self.points[:, 0] == ws][:, 1:]
        if not pts.size:
            raise PolarDiagramException(
                f"No points with wind speed={ws} found"
            )

        return pts[:, 0], pts[:, 1]

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
        logger.info(
            f"Method 'polar_plot_slice("
            f"ws={ws}, ax={ax}, "
            f"**plot_kw={plot_kw})' called"
        )

        wa, bsp = self._get_slice_data(ws)
        wa = list(np.deg2rad(wa))

        plot_polar(wa, bsp, ax, **plot_kw)

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
        logger.info(
            f"Method 'flat_plot_slice("
            f"ws={ws}, ax={ax}, "
            f"**plot_kw={plot_kw})' called"
        )

        wa, bsp = self._get_slice_data(ws)

        plot_flat(wa, bsp, ax, **plot_kw)

    def _get_slices(self, ws_range):
        wa_list, bsp_list = [], []

        if not ws_range:
            raise PolarDiagramException(f"No slices given")

        if isinstance(ws_range, tuple):
            ws_lower, ws_upper = ws_range
            ws_range = [
                ws for ws in self.wind_speeds if ws_lower <= ws <= ws_upper
            ]

        for ws in ws_range:
            pts = self._get_slice_data(ws)
            wa_list.append(pts[0])
            bsp_list.append(pts[1])

        return ws_range, wa_list, bsp_list

    def polar_plot(
        self,
        ws_range=(0, np.inf),
        ax=None,
        colors=("green", "red"),
        show_legend=True,
        legend_kw=None,
        **plot_kw,
    ):
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
        logger.info(
            f"Method 'polar_plot("
            f"ws_range={ws_range}, "
            f"ax={ax}, colors={colors}, "
            f"show_legend={show_legend}, "
            f"legend_kw={legend_kw}, "
            f"**plot_kw={plot_kw})' "
            f"called"
        )

        ws_list, wa_list, bsp_list = self._get_slices(ws_range)
        wa_list = [np.deg2rad(wa) for wa in wa_list]

        plot_polar_range(
            ws_list,
            wa_list,
            bsp_list,
            ax,
            colors,
            show_legend,
            legend_kw,
            **plot_kw,
        )

    def flat_plot(
        self,
        ws_range=(0, np.inf),
        ax=None,
        colors=("green", "red"),
        show_legend=True,
        legend_kw=None,
        **plot_kw,
    ):
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
        logger.info(
            f"Method 'flat_plot("
            f"ws_range={ws_range}, "
            f"ax={ax}, colors={colors}, "
            f"show_legend={show_legend}, "
            f"legend_kw={legend_kw}, "
            f"**plot_kw={plot_kw})' "
            f"called"
        )

        ws_list, wa_list, bsp_list = self._get_slices(ws_range)

        plot_flat_range(
            ws_list,
            wa_list,
            bsp_list,
            ax,
            colors,
            show_legend,
            legend_kw,
            **plot_kw,
        )

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
        logger.info(
            f"Method 'plot_3d("
            f"ax={ax}, "
            f"**plot_kw={plot_kw})' "
            f"called"
        )

        ws, wa, bsp = (self.points[:, 0], self.points[:, 1], self.points[:, 2])
        wa = np.deg2rad(wa)
        bsp, wa = (bsp * np.cos(wa), bsp * np.sin(wa))
        plot3d(ws, wa, bsp, ax, **plot_kw)

    def plot_color_gradient(
        self,
        ax=None,
        colors=("green", "red"),
        marker=None,
        show_legend=True,
        **legend_kw,
    ):
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
        logger.info(
            f"Method 'plot_color_gradient("
            f"ax={ax}, colors={colors}, "
            f"marker={marker}, "
            f"show_legend={show_legend}, "
            f"**legend_kw={legend_kw})' "
            f"called"
        )

        ws, wa, bsp = np.hsplit(self.points, 3)

        plot_color(ws, wa, bsp, ax, colors, marker, show_legend, **legend_kw)

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
        logger.info(
            f"Method 'plot_convex_hull_slice("
            f"ws={ws}, ax={ax}, "
            f"**plot_kw={plot_kw})' called"
        )

        wa, bsp = self._get_slice_data(ws)
        wa = list(np.deg2rad(wa))
        plot_convex_hull(wa, bsp, ax, **plot_kw)

    def plot_convex_hull_3d(self, ax=None, colors=None):
        """"""
        pass
