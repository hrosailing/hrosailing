"""
Classes to represent polar diagrams in various different forms
as well as small functions to save / load PolarDiagram-objects to files
in different forms and functions to manipulate PolarDiagram-objects
"""

# Author: Valentin F. Dannenberg / Ente


from abc import ABC, abstractmethod
import csv
import itertools
import logging.handlers
import pickle
from typing import List


from hrosailing.polardiagram.plotting import *
from hrosailing.wind import WindException, convert_wind, set_resolution

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
    """Custom exception for errors that may appear whilst handling
    polar diagrams"""

    pass


class FileReadingException(Exception):
    """Custom exception for errors that may appear whilst reading a file"""

    pass


class FileWritingException(Exception):
    """Custom exception for errors that may appear whilst writing to a file"""

    pass


def to_csv(csv_path, obj):
    """See also the to_csv()-method of PolarDiagram

    Parameters
    ----------
    csv_path : path-like
        Path to a .csv-file or where a new .csv file will be created

    obj : PolarDiagram
        PolarDiagram instance which will be written to .csv file

    Raises a FileWritingException if an error occurs whilst writing
    """
    obj.to_csv(csv_path)


def from_csv(csv_path, fmt="hro", tw=True):
    """Reads a .csv file and returns the PolarDiagram
    instance contained in it

    Parameters
    ----------
    csv_path : path-like
        Path to a .csv file

    fmt : string
        The "format" of the .csv file.
                - hro: format created by the to_csv-method of the
                PolarDiagram class
                - orc: format found at
                `ORC <https://jieter.github.io/orc-data/site/>`_
                - opencpn: format created by `OpenCPN Polar Plugin
                <https://opencpn.org/OpenCPN/plugins/polar.html>`_
                - array

    tw : bool
        Specifies if wind data in file should be viewed as true wind

        Defaults to True

    Returns
    -------
    out : PolarDiagram
        PolarDiagram instance contained in the .csv file

    Raises a FileReadingException
        - if an unknown format was specified
        - if an error occurs whilst reading
    """
    if fmt not in {"array", "hro", "opencpn", "orc"}:
        raise FileReadingException("`fmt` not implemented")
    try:
        with open(csv_path, "r", newline="") as file:
            if fmt == "hro":
                csv_reader = csv.reader(file, delimiter=",")
                first_row = next(csv_reader)[0]
                if first_row not in {
                    "PolarDiagramTable",
                    "PolarDiagramPointcloud",
                }:
                    raise FileReadingException(
                        f"hro-format for {first_row} not implemented"
                    )
                if first_row == "PolarDiagramTable":
                    ws_res, wa_res, bsps = _read_table(csv_reader)
                    return PolarDiagramTable(
                        ws_res=ws_res, wa_res=wa_res, bsps=bsps
                    )

                pts = _read_pointcloud(csv_reader)
                return PolarDiagramPointcloud(pts=pts, tw=tw)

            ws_res, wa_res, bsps = _read_extern_format(file, fmt)
            return PolarDiagramTable(ws_res=ws_res, wa_res=wa_res, bsps=bsps)
    except OSError as oe:
        raise FileReadingException(
            "While reading `csv_path` an error occured"
        ) from oe


def _read_table(csv_reader):
    next(csv_reader)
    ws_res = [eval(ws) for ws in next(csv_reader)]
    next(csv_reader)
    wa_res = [eval(wa) for wa in next(csv_reader)]
    next(csv_reader)
    bsps = [[eval(bsp) for bsp in row] for row in csv_reader]

    return ws_res, wa_res, bsps


def _read_pointcloud(csv_reader):
    next(csv_reader)
    return np.array([[eval(pt) for pt in row] for row in csv_reader])


def _read_extern_format(file, fmt):
    if fmt == "array":
        return _read_array_csv(file)

    delimiter = ","
    if fmt == "orc":
        delimiter = ";"

    return _read_sail_csv(file, delimiter)


def _read_sail_csv(file, delimiter):
    csv_reader = csv.reader(file, delimiter=delimiter)
    ws_res = [eval(ws) for ws in next(csv_reader)[1:]]
    if delimiter == ";":
        next(csv_reader)
    wa_res, bsps = list(
        zip(
            *(
                [
                    (
                        # delete °-symbol in case of opencpn format
                        eval(row[0].replace("°", "")),
                        [eval(bsp) if bsp != "" else 0 for bsp in row[1:]],
                    )
                    for row in csv_reader
                ]
            )
        )
    )
    return ws_res, wa_res, bsps


def _read_array_csv(file):
    file_data = np.genfromtxt(file, delimiter="\t")
    return file_data[0, 1:], file_data[1:, 0], file_data[1:, 1:]


def pickling(pkl_path, obj):
    """See also the pickling()-method of PolarDiagram

    Parameters
    ----------
    pkl_path : path-like
        Path to a .pkl file or where a new .pkl file will be created

    obj : PolarDiagram
        PolarDiagram instance which will be written to .csv file

    Raises a FileWritingException if an error occurs whilst writing
    """
    obj.pickling(pkl_path)


def depickling(pkl_path):
    """Reads a .pkl file and returns the PolarDiagram
    instance contained in it.

    Parameters
    ----------
    pkl_path : path-like
        Path to a .pkl file

    Returns
    -------
    out : PolarDiagram
        PolarDiagram instance contained in the .pkl file

    Raises a FileReadingException if an error occurs whilst reading
    """
    try:
        with open(pkl_path, "rb") as file:
            return pickle.load(file)
    except OSError as oe:
        raise FileReadingException(
            "While reading `pkl_path` an error occured"
        ) from oe


def symmetric_polar_diagram(obj):
    """See also the symmetrize()-method of PolarDiagram

    Parameters
    ----------
    obj : PolarDiagram
        PolarDiagram instance which will be symmetrized

    Returns
    -------
    out : PolarDiagram
        "symmetrized" version of input
    """
    return obj.symmetrize()


# TODO Add support for format strings in plot methods?


class PolarDiagram(ABC):
    """Base class for all polardiagram classes


    Methods
    -------
    pickling(pkl_path)
        Writes PolarDiagram instance to a .pkl file
    polar_plot_slice(ws, ax=None, **plot_kw)
        Creates a polar plot  of a given slice of the
        polar diagram
    flat_plot_slice(ws, ax=None, **plot_kw)
        Creates a cartesian plot of a given slice of the
        polar diagram
    plot_convex_hull_slice(ws, ax=None, **plot_kw)
        Computes the convex hull of a given slice of the
        polar diagram and creates a polar plot of it


    Abstract Methods
    ----------------
    to_csv(csv_path)
    symmetrize()
    get_slices(ws)
    plot_polar(
        ws,
        ax=None,
        colors=('green', 'red'),
        show_legend=False,
        legend_kw=None,
        **plot_kw
    )
    plot_flat(
        ws,
        ax=None,
        colors=('green', 'red'),
        show_legend=False,
        legend_kw=None,
        **plot_kw
    )
    plot_3d(ax=None, **plot_kw)
    plot_color_gradient(
        ax=None,
        colors=('green', 'red'),
        marker=None,
        show_legend=False,
        **legend_kw,
    )
    plot_convex_hull(
        ws,
        ax=None,
        colors=('green', 'red'),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    )
    """

    def pickling(self, pkl_path):
        """Writes PolarDiagram instance to a .pkl file

        Parameters
        ----------
        pkl_path: path-like
            Path to a .pkl file or where a new .pkl file will be created

        Raises a FileWritingException if an error occurs whilst writing
        """
        try:
            with open(pkl_path, "wb") as file:
                pickle.dump(self, file)
        except OSError as oe:
            raise FileWritingException(
                "While writing to `pkl_path` an error occured"
            ) from oe

    @abstractmethod
    def to_csv(self, csv_path):
        pass

    @abstractmethod
    def symmetrize(self):
        pass

    @abstractmethod
    def get_slices(self, ws):
        pass

    def plot_polar_slice(self, ws, ax=None, **plot_kw):
        """Creates a polar plot of a given slice of the
        polar diagram

        Parameters
        ----------
        ws : int/float
            Slice of the polar diagram

            For a description of what the slice is made of,
            see the plot_polar()-method of the respective
            PolarDiagram subclasses

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created

            If nothing is passed, the function will create
            a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot


        Raises a PolarDiagramException, if the plot_polar()-method
        of the respective PolarDiagram subclass raises one
        """
        logger.info(
            f"Method 'polar_plot_slice(ws={ws}, ax={ax}, "
            f"plot_kw={plot_kw})' called"
        )

        self.plot_polar(ws, ax, None, False, None, **plot_kw)

    def plot_flat_slice(self, ws, ax=None, **plot_kw):
        """Creates a cartesian plot of a given slice of the
        polar diagram

        Parameters
        ----------
        ws : int/float
            Slice of the polar diagram

            For a description of what the slice is made of,
            see the plot_flat()-method of the respective
            PolarDiagram subclass

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot

        Raises a PolarDiagramException, if the plot_flat()-method
        of the respective PolarDiagram subclass raises one
        """
        logger.info(
            f"Method 'flat_plot_slice(ws={ws}, ax={ax}, "
            f"plot_kw={plot_kw})' called"
        )

        self.plot_flat(ws, ax, None, False, None, **plot_kw)

    @abstractmethod
    def plot_polar(
        self,
        ws,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        pass

    @abstractmethod
    def plot_flat(
        self,
        ws,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
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
        show_legend=False,
        **legend_kw,
    ):
        pass

    def plot_convex_hull_slice(self, ws, ax=None, **plot_kw):
        """Computes the convex hull of a given slice of
        the polar diagram and creates a polar plot of it

        Parameters
        ----------
        ws : int/float
            Slice of the polar diagram

            For a description of what the slice is made of,
            see the plot_convex_hull()-method of the respective
            PolarDiagram subclass

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot

        Raises a PolarDiagramException, if the plot_convex_hull()-method
        of the respective PolarDiagram subclass raises one
        """
        logger.info(
            f"Method 'plot_convex_hull_slice(ws={ws}, ax={ax}, "
            f"plot_kw={plot_kw})' called"
        )

        self.plot_convex_hull(ws, ax, None, False, None, **plot_kw)

    @abstractmethod
    def plot_convex_hull(
        self,
        ws,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        pass


def _get_indices(wind, res):
    if wind is None:
        return range(len(res))

    if isinstance(wind, (int, float)):
        try:
            return [list(res).index(wind)]
        except ValueError:
            raise PolarDiagramException(f"{wind} is not contained in {res}")

    wind = set(wind)
    if not wind:
        raise PolarDiagramException("Empty slice-list was passed")

    if not wind.issubset(set(res)):
        raise PolarDiagramException(f"{wind} is not a subset of {res}")

    return [i for i, w in enumerate(res) if w in wind]


# TODO: Standardize wind angles, such that they are in [0, 360),
#       because 360° should be equal to 0°
class PolarDiagramTable(PolarDiagram):
    """A class to represent, visualize and work with
    a polar diagram in the form of a table.

    Parameters
    ----------
    ws_res : array_like or positive int/float, optional
        Wind speeds that will correspond to the columns
        of the table

        Can either be a sequence of length cdim or an
        int/float value

        If a number num is passed, numpy.arange(num, 40, num)
        will be assigned to ws_res

        If nothing is passed, it will default to
        numpy.arange(2, 42, 2)

    wa_res : array_like or positive int/float, optional
        Wind angles that will correspond to the rows
        of the table. Should be between 0° and 360°

        Can either be sequence of length rdim or an
        int/float value

        If a number num is passed, numpy.arange(num, 360, num)
        will be assigned to wa_res

        If nothing is passed, it will default to
        numpy.arange(0, 360, 5)

    bsps : array_like, optional
        Boatspeeds that will correspond to the entries of
        the table

        Should be broadcastable to the shape (rdim, cdim)

        If nothing is passed it will default to
        numpy.zeros((rdim, cdim))

    Raises a PolarDiagramException
        - if bsps can't be broadcasted
        to a fitting shape
        - if bsps is not of dimension 2
        - if bsps is an empty array


    Methods
    -------
    wind_speeds
        Returns a read only version of self._res_wind_speed
    wind_angles
        Returns a read only version of self._res_wind_angle
    boat_speeds
        Returns a read only version of self._boat_speeds
    to_csv(csv_path, fmt='hro')
        Creates a .csv-file with delimiter ',' and the
        following format:
            PolarDiagramTable
            Wind speed resolution:
            self.wind_speeds
            Wind angle resolution:
            self.wind_angles
            Boat speeds:
            self.boat_speeds
    symmetrize()
        Constructs a symmetric version of the polar diagram, by mirroring it
        at the 0° - 180° axis and returning a new instance
    change_entries(data, ws=None, wa=None, tw=True)
        Changes specified entries in the table
    plot_polar(
        ws=None,
        ax=None,
        colors=('green', 'red'),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    )
        Creates a polar plot of one or more slices of the polar diagram
    plot_flat(
        ws=None,
        ax=None,
        colors=('green', 'red'),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    )
        Creates a cartesian plot of one or more slices of the polar diagram
    plot_3d(ax=None, colors=('blue', 'blue'))
        Creates a 3d plot of the polar diagram
    plot_color_gradient(
        ax=None,
        colors=('green', 'red'),
        marker=None,
        show_legend=False,
        **legend_kw,
    )
        Creates a 'wind speed vs. wind angle' color gradient plot
        of the polar diagram with respect to the respective boat speeds
    plot_convex_hull(
        ws=None,
        ax=None,
        colors=('green', 'red'),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    )
        Computes the (seperate) convex hull of one or more
        slices of the polar diagram and creates a polar plot of them
    """

    def __init__(self, ws_res=None, wa_res=None, bsps=None):
        logger.info(
            f"Class 'PolarDiagramTable(ws_res={ws_res}, wa_res={wa_res}, "
            f"bsps={bsps})' called"
        )

        try:
            ws_res = set_resolution(ws_res, "speed")
            wa_res = set_resolution(wa_res, "angle")
        except WindException as we:
            raise PolarDiagramException("") from we

        rows, cols = len(wa_res), len(ws_res)
        if bsps is None:
            bsps = np.zeros((rows, cols))
        else:
            # NaN's and infinite values can't be handled
            try:
                bsps = np.asarray_chkfinite(bsps, float)
            except ValueError as ve:
                raise PolarDiagramException(
                    "`bsps` contains infinite or NaN values"
                ) from ve

            # Non array_like `bsps` are not allowed
            if bsps.dtype is object:
                raise PolarDiagramException("`bsps` is not array_like")

            if bsps.shape != (rows, cols) or bsps.ndim != 2:
                raise PolarDiagramException("`bsps` has incorrect shape")

        # Sort wind angles and the corresponding order of rows in bsps
        wa_res, bsps = zip(*sorted(zip(wa_res, bsps), key=lambda x: x[0]))
        self._res_wind_angle = np.asarray(wa_res)
        bsps = np.asarray(bsps, float)

        # Sort wind speeds and the corresponding order of columns in bsps
        ws_res, bsps = zip(*sorted(zip(ws_res, bsps.T), key=lambda x: x[0]))
        self._res_wind_speed = np.asarray(ws_res)
        self._boat_speeds = np.asarray(bsps, float).T

    def __str__(self):
        table = ["  TWA \\ TWS"]
        bsps = self.boat_speeds
        if len(self.wind_speeds) <= 15:
            wind = self.wind_speeds
            table.extend([f"    {float(ws):.1f}" for ws in wind])
            table.append("\n-----------")
            for ws in wind:
                le = len(f"{float(ws):.1f}")
                table.append("  ".ljust(le + 4, "-"))
            table.append("\n")
            for i, wa in enumerate(self.wind_angles):
                angle = f"{float(wa):.1f}"
                table.append(angle.ljust(11))
                for j, ws in enumerate(wind):
                    entry = f"{bsps[i][j]:.2f}"
                    le = len(str(ws))
                    table.append(entry.rjust(4 + le))
                table.append("\n")
            return "".join(table)

        wind = []
        wind.extend(self.wind_speeds[:5])
        wind.extend(self.wind_speeds[-5:])
        for i, ws in enumerate(wind):
            if i == 5:
                table.append("  ...")
            table.append(f"    {float(ws):.1f}")
        table.append("\n-----------")
        for i, ws in enumerate(wind):
            if i == 5:
                table.append("  ---")
            le = len(f"{float(ws):.1f}")
            table.append("  ".ljust(le + 4, "-"))
        table.append("\n")
        for i, wa in enumerate(self.wind_angles):
            angle = f"{float(wa):.1f}"
            table.append(angle.rjust(11))
            for j, ws in enumerate(wind):
                if j == 5:
                    table.append("  ...")
                entry = f"{bsps[i][j]:.2f}"
                le = len(str(ws))
                table.append(entry.rjust(4 + le))
            table.append("\n")
        return "".join(table)

    def __repr__(self):
        return (
            f"PolarDiagramTable(ws_res={self.wind_speeds}, "
            f"wa_res={self.wind_angles}, bsps={self.boat_speeds})"
        )

    def __getitem__(self, key):
        ws, wa = key
        col = _get_indices(ws, self.wind_speeds)
        row = _get_indices(wa, self.wind_angles)
        return self.boat_speeds[row, col]

    @property
    def wind_angles(self):
        """Returns a read only version of self._res_wind_angle"""
        return self._res_wind_angle.copy()

    @property
    def wind_speeds(self):
        """Returns a read only version of self._res_wind_speed"""
        return self._res_wind_speed.copy()

    @property
    def boat_speeds(self):
        """Returns a read only version of self._boat_speeds"""
        return self._boat_speeds.copy()

    def to_csv(self, csv_path, fmt="hro"):
        """Creates a .csv file with delimiter ',' and the
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
        csv_path : path-like
            Path to a .csv file or where a new .csv file will be created

        fmt : string


        Raises a FileWritingException
            - inputs are not of the specified types
            - if an error occurs whilst writing
            - unknown format was specified
        """
        logger.info(f"Method '.to_csv({csv_path}, fmt={fmt})' called")

        if fmt not in {"hro", "opencpn", "orc"}:
            raise PolarDiagramException("`fmt` not implemented")

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
                csv_writer.writerow(["TWS:"])
                csv_writer.writerow(self.wind_speeds)
                csv_writer.writerow(["TWA:"])
                csv_writer.writerow(self.wind_angles)
                csv_writer.writerow(["Boat speeds:"])
                csv_writer.writerows(self.boat_speeds)
        except OSError as oe:
            raise FileWritingException(
                "While writing to `csv_path` an error occured"
            ) from oe

    def symmetrize(self):
        """Constructs a symmetric version of the
        polar diagram, by mirroring it at the 0° - 180° axis
        and returning a new instance

        Warning
        -------
        Should only be used if all the wind angles of the initial
        polar diagram are on one side of the 0° - 180° axis,
        otherwise this can lead to duplicate data, which can
        overwrite or live alongside old data
        """
        below_180 = [wa for wa in self.wind_angles if wa <= 180]
        above_180 = [wa for wa in self.wind_angles if wa > 180]
        if below_180 and above_180:
            print(
                "Warning: There are wind angles on both sides of the "
                "0° - 180° axis. This might result in duplicate data, "
                "which can overwrite or live alongside old data"
            )

        wa_res = np.concatenate(
            [self.wind_angles, 360 - np.flip(self.wind_angles)]
        )
        bsps = np.row_stack(
            (self.boat_speeds, np.flip(self.boat_speeds, axis=0))
        )

        # deleting multiple 180° and 0° occurences in the table
        if 180 in self.wind_angles:
            wa_res = list(wa_res)
            mid = wa_res.index(180) or wa_res.index(180.0)
            del wa_res[mid]
            bsps = np.row_stack((bsps[:mid, :], bsps[mid + 1 :, :]))
        if 0 in self.wind_angles:
            bsps = bsps[:-1, :]
            wa_res = wa_res[:-1]

        return PolarDiagramTable(
            ws_res=self.wind_speeds, wa_res=wa_res, bsps=bsps
        )

    def change_entries(self, new_bsps, ws=None, wa=None):
        """Changes specified entries in the table

        Parameters
        ----------
        new_bsps: array_like of matching shape
            Sequence containing the new boat speeds to be inserted
            in the specified entries

        ws: Iterable or int or float, optional
            Element(s) of self.wind_speeds, specifying the columns,
            where new boat speeds will be inserted

            If nothing is passed it will default to self.wind_speeds

        wa: Iterable or int or float, optional
            Element(s) of self.wind_angles, specifiying the rows,
            where new boat speeds will be inserted

            If nothing is passed it will default to self.wind_angles


        Raises a PolarDiagramException
            - If ws is not contained in self.wind_speeds
            - If wa is not contained in self.wind_angles
            - If new_bsps can't be broadcasted to a fitting shape
        """
        logger.info(
            f"Method 'PolarDiagramTable.change_entries("
            f"new_bsps={new_bsps}, ws={ws}, wa={wa}) called"
        )

        try:
            new_bsps = np.asarray_chkfinite(new_bsps)
        except ValueError as ve:
            raise PolarDiagramException(
                "`new_bsps` contains infinite or NaN values"
            ) from ve
        if new_bsps.dtype is object:
            raise PolarDiagramException("`new_bsps` is not array_like")

        ws_ind = _get_indices(ws, self.wind_speeds)
        wa_ind = _get_indices(wa, self.wind_angles)

        if new_bsps.shape != (len(wa_ind), len(ws_ind)):
            raise PolarDiagramException("`new_bsps` has incorrect shape")

        mask = np.zeros(self.boat_speeds.shape, dtype=bool)
        for i in wa_ind:
            for j in ws_ind:
                mask[i, j] = True
        self._boat_speeds[mask] = new_bsps.flat

    def _get_radians(self):
        return np.deg2rad(self.wind_angles)

    def get_slices(self, ws=None):
        """

        """
        if ws is None:
            ws = self.wind_speeds
        if isinstance(ws, (int, float)):
            ws = [ws]

        ws = sorted(list(ws))
        if not ws:
            raise PolarDiagramException("No slices were given")

        ind = _get_indices(ws, self.wind_speeds)
        wa = self._get_radians()
        return ws, wa, self.boat_speeds[:, ind]

    def plot_polar(
        self,
        ws=None,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Creates a polar plot of one or more slices of the polar diagram

        Parameters
        ----------
        ws : Iterable, int or float, optional
            Slices of the polar diagram table, given as either
                - an Iterable containing only elements of
                self.wind_speeds

                - a single element of self.wind_speeds

            The slices are then equal to the corresponding
            columns of the table together with self.wind_angles

            If nothing it passed, it will default to self.wind_speeds

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        colors : tuple, optional
            Specifies the colors to be used for the different
            slices. There are four options for the tuple
                - If as many or more colors
                as slices are passed, each
                slice will be plotted in the
                specified color

                - If exactly 2 colors are passed,
                the slices will be plotted with a
                color gradient consiting of the
                two colors

                - If more than 2 colors but less
                than slices are passed, the first
                n_color slices will be plotted in
                the specified colors, and the rest
                will be plotted in the default color 'blue'

                Alternatively one can specify certain slices
                to be plotted in a certain color by passing
                a tuple of (ws, color) pairs

            Defaults to ('green', 'red')

        show_legend : bool, optional
            Specifies wether or not a legend will be shown
            next to the plot

            The type of legend depends
            on the color options
                - If the slices are plotted with a
                color gradient, a matplotlib.colorbar.Colorbar
                object will be created and assigned to ax.

                - Otherwise a matplotlib.legend.Legend
                will be created and assigned to ax.

            Default to False

        legend_kw : dict, optional
            Keyword arguments to be passed to either the
            matplotlib.colorbar.Colorbar or matplotlib.legend.Legend
            classes to change position and appearence of the legend

            Will only be used if show_legend is True

            If nothing is passed it will default to {}

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot


        Raises a PolarDiagramException if at least one element
        of ws is not in self.wind_speeds
        """
        logger.info(
            f"Method 'polar_plot(ws={ws}, ax={ax}, colors={colors}, "
            f"show_legend={show_legend}, legend_kw={legend_kw}, "
            f"plot_kw={plot_kw})' called"
        )

        ws, wa, bsp = self.get_slices(ws)
        bsp = list(bsp.T)
        wa = [wa] * len(bsp)
        plot_polar(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw)

    def plot_flat(
        self,
        ws=None,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Creates a cartesian plot of one or more slices of the polar diagram

        Parameters
        ----------
        ws : Iterable, int or float, optional
            Slices of the polar diagram table, given as either
                - an Iterable containing only elements of
                self.wind_speeds

                - a single element of self.wind_speeds

            The slices are then equal to the corresponding
            columns of the table together with self.wind_angles

            If nothing it passed, it will default to self.wind_speeds

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        colors : tuple, optional
            Specifies the colors to be used for the different
            slices. There are four options
                - If as many or more colors
                as slices are passed, each
                slice will be plotted in the
                specified color

                - If exactly 2 colors are passed,
                the slices will be plotted with a
                color gradient consiting of the
                two colors

                - If more than 2 colors but less
                than slices are passed, the first
                n_color slices will be plotted in
                the specified colors, and the rest
                will be plotted in the default color 'blue'

                Alternatively one can specify certain slices
                to be plotted in a certain color by passing
                a tuple of (ws, color) pairs

            Defaults to ('green', 'red')

        show_legend : bool, optional
            Specifies wether or not a legend will be shown
            next to the plot

            The type of legend depends
            on the color options
                - If the slices are plotted with a
                color gradient, a matplotlib.colorbar.Colorbar
                object will be created and assigned to ax.

                - Otherwise a matplotlib.legend.Legend
                will be created and assigned to ax.

            Default to False

        legend_kw : dict, optional
            Keyword arguments to be passed to either the
            matplotlib.colorbar.Colorbar or matplotlib.legend.Legend
            classes to change position and appearence of the legend

            Will only be used if show_legend is True

            If nothing is passed it will default to {}

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot


        Raises a PolarDiagramException if at least one element
        of ws is not in self.wind_speeds
        """
        logger.info(
            f"Method 'flat_plot(ws={ws}, ax={ax}, colors={colors}, "
            f"show_legend={show_legend}, legend_kw={legend_kw}, "
            f"plot_kw={plot_kw})' called"
        )

        ws, wa, bsp = self.get_slices(ws)
        bsp = list(bsp.T)
        wa = [np.rad2deg(wa)] * len(bsp)
        plot_flat(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw)

    def plot_3d(self, ax=None, colors=("blue", "blue")):
        """Creates a 3d plot of the polar diagram

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        colors : tuple of length 2, optional
            Colors which specify the color gradient with
            which the polar diagram will be plotted.

            If no color gradient is desired, set both elements
            to the same color

            Defaults to ('blue', 'blue')
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
        show_legend=False,
        **legend_kw,
    ):
        """Creates a 'wind speed vs. wind angle' color gradient plot
        of the polar diagram with respect to the respective boat speeds

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        colors : tuple of length 2, optional
            Colors which specify the color gradient with
            which the polar diagram will be plotted.

            Defaults to ('green', 'red')

        marker : matplotlib.markers.Markerstyle or equivalent, optional
            Markerstyle for the created scatter plot

            If nothing is passed, it will default to 'o'

        show_legend : bool, optional
            Specifies wether or not a legend will be shown next
            to the plot

            Legend will be a matplotlib.colorbar.Colorbar object.

            Defaults to False

        legend_kw : Keyword arguments
            Keyword arguments to be passed to the
            matplotlib.colorbar.Colorbar class to change position
            and appearence of the legend.

            Will only be used if show_legend is True

            If nothing is passed, it will default to {}
        """
        logger.info(
            f"Method 'plot_color_gradient( ax={ax}, colors={colors}, "
            f"marker={marker}, show_legend={show_legend},"
            f"legend_kw={legend_kw})' called"
        )

        ws, wa = np.meshgrid(self.wind_speeds, self.wind_angles)
        ws = ws.ravel()
        wa = wa.ravel()
        bsp = self.boat_speeds.ravel()
        plot_color_gradient(
            ws, wa, bsp, ax, colors, marker, show_legend, **legend_kw
        )

    def plot_convex_hull(
        self,
        ws=None,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Computes the (seperate) convex hull of one or more
        slices of the polar diagram and creates a polar plot of them

        Parameters
        ----------
        ws : Iterable, int or float, optional
            Slices of the polar diagram table, given as either
                - an Iterable containing only elements of
                self.wind_speeds

                - a single element of self.wind_speeds

            The slices are then equal to the corresponding
            columns of the table together with self.wind_angles

            If nothing it passed, it will default to self.wind_speeds

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        colors : tuple, optional
            Specifies the colors to be used for the different
            slices. There are four options
                - If as many or more colors
                as slices are passed, each
                slice will be plotted in the
                specified color

                - If exactly 2 colors are passed,
                the slices will be plotted with a
                color gradient consiting of the
                two colors

                - If more than 2 colors but less
                than slices are passed, the first
                n_color slices will be plotted in
                the specified colors, and the rest
                will be plotted in the default color 'blue'

                Alternatively one can specify certain slices
                to be plotted in a certain color by passing
                a tuple of (ws, color) pairs

            Defaults to ('green', 'red')

        show_legend : bool, optional
            Specifies wether or not a legend will be shown
            next to the plot

            The type of legend depends
            on the color options
                - If the slices are plotted with a
                color gradient, a matplotlib.colorbar.Colorbar
                object will be created and assigned to ax.

                - Otherwise a matplotlib.legend.Legend
                will be created and assigned to ax.

            Default to False

        legend_kw : dict, optional
            Keyword arguments to be passed to either the
            matplotlib.colorbar.Colorbar or matplotlib.legend.Legend
            classes to change position and appearence of the legend

            Will only be used if show_legend is True

            If nothing is passed it will default to {}

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot

        Raises a PolarDiagramException if at least one element
        of ws is not in self.wind_speeds
        """
        logger.info(
            f"Method 'plot_convex_hull(ws={ws}, ax={ax}, colors={colors},"
            f" show_legend={show_legend}, legend_kw={legend_kw}, "
            f"**plot_kw={plot_kw})' called"
        )

        ws, wa, bsp = self.get_slices(ws)
        bsp = list(bsp.T)
        wa = [wa] * len(bsp)
        plot_convex_hull(
            ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw
        )


def interpolate():
    pass


# TODO: Standardize wind angles, such that they are in [0, 360),
#       because 360° should be equal to 0°

# TODO: Docstrings
class PolarDiagramMultiSails(PolarDiagram):
    """A class to represent, visualize and work with
    a polar diagram made up of multiple sets of sails,
    represented by a PolarDiagramTable

    Parameters
    ----------
    pds : list

    sails : list


    Raises a PolarDiagramException if


    Methods
    -------
    wind_speeds
        Returns a read only version of self._res_wind_speed
    wind_angles
        Returns a read only version of self._res_wind_angle
    boat_speeds
        Returns a list of pt.boat_speeds for every PolarDiagramTable
        pt in self._sails
    to_csv(csv_path)

    symmetrize()
        Constructs a symmetric version of the polar diagram, by
        mirroring each PolarDiagramTable  at the 0° - 180° axis
        and returning a new instance
    get_slices(ws)
    plot_polar(
        ws=None,
        ax=None,
        colors=('green', 'red'),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    )
        Creates a polar plot of one or more slices of the polar diagram
    plot_flat(
        ws=None,
        ax=None,
        colors=('green', 'red'),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    )
        Creates a cartesian plot of one or more slices of the polar diagram
    plot_3d(ax=None, colors=('blue', 'blue'))
        Creates a 3d plot of the polar diagram
    plot_color_gradient(
        ax=None,
        colors=('green', 'red'),
        marker=None,
        show_legend=False,
        **legend_kw,
    )
        Creates a 'wind speed vs. wind angle' color gradient plot
        of the polar diagram with respect to the respective boat speeds
    plot_convex_hull(
        ws=None,
        ax=None,
        colors=('green', 'red'),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    )
        Computes the (seperate) convex hull of one or more
        slices of the polar diagram and creates a polar plot of them
    """

    def __init__(self, pds: List[PolarDiagramTable], sails: List[str] = None):
        ws = pds[0].wind_speeds
        for pd in pds:
            if not np.array_equal(ws, pd.wind_speeds):
                raise PolarDiagramException(
                    "wind speed resolution of `pds` does not coincide"
                )

        if sails is None:
            sails = [f"Sail {i}" for i in range(len(pds))]

        self._sails = sails
        self._tables = pds

    @property
    def wind_speeds(self):
        """"""
        return

    @property
    def wind_angles(self):
        """"""
        return

    @property
    def boat_speeds(self):
        """"""
        return

    def to_csv(self, csv_path):
        pass

    def symmetrize(self):
        """Constructs a symmetric version of the polar diagram, by
        mirroring each PolarDiagramTable at the 0° - 180° axis and
        returning a new instance. See also the symmetrize()-method
        of the PolarDiagramTable class

        Warning
        -------
        Should only be used if all the wind angles of the PolarDiagramTables
        are each on one side of the 0° - 180° axis, otherwise this can lead
        to duplicate data, which can overwrite or live alongside old data
        """
        pds = [pd.symmetrize() for pd in self._tables]
        return PolarDiagramMultiSails(pds, self._sails)

    def get_slices(self, ws):
        """"""
        wa = []
        temp = []
        for pd in self._tables:
            ws, w, b = pd.get_slices(ws)
            wa.append(w)
            temp.append(b)

        flatten = itertools.chain.from_iterable
        members = [[self._sails[i]] * len(w) for i, w in enumerate(wa)]
        members = list(flatten(members))

        wa = [np.asarray(wa).ravel()] * len(ws)
        bsp = []
        for i in range(len(ws)):
            b = np.asarray([b_[:, i] for b_ in temp]).ravel()
            bsp.append(b)

        return ws, wa, bsp, members

    def plot_polar(
        self,
        ws,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        if ax is None:
            ax = plt.axes(projection="polar")

        for i, pd in enumerate(self._tables):
            if i == 0 and show_legend:
                pd.plot_polar(
                    ws, ax, colors, show_legend, legend_kw, **plot_kw
                )
                continue

            pd.plot_polar(ws, ax, colors, False, None, **plot_kw)

    def plot_flat(
        self,
        ws,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        for i, pd in enumerate(self._tables):
            if i == 0 and show_legend:
                pd.plot_flat(ws, ax, colors, show_legend, legend_kw, **plot_kw)
                continue

            pd.plot_flat(ws, ax, colors, False, None, **plot_kw)

    def plot_3d(self, ax=None, colors=None):
        if ax is None:
            ax = plt.axes(projection="3d")

        for pd in self._tables:
            pd.plot_3d(ax, colors)

    def plot_color_gradient(
        self,
        ax=None,
        colors=("green", "red"),
        marker=None,
        show_legend=False,
        **legend_kw,
    ):
        pass

    def plot_convex_hull(
        self,
        ws,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        ws, wa, bsp, members = self.get_slices(ws)
        plot_convex_hull_multisails(
            ws, wa, bsp, members, ax, colors, show_legend, legend_kw, **plot_kw
        )


class PolarDiagramCurve(PolarDiagram):
    """A class to represent, visualize and work with a polar diagram
    given by a fitted curve/surface.

    Parameters
    ----------
    f : function
        Curve/surface that describes the polar diagram, given as
        a function, with the signature f(x, *params) -> y,
        where x is a numpy.ndarray of shape (n, 2)
        which corresponds to pairs of wind speed and wind angle
        and y is a numpy.ndarray of shape (n, ) or (n, 1)
        which corresponds to the boat speed at the resp.
        wind speed and wind angle.

    params : tuple or Sequence
        Optimal parameters for f

    radians : bool, optional
        Specifies if f takes the wind angles to be in radians or degrees

        Defaults to False


    Methods
    -------
    curve
        Returns a read only version of self._f
    parameters
        Returns a read only version of self._params
    radians
        Returns a read only version of self._rad
    to_csv(csv_path)
        Creates a .csv-file with delimiter ':' and the
        following format:
            PolarDiagramCurve
            Function: self.curve
            Radians: self.rad
            Parameters: self.parameters
    symmetrize()
    get_slices(ws)
    plot_polar(
        ws=(0, 20, 5),
        ax=None,
        colors=('green', 'red'),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    )
        Creates a polar plot of one or more slices of the polar diagram
    plot_flat(
        ws=(0, 20, 5),
        ax=None,
        colors=('green', 'red'),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    )
        Creates a cartesian plot of one or more slices of the polar diagram
    plot_3d(ws_range=(0, 20, 100), ax=None, colors=('blue', 'blue'))
        Creates a 3d plot of a part of the polar diagram
    plot_color_gradient(
        ws=(0, 20, 100),
        ax=None,
        colors=('green', 'red'),
        marker=None,
        show_legend=False,
        **legend_kw,
    )
        Creates a 'wind speed  vs. wind angle' color gradient
        plot of a part of the polar diagram with respect to the
        respective boat speeds
    plot_convex_hull(
        ws=(0, 20, 5),
        ax=None,
        colors=('green', 'red'),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    )
        Computes the (seperate) convex hull of one or more
        slices of the polar diagram and creates a polar plot of them
    """

    def __init__(self, f, params, radians=False):
        if not callable(f):
            raise PolarDiagramException("`f` is not callable")

        logger.info(
            f"Class 'PolarDiagramCurve(f={f.__name__}, {params}, "
            f"radians={radians})' called"
        )

        self._f = f
        self._params = params
        self._rad = radians

    def __repr__(self):
        return (
            f"PolarDiagramCurve(f={self._f.__name__},"
            f"{self._params}, radians={self._rad})"
        )

    def __call__(self, ws, wa):
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
        """Creates a .csv file with delimiter ':' and the
        following format:
            PolarDiagramCurve
            Function: self.curve.__name__
            Radians: self.rad
            Parameters: self.parameters

        Parameters
        ----------
        csv_path : path-like
            Path to a .csv file or where a new .csv file will be created

        Raises a FileWritingException if an error occurs whilst writing
        """
        logger.info(f"Method '.to_csv({csv_path})' called")

        try:
            with open(csv_path, "w", newline="") as file:
                csv_writer = csv.writer(file, delimiter=":")
                csv_writer.writerow(["PolarDiagramCurve"])
                csv_writer.writerow(["Function"] + [self.curve.__name__])
                csv_writer.writerow(["Radians"] + [str(self.radians)])
                csv_writer.writerow(["Parameters"] + list(self.parameters))
        except OSError as oe:
            raise FileWritingException(
                "While writing to `csv_path` an error occured"
            ) from oe

    def symmetrize(self):
        """

        """

        def sym_func(w_arr, *params):
            sym_w_arr = w_arr.copy()
            sym_w_arr[:, 1] = 360 - sym_w_arr[:, 1]
            return 0.5 * (
                self.curve(w_arr, *params) + self.curve(sym_w_arr, *params)
            )

        return PolarDiagramCurve(
            sym_func, self.parameters, radians=self.radians
        )

    def _get_wind_angles(self):
        wa = np.linspace(0, 360, 1000)
        if self.radians:
            wa = np.deg2rad(wa)
        return wa

    def get_slices(self, ws):
        if isinstance(ws, (int, float)):
            ws = [ws]
        elif isinstance(ws, tuple):
            ws = list(np.linspace(ws[0], ws[1], ws[2]))

        wa = self._get_wind_angles()
        bsp = [self(np.array([w] * 1000), wa) for w in ws]
        if not self.radians:
            wa = np.deg2rad(wa)

        return ws, wa, bsp

    def plot_polar(
        self,
        ws=(0, 20, 5),
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Creates a polar plot of one or more slices of the polar diagram

        Parameters
        ----------
        ws : tuple of length 3, list, int or float, optional
            Slices of the polar diagram
            given as either
                - a tuple of three values, which will be
                interpreted as a start and end point of an
                interval aswell as a number of slices,
                which will be evenly spaced in the given
                interval

                - a list of specific wind speeds

                - a single wind speed

            Slices will then equal self(w, wa) where w
            takes the given values in ws and wa goes through
            a fixed number of angles between 0° and 360°

            Defaults to (0, 20, 5)

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        colors : tuple, optional
            Specifies the colors to be used for the different
            slices. There are four options
                - If as many or more colors as slices are passed,
                each slice will be plotted in the specified color

                - If exactly 2 colors are passed, the slices will
                be plotted with a color gradient consiting of the
                two colors

                - If more than 2 colors but less than slices are passed,
                the first n_color slices will be plotted in the specified
                colors, and the rest will be plotted in the default color
                'blue'

                Alternatively one can specify certain slices
                to be plotted in a certain color by passing
                a tuple of (ws, color) pairs

            Defaults to ('green', 'red')

        show_legend : bool, optional
            Specifies wether or not a legend will be shown
            next to the plot

            The type of legend depends
            on the color options
                - If the slices are plotted with a
                color gradient, a matplotlib.colorbar.Colorbar
                object will be created and assigned to ax.

                - Otherwise a matplotlib.legend.Legend
                will be created and assigned to ax.

            Default to False

        legend_kw : dict, optional
            Keyword arguments to be passed to either the
            matplotlib.colorbar.Colorbar or matplotlib.legend.Legend
            classes to change position and appearence of the legend

            Will only be used if show_legend is True

            If nothing is passed it will default to {}

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot
        """
        logger.info(
            f"Method 'polar_plot( ws={ws}, ax={ax}, colors={colors},"
            f"show_legend={show_legend}, legend_kw={legend_kw},"
            f"**plot_kw={plot_kw})' called"
        )

        ws, wa, bsp = self.get_slices(ws)
        wa = [wa] * len(ws)

        plot_polar(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw)

    def plot_flat(
        self,
        ws=(0, 20, 5),
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Creates a cartesian plot of multiple slices of the polar diagram

        Parameters
        ----------
        ws : tuple of length 3, list, int or float, optional
            Slices of the polar diagram
            given as either
                - a tuple of three values, which will be
                interpreted as a start and end point of an
                interval aswell as a number of slices,
                which will be evenly spaced in the given
                interval

                - a list of specific wind speeds

                - a single wind speed

            Slices will then equal self(w, wa) where w
            takes the given values in ws and wa goes through
            a fixed number of angles between 0° and 360°

            Defaults to (0, 20, 5)

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        colors : tuple, optional
            Specifies the colors to be used for the different
            slices. There are four options
                - If as many or more colors as slices are passed,
                each slice will be plotted in the specified color

                - If exactly 2 colors are passed, the slices will
                be plotted with a color gradient consiting of the
                two colors

                - If more than 2 colors but less than slices are passed,
                the first n_color slices will be plotted in the specified
                colors, and the rest will be plotted in the default color
                'blue'

                Alternatively one can specify certain slices
                to be plotted in a certain color by passing
                a tuple of (ws, color) pairs

            Defaults to ('green', 'red')

        show_legend : bool, optional
            Specifies wether or not a legend will be shown
            next to the plot

            The type of legend depends
            on the color options
                - If the slices are plotted with a
                color gradient, a matplotlib.colorbar.Colorbar
                object will be created and assigned to ax.

                - Otherwise a matplotlib.legend.Legend
                will be created and assigned to ax.

            Default to False

        legend_kw : dict, optional
            Keyword arguments to be passed to either the
            matplotlib.colorbar.Colorbar or matplotlib.legend.Legend
            classes to change position and appearence of the legend

            Will only be used if show_legend is True

            If nothing is passed it will default to {}

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot
        """
        logger.info(
            f"Method 'polar_plot("
            f"ws={ws}, "
            f"ax={ax}, colors={colors},"
            f"show_legend={show_legend}, "
            f"legend_kw={legend_kw},"
            f"**plot_kw={plot_kw})' called"
        )

        ws, wa, bsp = self.get_slices(ws)
        wa = [np.rad2deg(wa)] * len(ws)

        plot_flat(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw)

    def plot_3d(self, ws=(0, 20, 100), ax=None, colors=("blue", "blue")):
        """Creates a 3d plot of a part of the polar diagram

        Parameters
        ----------
        ws : tuple of length 3, optional
            A region of the polar diagram given as a tuple
            of three values, which will be interpreted as a
            start and end point of an interval aswell as a
            number of slices, which will be evenly spaced
            in the given interval

            Slices will then equal self(w, wa) where w
            takes the given values in ws and wa goes through
            a fixed number of angles between 0° and 360°

            Defaults to (0, 20, 100)

        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        colors : tuple of length 2, optional
            Colors which specify the color gradient with
            which the polar diagram will be plotted.

            If no color gradient is  desired, set both
            elements to the same color

            Defaults to ('blue', 'blue')
        """
        logging.info(
            f"Method 'plot_3d(ws={ws}, ax={ax}, " f"colors={colors})' called"
        )

        ws, wa, bsp = self.get_slices(ws)
        bsp = np.array(bsp)
        ws, wa = np.meshgrid(ws, wa)
        bsp, wa = bsp * np.cos(wa), bsp * np.sin(wa)

        plot_surface(ws, wa, bsp, ax, colors)

    def plot_color_gradient(
        self,
        ws=(0, 20, 100),
        ax=None,
        colors=("green", "red"),
        marker=None,
        show_legend=False,
        **legend_kw,
    ):
        """Creates a 'wind speed vs. wind angle' color gradient plot
        of a part of the polar diagram with respect to the respective
        boat speeds

        Parameters
        ----------
        ws :  tuple of length 3, optional
            A region of the polar diagram given as a tuple
            of three values, which will be interpreted as a
            start and end point of an interval aswell as a
            number of slices, which will be evenly spaced
            in the given interval

            Slices will then equal self(w, wa) where w
            takes the given values in ws and wa goes through
            a fixed number of angles between 0° and 360°

            Defaults to (0, 20, 100)

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        colors : tuple of length 2, optional
            Colors which specify the color gradient with
            which the polar diagram will be plotted.

            Defaults to ('green', 'red')

        marker : matplotlib.markers.Markerstyle or equivalent, optional
            Markerstyle for the created scatter plot

            If nothing is passed, it will default to 'o'

        show_legend : bool, optional
            Specifies wether or not a legend will be shown next
            to the plot

            Legend will be a matplotlib.colorbar.Colorbar object.

            Defaults to False

        legend_kw : Keyword arguments
            Keyword arguments to be passed to the
            matplotlib.colorbar.Colorbar class to change position
            and appearence of the legend.

            Will only be used if show_legend is True

            If nothing is passed, it will default to {}
        """
        logger.info(
            f"Method 'plot_color_gradient(ws={ws}, ax={ax}, "
            f"colors={colors}, marker={marker}, "
            f"show_legend={show_legend}, **legend_kw={legend_kw})' "
            f"called"
        )

        ws, wa = np.meshgrid(
            np.linspace(ws[0], ws[1], ws[2]), np.linspace(0, 360, 1000)
        )
        ws = ws.ravel()
        wa = wa.ravel()

        if self.radians:
            bsp = self(ws, np.deg2rad(wa)).ravel()
        else:
            bsp = self(ws, wa).ravel()

        plot_color_gradient(
            ws, wa, bsp, ax, colors, marker, show_legend, **legend_kw
        )

    def plot_convex_hull(
        self,
        ws=(0, 20, 5),
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Computes the (seperate) convex hull of one or more
        slices of the polar diagram and creates a polar plot of them

        Parameters
        ----------
        ws : tuple of length 3, list, int or float, optional
            Slices of the polar diagram
            given as either
                - a tuple of three values, which will be
                interpreted as a start and end point of an
                interval aswell as a number of slices,
                which will be evenly spaced in the given
                interval

                - a list of specific wind speeds

                - a single wind speed

            Slices will then equal self(w, wa) where w
            takes the given values in ws and wa goes through
            a fixed number of angles between 0° and 360°

            Defaults to (0, 20, 5)

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        colors : tuple, optional
            Specifies the colors to be used for the different
            slices. There are four options
                - If as many or more colors as slices are passed,
                each slice will be plotted in the specified color

                - If exactly 2 colors are passed, the slices will
                be plotted with a color gradient consiting of the
                two colors

                - If more than 2 colors but less than slices are passed,
                the first n_color slices will be plotted in the specified
                colors, and the rest will be plotted in the default color
                'blue'

                Alternatively one can specify certain slices
                to be plotted in a certain color by passing
                a tuple of (ws, color) pairs

            Defaults to ('green', 'red')

        show_legend : bool, optional
            Specifies wether or not a legend will be shown
            next to the plot

            The type of legend depends
            on the color options
                - If the slices are plotted with a
                color gradient, a matplotlib.colorbar.Colorbar
                object will be created and assigned to ax.

                - Otherwise a matplotlib.legend.Legend
                will be created and assigned to ax.

            Default to False

        legend_kw : dict, optional
            Keyword arguments to be passed to either the
            matplotlib.colorbar.Colorbar or matplotlib.legend.Legend
            classes to change position and appearence of the legend

            Will only be used if show_legend is True

            If nothing is passed it will default to {}

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot
        """
        logger.info(
            f"Method 'plot_convex_hull(ws={ws}, ax={ax}, colors={colors},"
            f" show_legend={show_legend}, legend_kw={legend_kw}, "
            f"**plot_kw={plot_kw})' called"
        )

        ws, wa, bsp = self.get_slices(ws)

        plot_convex_hull(
            ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw
        )


# TODO: Standardize wind angles, such that they are in [0, 360),
#       because 360° should be equal to 0°
class PolarDiagramPointcloud(PolarDiagram):
    """A class to represent, visualize and work with a polar diagram
    given by a point cloud

    Parameters
    ----------
    pts : array_like of shape (n, 3), optional
        Initial points of the point cloud, given as a sequence of
        points consisting of wind speed, wind angle and boat speed

        If nothing is passed, the point cloud will be initialized
        as an empty point cloud

    tw : bool, optional
        Specifies if the given wind data should be viewed as true wind

        If False, wind data will be converted to true wind

        Defaults to True

    Raises a PolarDiagramException
        - if
        - if


    Methods
    -------
    wind_speeds
        Returns all unique wind speeds in the point cloud
    wind_angles
        Returns all unique wind angles in the point cloud
    points
        Returns a read only version  of self._pts
    to_csv(csv_path)
        Creates a .csv-file with delimiter ',' and the
        following format:
            PolarDiagramPointcloud
            True wind speed: , True wind angle: , Boat speed:
            self.get_points
    symmetrize()
    add_points(new_pts)
        Adds additional points to the point cloud
    get_slices(ws)
    polar_plot(
        ws=(0, np.inf),
        ax=None,
        colors=('green', 'red'),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    )
        Creates a polar plot of one or more slices of the polar diagram
    flat_plot(
        ws=(0, np.inf),
        ax=None,
        colors=('green', 'red'),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    )
        Creates a cartesian plot of one or more slices of the polar diagram
    plot_3d(ax=None, **plot_kw)
        Creates a 3d plot of the polar diagram
    plot_color_gradient(
        ax=None,
        colors=('green', 'red'),
        marker=None,
        show_legend=False,
        **legend_kw
    )
        Creates a 'wind speed vs. wind angle' color gradient plot
        of the polar diagram with respect to the respective boat speeds
    plot_convex_hull(
        ws=(0, np.inf),
        ax=None,
        colors=('green', 'red'),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    )
        Computes the (seperate) convex hull of one or more
        slices of the polar diagram and creates a polar plot of them
    """

    def __init__(self, pts=None, tw=True):
        logger.info(
            f"Class 'PolarDiagramPointcloud(pts={pts}, tw={tw})' called"
        )

        if pts is None:
            self._pts = np.array([])
            return

        try:
            self._pts = convert_wind(pts, -1, tw)
        except WindException as we:
            raise PolarDiagramException("") from we

    def __str__(self):
        table = ["   TWS      TWA     BSP\n", "------  -------  ------\n"]
        for point in self.points:
            for i in range(3):
                entry = f"{float(point[i]):.2f}"
                if i == 1:
                    table.append(entry.rjust(7))
                    table.append("  ")
                    continue

                table.append(entry.rjust(6))
                table.append("  ")
            table.append("\n")
        return "".join(table)

    def __repr__(self):
        return f"PolarDiagramPointcloud(pts={self.points})"

    @property
    def wind_speeds(self):
        """Returns all unique wind speeds in the point cloud"""
        return np.array(sorted(list(set(self.points[:, 0]))))

    @property
    def wind_angles(self):
        """Returns all unique wind angles in the point cloud"""
        return np.array(sorted(list(set(self.points[:, 1]))))

    @property
    def boat_speeds(self):
        return self.points[:, 2]

    @property
    def points(self):
        """Returns a read only version of self._pts"""
        return self._pts.copy()

    def to_csv(self, csv_path):
        """Creates a .csv file with delimiter ',' and the
        following format:
            PolarDiagramPointcloud
            True wind speed ,True wind angle ,Boat speed
            self.points

        Parameters
        ----------
        csv_path : path-like
            Path to a .csv-file or where a new .csv file will be created

        Raises a FileWritingException if an error occurs whilst writing
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
        except OSError as oe:
            raise FileWritingException(
                "While writing to `csv_path` an error occured"
            ) from oe

    def symmetrize(self):
        """Constructs a symmetric version of the polar diagram,
        by mirroring it at the 0° - 180° axis and returning a new instance

        Warning
        -------
        Should only be used if all the wind angles of the initial
        polar diagram are on one side of the 0° - 180° axis,
        otherwise this can result in the construction of duplicate points,
        that might overwrite or live alongside old points
        """
        if not self.points.size:
            return self

        below_180 = [wa for wa in self.wind_angles if wa <= 180]
        above_180 = [wa for wa in self.wind_angles if wa > 180]
        if below_180 and above_180:
            print(
                "Warning: There are wind angles on both sides of the "
                "0° - 180° axis. This might result in duplicate data, "
                "which can overwrite or live alongside old data"
            )

        sym_pts = self.points
        sym_pts[:, 1] = 360 - sym_pts[:, 1]
        pts = np.row_stack((self.points, sym_pts))
        return PolarDiagramPointcloud(pts=pts)

    def add_points(self, new_pts, tw=True):
        """Adds additional points to the point cloud

        Parameters
        ----------
        new_pts: array_like of shape (n, 3)
            New points to be added to the point cloud given as a sequence
            of points consisting of wind speed, wind angle and boat speed

        tw : bool, optional
            Specifies if the given wind data should be viewed as true wind

            If False, wind data will be converted to true wind

            Defaults to True

        Raises a PolarDiagramException

        """
        logger.info(f"Method 'add_points(new_pts{new_pts}, tw={tw})' called")

        try:
            new_pts = convert_wind(new_pts, -1, tw)
        except WindException as we:
            raise PolarDiagramException("") from we

        if not self.points.size:
            self._pts = new_pts
            return

        self._pts = np.row_stack((self.points, new_pts))

    def get_slices(self, ws):
        """
        """
        if isinstance(ws, (int, float)):
            ws = [ws]
        elif isinstance(ws, tuple):
            ws = [w for w in self.wind_speeds if ws[0] <= w <= ws[1]]

        if not ws:
            raise PolarDiagramException(f"No slices were given")

        wa = []
        bsp = []
        for w in ws:
            pts = self.points[self.points[:, 0] == w][:, 1:]
            if not pts.size:
                raise PolarDiagramException(
                    f"No points with wind speed={w} found"
                )
            wa.append(np.deg2rad(pts[:, 0]))
            bsp.append(pts[:, 1])

        if not wa or not bsp:
            raise PolarDiagramException(
                "There are no slices in the given range `ws`"
            )

        return ws, wa, bsp

    def plot_polar(
        self,
        ws=(0, np.inf),
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Creates a polar plot of one or more slices of the polar diagram

        Parameters
        ----------
        ws : tuple of length 2, list, int or float, optional
            Slices of the polar diagram given as either
                - a tuple of two values which
                represent a lower and upper bound
                of considered wind speeds

                - a list of specific wind speeds

                - a single wind speed

            Slices will then consist of all the rows in self.points
            whose first entry is equal to the values in ws

            Defaults to (0, numpy.inf)

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        colors : tuple, optional
            Specifies the colors to be used for the different
            slices. There are four options
                - If as many or more colors as slices are passed,
                each slice will be plotted in the specified color

                - If exactly 2 colors are passed, the slices will
                be plotted with a color gradient consiting of the
                two colors

                - If more than 2 colors but less than slices are passed,
                the first n_color slices will be plotted in the specified
                colors, and the rest will be plotted in the default color
                'blue'

                Alternatively one can specify certain slices
                to be plotted in a certain color by passing
                a tuple of (ws, color) pairs

            Defaults to ('green', 'red')

        show_legend : bool, optional
            Specifies wether or not a legend will be shown
            next to the plot

            The type of legend depends
            on the color options
                - If the slices are plotted with a
                color gradient, a matplotlib.colorbar.Colorbar
                object will be created and assigned to ax.

                - Otherwise a matplotlib.legend.Legend
                will be created and assigned to ax.

            Default to False

        legend_kw : dict, optional
            Keyword arguments to be passed to either the
            matplotlib.colorbar.Colorbar or matplotlib.legend.Legend
            classes to change position and appearence of the legend

            Will only be used if show_legend is True

            If nothing is passed it will default to {}

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot


        Raises a PolarDiagramException if ws is given as a single
        value or a list and there is a value w in ws, such that
        there are no rows in self.points whose first entry
        is equal to w
        """
        logger.info(
            f"Method 'polar_plot(ws={ws}, ax={ax}, colors={colors}, "
            f"show_legend={show_legend}, legend_kw={legend_kw}, "
            f"**plot_kw={plot_kw})' called"
        )

        ws, wa, bsp = self.get_slices(ws)

        plot_polar(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw)

    def plot_flat(
        self,
        ws=(0, np.inf),
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Creates a cartesian plot of one or more slices of the polar diagram

        Parameters
        ----------
        ws : tuple of length 2 or list, optional
            Slices of the polar diagram given as either
                - a tuple of two values which
                represent a lower and upper bound
                of considered wind speeds

                - a list of specific wind speeds

                - a single wind speed

            Slices will then consist of all the rows in self.points
            whose first entry is equal to the values in ws

            Defaults to (0, np.inf)

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        colors : tuple, optional
            Specifies the colors to be used for the different
            slices. There are four options
                - If as many or more colors
                as slices are passed, each
                slice will be plotted in the
                specified color

                - If exactly 2 colors are passed,
                the slices will be plotted with a
                color gradient consiting of the
                two colors

                - If more than 2 colors but less
                than slices are passed, the first
                n_color slices will be plotted in
                the specified colors, and the rest
                will be plotted in the default color 'blue'

                Alternatively one can specify certain slices
                to be plotted in a certain color by passing
                a tuple of (ws, color) pairs

            Defaults to ('green', 'red')

        show_legend : bool, optional
            Specifies wether or not a legend will be shown
            next to the plot

            The type of legend depends
            on the color options
                - If the slices are plotted with a
                color gradient, a matplotlib.colorbar.Colorbar
                object will be created and assigned to ax.

                - Otherwise a matplotlib.legend.Legend
                will be created and assigned to ax.

            Default to False

        legend_kw : dict, optional
            Keyword arguments to be passed to either the
            matplotlib.colorbar.Colorbar or matplotlib.legend.Legend
            classes to change position and appearence of the legend

            Will only be used if show_legend is True

            If nothing is passed it will default to {}

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot

        Raises a PolarDiagramException if ws is given as a single
        value or a list and there is a value w in ws, such that
        there are no rows in self.points whose first entry
        is equal to w
        """
        logger.info(
            f"Method 'flat_plot(ws={ws}, ax={ax}, colors={colors}, "
            f"show_legend={show_legend}, legend_kw={legend_kw}, "
            f"**plot_kw={plot_kw})' called"
        )

        ws, wa, bsp = self.get_slices(ws)
        wa = [np.rad2deg(a) for a in wa]

        plot_flat(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw)

    def plot_3d(self, ax=None, **plot_kw):
        """Creates a 3d plot of the polar diagram

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot


        Raises a PolarDiagramException if there are no points
        in the point cloud
        """
        logger.info(f"Method 'plot_3d(ax={ax}, **plot_kw={plot_kw})' called")

        try:
            ws, wa, bsp = (
                self.points[:, 0],
                self.points[:, 1],
                self.points[:, 2],
            )
        except IndexError as ie:
            raise PolarDiagramException(
                "Point cloud contains no points"
            ) from ie

        wa = np.deg2rad(wa)
        bsp, wa = bsp * np.cos(wa), bsp * np.sin(wa)
        plot3d(ws, wa, bsp, ax, **plot_kw)

    def plot_color_gradient(
        self,
        ax=None,
        colors=("green", "red"),
        marker=None,
        show_legend=False,
        **legend_kw,
    ):
        """Creates a 'wind speed vs. wind angle' color gradient plot
        of the polar diagram with respect to the respective boat speeds

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        colors : tuple of length 2, optional
            Colors which specify the color gradient with
            which the polar diagram will be plotted.

            Defaults to ('green', 'red')

        marker : matplotlib.markers.Markerstyle or equivalent, optional
            Markerstyle for the created scatter plot

            If nothing is passed, it will default to 'o'

        show_legend : bool, optional
            Specifies wether or not a legend will be shown next
            to the plot

            Legend will be a matplotlib.colorbar.Colorbar object.

            Defaults to False

        legend_kw : Keyword arguments
            Keyword arguments to be passed to the
            matplotlib.colorbar.Colorbar class to change position
            and appearence of the legend.

            Will only be used if show_legend is True

            If nothing is passed, it will default to {}


        Raises a PolarDiagramException if there are no points
        in the point cloud
        """
        logger.info(
            f"Method 'plot_color_gradient(ax={ax}, colors={colors}, "
            f"marker={marker}, show_legend={show_legend}, "
            f"**legend_kw={legend_kw})' called"
        )

        try:
            ws, wa, bsp = (
                self.points[:, 0],
                self.points[:, 1],
                self.points[:, 2],
            )
        except IndexError as ie:
            raise PolarDiagramException(
                "Point cloud contains no points"
            ) from ie
        plot_color_gradient(
            ws, wa, bsp, ax, colors, marker, show_legend, **legend_kw
        )

    def plot_convex_hull(
        self,
        ws=(0, np.inf),
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Computes the (seperate) convex hull of one or more
        slices of the polar diagram and creates a polar plot of them

        Parameters
        ----------
        ws : tuple of length 2 or list, optional
            Slices of the polar diagram given as either
                - a tuple of two values which
                represent a lower and upper bound
                of considered wind speeds

                - a list of specific wind speeds

                - a single wind speed

            Slices will then consist of all the rows in self.points
            whose first entry is equal to the values in ws

            Defaults to (0, np.inf)

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

            If nothing is passed, the function will create
            a suitable axes

        colors : tuple, optional
            Specifies the colors to be used for the different
            slices. There are four options
                - If as many or more colors as slices are passed,
                each slice will be plotted in the specified color

                - If exactly 2 colors are passed, the slices will
                be plotted with a color gradient consiting of the
                two colors

                - If more than 2 colors but less than slices are passed,
                the first n_color slices will be plotted in the specified
                colors, and the rest will be plotted in the default color
                'blue'

                Alternatively one can specify certain slices
                to be plotted in a certain color by passing
                a tuple of (ws, color) pairs

            Defaults to ('green', 'red')

        show_legend : bool, optional
            Specifies wether or not a legend will be shown
            next to the plot

            The type of legend depends
            on the color options
                - If the slices are plotted with a
                color gradient, a matplotlib.colorbar.Colorbar
                object will be created and assigned to ax.

                - Otherwise a matplotlib.legend.Legend
                will be created and assigned to ax.

            Default to False

        legend_kw : dict, optional
            Keyword arguments to be passed to either the
            matplotlib.colorbar.Colorbar or matplotlib.legend.Legend
            classes to change position and appearence of the legend

            Will only be used if show_legend is True

            If nothing is passed it will default to {}

        plot_kw : Keyword arguments
            Keyword arguments that will be passed to the
            matplotlib.axes.Axes.plot function, to change
            certain appearences of the plot


        Raises a PolarDiagramException if ws is given as a single
        value or a list and there is a value w in ws, such that
        there are no rows in self.points whose first entry
        is equal to w
        """
        logger.info(
            f"Method 'plot_convex_hull(ws={ws}, ax={ax}, colors={colors},"
            f" show_legend={show_legend}, legend_kw={legend_kw}, "
            f"**plot_kw={plot_kw})' called"
        )

        ws, wa, bsp = self.get_slices(ws)

        plot_convex_hull(
            ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw
        )
