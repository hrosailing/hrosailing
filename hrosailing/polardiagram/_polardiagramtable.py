# pylint: disable=missing-module-docstring
# pylint: disable=too-many-lines

import csv
import enum
import warnings
from ast import literal_eval
from typing import Iterable

import numpy as np

from hrosailing.pipelinecomponents import (
    ArithmeticMeanInterpolator,
    Ball,
    WeightedPoints,
)

from ._basepolardiagram import (
    PolarDiagram,
    PolarDiagramException,
    PolarDiagramInitializationException,
)
from ._plotting import (
    plot_color_gradient,
    plot_convex_hull,
    plot_flat,
    plot_polar,
    plot_surface,
)


class PolarDiagramTable(PolarDiagram):
    """A class to represent, visualize and work with a polar diagram
    in the form of a table.

    Parameters
    ----------
    ws_resolution : array_like, default: `numpy.arange(2, 42, 2)`
        Wind speeds that will correspond to the columns of the table.
        - If array_like, resolution will be `numpy.array(ws_resolution)`.
        - If a scalar `num`, resolution will be `numpy.arange(num, 40, num)`.
    wa_resolution : array_like, default: `numpy.arange(0, 360, 5)`
        Wind angles that will correspond to the rows of the table.
        - If array_like, resolution will be `numpy.array(wa_resolution)`.
        - If a scalar `num`, resolution will be `numpy.arange(num, 360, num)`.
    bsps : array_like, default: `numpy.zeros((rdim, cdim))`
        Boat speeds that will correspond to the entries of the table.
        Needs to have dimensions matching `ws_resolution` and `wa_resolution`.

    Attributes
    ----------
    wind_speeds
    wind_angles
    boat_speeds

    Examples
    --------
    >>> pd = PolarDiagramTable()
    >>> pd.wind_speeds
    [ 2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40]
    >>> pd.wind_angles
    [  0   5  10  15  20  25  30  35  40  45  50  55  60  65  70  75  80
      85  90  95 100 105 110 115 120 125 130 135 140 145 150 155 160 165
     170 175 180 185 190 195 200 205 210 215 220 225 230 235 240 245 250
     255 260 265 270 275 280 285 290 295 300 305 310 315 320 325 330 335
     340 345 350 355]
    >>> pd = PolarDiagramTable(
    ...     bsps=[
    ...         [5.33, 6.32, 6.96, 7.24, 7.35],
    ...         [5.64, 6.61, 7.14, 7.42, 7.56],
    ...         [5.89, 6.82, 7.28, 7.59, 7.84],
    ...         [5.92, 6.98, 7.42, 7.62, 7.93],
    ...         [5.98, 7.07, 7.59, 8.02, 8.34],
    ...         [5.8, 6.95, 7.51, 7.98, 8.52],
    ...         [5.2, 6.41, 7.19, 7.66, 8.14]
    ...     ],
    ...     ws_resolution=[6, 8, 10, 12, 14],
    ...     wa_resolution=[52, 60, 75, 90, 110, 120, 135],
    ... )
    >>> print(pd)
      TWA / TWS    6.0    8.0    10.0    12.0    14.0
    +++++++++++  +++++  +++++  ++++++  ++++++  ++++++
    52.0          5.33   6.32    6.96    7.24    7.35
    60.0          5.64   6.61    7.14    7.42    7.56
    75.0          5.89   6.82    7.28    7.59    7.84
    90.0          5.92   6.98    7.42    7.62    7.93
    110.0         5.98   7.07    7.59    8.02    8.34
    120.0         5.80   6.95    7.51    7.98    8.52
    135.0         5.20   6.41    7.19    7.66    8.14

    Single entries can be extracted like so
    >>> pd[10, 90]
    7.42

    or if a wind speed (resp. wind angle) isn't in the table
    a value can be interpolated
    >>> pd(11, 90)
    7.50924383603392
    """

    def __init__(self, ws_resolution=None, wa_resolution=None, bsps=None):
        ws_resolution = _Resolution.WIND_SPEED.set_resolution(ws_resolution)
        wa_resolution = _Resolution.WIND_ANGLE.set_resolution(wa_resolution)

        if bsps is None:
            self._create_zero_table(ws_resolution, wa_resolution)
            return

        bsps = np.asarray_chkfinite(bsps, float)

        # sanity checks
        if bsps.dtype is object:
            raise PolarDiagramInitializationException(
                "`bsps` is not array_like"
            )
        if _incompatible_shapes(bsps, ws_resolution, wa_resolution):
            raise PolarDiagramInitializationException(
                "`bsps` has incorrect shape"
            )

        (
            self._ws_resolution,
            self._wa_resolution,
            self._boat_speeds,
        ) = _sort_table(ws_resolution, wa_resolution, bsps)

    def _create_zero_table(self, ws_resolution, wa_resolution):
        rows, cols = len(wa_resolution), len(ws_resolution)
        self._boat_speeds = np.zeros((rows, cols))
        self._ws_resolution = sorted(ws_resolution)
        self._wa_resolution = sorted(wa_resolution)

    def __str__(self):
        table = ["  TWA / TWS"]
        ws = self.wind_speeds
        if len(ws) <= 15:
            self._create_short_table(table, ws)
        else:
            wind = []
            wind.extend(ws[:5])
            wind.extend(ws[-5:])
            self._create_long_table(table, wind)

        return "".join(table)

    def _create_short_table(self, table, wind):
        bsps = self.boat_speeds

        table.extend([f"    {float(ws):.1f}" for ws in wind])
        table.append("\n+++++++++++")

        for ws in wind:
            le = len(f"{float(ws):.1f}")
            table.append("  ".ljust(le + 4, "+"))

        table.append("\n")

        for i, wa in enumerate(self.wind_angles):
            table.append(f"{float(wa):.1f}".ljust(11))

            for j, ws in enumerate(wind):
                le = len(f"{float(ws):.1f}")
                table.append(f"{bsps[i][j]:.2f}".rjust(4 + le))

            table.append("\n")

    def _create_long_table(self, table, wind):
        bsps = self.boat_speeds

        for i, ws in enumerate(wind):
            if i == 5:
                table.append("  ...")

            table.append(f"    {float(ws):.1f}")

        table.append("\n+++++++++++")

        for i, ws in enumerate(wind):
            if i == 5:
                table.append("  +++")

            le = len(f"{float(ws):.1f}")
            table.append("  ".ljust(le + 4, "+"))

        table.append("\n")

        for i, wa in enumerate(self.wind_angles):
            table.append(f"{float(wa):.1f}".rjust(11))

            for j, ws in enumerate(wind):
                if j == 5:
                    table.append("  ...")

                le = len(f"{float(ws):.1f}")
                table.append(f"{bsps[i][j]:.2f}".rjust(4 + le))

            table.append("\n")

    def __repr__(self):
        return (
            f"PolarDiagramTable({self.wind_speeds}, {self.wind_angles}, "
            f"{self.boat_speeds})"
        )

    def __call__(
        self,
        ws,
        wa,
        interpolator=ArithmeticMeanInterpolator(params=(50,)),
        neighbourhood=Ball(radius=1),
    ):
        """Calculates the boat speed for given `ws` and `wa`.

        If the `ws-wa` point is in the table, the corresponding entry is
        returned, otherwise the value is interpolated.

        Parameters
        ----------
        ws : int or float
            Wind speed.

        wa : int or float
            Wind angle.

        interpolator : hrosailing.pipelinecomponents.Interpolator, optional
            Interpolator subclass that determines the interpolation
            method used to determine the value at the ws-wa point.

            Defaults to: `hrosailing.pipelinecomponents.ArithmeticMeanInterpolator(params = (50, ))`

        neighbourhood : hrosailing.pipelinecomponents.Neighbourhood, default: `hrosailing.pipelinecomponents.Ball(1)`
            Neighbourhood subclass used to determine the grid points
            in the table that will be used in the interpolation.

        Returns
        -------
        bsp : int or float
            Boat speed value as determined above.
        """
        try:
            return self[ws, wa]
        except PolarDiagramException:
            point = np.array([ws, wa])
            ws, wa = np.meshgrid(self.wind_speeds, self.wind_angles)
            points = np.column_stack(
                (ws.ravel(), wa.ravel(), self.boat_speeds.ravel())
            )

            weighted_points = WeightedPoints(points, weights=1)

            considered_points = neighbourhood.is_contained_in(
                points[:, :2] - point
            )

            return interpolator.interpolate(
                weighted_points[considered_points], point
            )

    def __getitem__(self, *key):
        """Returns the value of a given entry in the table."""
        ws, wa = key[0]

        col = self._get_indices(np.atleast_1d(ws), _Resolution.WIND_SPEED)
        row = self._get_indices(np.atleast_1d(wa), _Resolution.WIND_ANGLE)
        return float(self.boat_speeds[row, col])

    def _get_indices(self, wind, soa):
        res = (
            self.wind_speeds
            if soa.name == _Resolution.WIND_SPEED.name
            else self.wind_angles
        )

        if wind is None:
            return range(len(res))

        wind = soa.normalize_wind(wind)

        # sanity checks
        if not wind:
            raise PolarDiagramException("empty slice-list was passed")

        if not wind.issubset(set(res)):
            raise PolarDiagramException(f"{wind} is not contained in {res}")

        return [i for i, w in enumerate(res) if w in wind]

    @property
    def wind_angles(self):
        """Returns a read only version of `self._wa_resolution`."""
        return self._wa_resolution.copy()

    @property
    def wind_speeds(self):
        """Returns a read only version of `self._ws_resolution`."""
        return self._ws_resolution.copy()

    @property
    def boat_speeds(self):
        """Returns a read only version of `self._boat_speeds`."""
        return self._boat_speeds.copy()

    def to_csv(self, csv_path, fmt="hro"):
        """Creates a .csv file with delimiter ',' and the following format:

            `PolarDiagramTable`
            TWS:
            `self.wind_speeds`
            TWA:
            `self.wind_angles`
            Boat speeds:
            `self.boat_speeds`

        Parameters
        ----------
        csv_path : path_like
            Path for the .csv file.
        fmt : {"hro", "orc", "opencpn", "array"}, default: "hro"
            Format for the .csv file.

        See Also
        --------
        `from_csv`

        Examples
        --------
        >>> pd = PolarDiagramTable(
        ...     bsps=[
        ...         [5.33, 6.32, 6.96, 7.24, 7.35],
        ...         [5.64, 6.61, 7.14, 7.42, 7.56],
        ...         [5.89, 6.82, 7.28, 7.59, 7.84],
        ...         [5.92, 6.98, 7.42, 7.62, 7.93],
        ...         [5.98, 7.07, 7.59, 8.02, 8.34],
        ...         [5.8, 6.95, 7.51, 7.98, 8.52],
        ...         [5.2, 6.41, 7.19, 7.66, 8.14]
        ...     ],
        ...     ws_resolution=[6, 8, 10, 12, 14],
        ...     wa_resolution=[52, 60, 75, 90, 110, 120, 135],
        ... )
        >>> print(pd)
          TWA / TWS    6.0    8.0    10.0    12.0    14.0
        +++++++++++  +++++  +++++  ++++++  ++++++  ++++++
        52.0          5.33   6.32    6.96    7.24    7.35
        60.0          5.64   6.61    7.14    7.42    7.56
        75.0          5.89   6.82    7.28    7.59    7.84
        90.0          5.92   6.98    7.42    7.62    7.93
        110.0         5.98   7.07    7.59    8.02    8.34
        120.0         5.80   6.95    7.51    7.98    8.52
        135.0         5.20   6.41    7.19    7.66    8.14
        >>> pd.to_csv("example.csv")
        >>> pd2 = from_csv("example.csv")
        >>> print(pd2)
          TWA / TWS    6.0    8.0    10.0    12.0    14.0
        +++++++++++  +++++  +++++  ++++++  ++++++  ++++++
        52.0          5.33   6.32    6.96    7.24    7.35
        60.0          5.64   6.61    7.14    7.42    7.56
        75.0          5.89   6.82    7.28    7.59    7.84
        90.0          5.92   6.98    7.42    7.62    7.93
        110.0         5.98   7.07    7.59    8.02    8.34
        120.0         5.80   6.95    7.51    7.98    8.52
        135.0         5.20   6.41    7.19    7.66    8.14
        """
        if fmt not in {"hro", "orc", "opencpn", "array"}:
            raise PolarDiagramException("`fmt` not implemented")

        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            if fmt == "orc":
                self._write_orc_format(file)
                return

            if fmt == "array":
                self._write_array_format(file)
                return

            csv_writer = csv.writer(file, delimiter=",")

            if fmt == "opencpn":
                self._write_opencpn_format(csv_writer)
                return

            self._write_hro_format(csv_writer)

    def _write_orc_format(self, file):
        csv_writer = csv.writer(file, delimiter=";")
        csv_writer.writerow(["twa/tws"] + list(self.wind_speeds))
        csv_writer.writerow([0] * (len(self.wind_speeds) + 1))

        self._write_rows(csv_writer)

    def _write_rows(self, csv_writer):
        rows = np.column_stack((self.wind_angles, self.boat_speeds))
        csv_writer.writerows(rows)

    def _write_array_format(self, file):
        csv_writer = csv.writer(file, delimiter="\t")
        csv_writer.writerow([r"TWA\TWS"] + list(self.wind_speeds))

        self._write_rows(csv_writer)

    def _write_opencpn_format(self, csv_writer):
        csv_writer.writerow([r"TWA \ TWS"] + list(self.wind_speeds))

        self._write_rows(csv_writer)

    def _write_hro_format(self, csv_writer):
        csv_writer.writerow([self.__class__.__name__])
        csv_writer.writerow(["TWS"])
        csv_writer.writerow(self.wind_speeds)
        csv_writer.writerow(["TWA"])
        csv_writer.writerow(self.wind_angles)
        csv_writer.writerow(["BSP"])
        csv_writer.writerows(self.boat_speeds)

    @classmethod
    def __from_csv__(cls, file):
        csv_reader = csv.reader(file, delimiter=",")
        next(csv_reader)
        ws_res = [literal_eval(ws) for ws in next(csv_reader)]
        next(csv_reader)
        wa_res = [literal_eval(wa) for wa in next(csv_reader)]
        next(csv_reader)
        bsps = [[literal_eval(bsp) for bsp in row] for row in csv_reader]

        return PolarDiagramTable(ws_res, wa_res, bsps)

    def symmetrize(self):
        """Constructs a symmetric version of the polar diagram.

        Symmetrizes polar diagram by mirroring it at the 0° - 180° axis
        and returning a new instance.

        Warnings
        --------
        Should only be used if all the wind angles of the initial
        polar diagram are on one side of the 0° - 180° axis,
        otherwise this can lead to duplicate data, which can
        overwrite or live alongside old data.

        Examples
        --------
        >>> pd = PolarDiagramTable(
        ...     bsps=[
        ...         [5.33, 6.32, 6.96, 7.24, 7.35],
        ...         [5.64, 6.61, 7.14, 7.42, 7.56],
        ...         [5.89, 6.82, 7.28, 7.59, 7.84],
        ...         [5.92, 6.98, 7.42, 7.62, 7.93],
        ...     ],
        ...     ws_resolution = [6, 8, 10, 12, 14],
        ...     wa_resolution = [52, 60, 75, 90]
        ... )
        >>> print(pd)
          TWA / TWS    6.0    8.0    10.0    12.0    14.0
        +++++++++++  +++++  +++++  ++++++  ++++++  ++++++
        52.0          5.33   6.32    6.96    7.24    7.35
        60.0          5.64   6.61    7.14    7.42    7.56
        75.0          5.89   6.82    7.28    7.59    7.84
        90.0          5.92   6.98    7.42    7.62    7.93
        >>> sym_pd = pd.symmetrize()
        >>> print(sym_pd)
          TWA / TWS    6.0    8.0    10.0    12.0    14.0
        +++++++++++  +++++  +++++  ++++++  ++++++  ++++++
        52.0          5.33   6.32    6.96    7.24    7.35
        60.0          5.64   6.61    7.14    7.42    7.56
        75.0          5.89   6.82    7.28    7.59    7.84
        90.0          5.92   6.98    7.42    7.62    7.93
        270.0         5.92   6.98    7.42    7.62    7.93
        285.0         5.89   6.82    7.28    7.59    7.84
        300.0         5.64   6.61    7.14    7.42    7.56
        308.0         5.33   6.32    6.96    7.24    7.35
        """

        symmetric_wa_resolution = np.concatenate(
            [self.wind_angles, 360 - np.flip(self.wind_angles)]
        )
        symmetric_bsps = np.row_stack(
            (self.boat_speeds, np.flip(self.boat_speeds, axis=0))
        )

        if 180 in self.wind_angles:
            (
                symmetric_wa_resolution,
                symmetric_bsps,
            ) = _delete_multiple_180_degree_occurrences(
                symmetric_wa_resolution, symmetric_bsps
            )
        if 0 in self.wind_angles:
            (
                symmetric_wa_resolution,
                symmetric_bsps,
            ) = _delete_multiple_0_degree_occurrences(
                symmetric_wa_resolution, symmetric_bsps
            )

        return PolarDiagramTable(
            ws_resolution=self.wind_speeds,
            wa_resolution=symmetric_wa_resolution,
            bsps=symmetric_bsps,
        )

    def change_entries(self, new_bsps, ws=None, wa=None):
        """Changes specified entries in the table.

        Parameters
        ----------
        new_bsps: array_like of matching shape
            Sequence containing the new boat speeds to be inserted
            in the specified entries.
        ws: array_like, default: `self.wind_speeds`
            Columns of table given as elements of `self.wind_speed`.
        wa: array_like, default: `self.wind_angles`
            Rows of table given as elements of `self.wind_angles`.

        Examples
        --------
        >>> pd = PolarDiagramTable(
        ...     ws_resolution=[6, 8, 10, 12, 14],
        ...     wa_resolution=[52, 60, 75, 90, 110, 120, 135]
        ... )
        >>> print(pd)
          TWA / TWS    6.0    8.0    10.0    12.0    14.0
        +++++++++++  +++++  +++++  ++++++  ++++++  ++++++
        52.0          0.00   0.00    0.00    0.00    0.00
        60.0          0.00   0.00    0.00    0.00    0.00
        75.0          0.00   0.00    0.00    0.00    0.00
        90.0          0.00   0.00    0.00    0.00    0.00
        110.0         0.00   0.00    0.00    0.00    0.00
        120.0         0.00   0.00    0.00    0.00    0.00
        135.0         0.00   0.00    0.00    0.00    0.00
        >>> pd.change_entries(
        ...     new_bsps=[5.33, 5.64, 5.89, 5.92, 5.98, 5.8, 5.2],
        ...     ws=6
        ... )
        >>> print(pd)
          TWA / TWS    6.0    8.0    10.0    12.0    14.0
        +++++++++++  +++++  +++++  ++++++  ++++++  ++++++
        52.0          5.33   0.00    0.00    0.00    0.00
        60.0          5.64   0.00    0.00    0.00    0.00
        75.0          5.89   0.00    0.00    0.00    0.00
        90.0          5.92   0.00    0.00    0.00    0.00
        110.0         5.98   0.00    0.00    0.00    0.00
        120.0         5.80   0.00    0.00    0.00    0.00
        135.0         5.20   0.00    0.00    0.00    0.00
        >>> pd.change_entries(
        ...     new_bsps=[5.7, 6.32, 6.96, 7.24, 7.35],
        ...     wa=52
        ... )
        >>> print(pd)
          TWA / TWS    6.0    8.0    10.0    12.0    14.0
        +++++++++++  +++++  +++++  ++++++  ++++++  ++++++
        52.0          5.70   6.32    6.96    7.24    7.35
        60.0          5.64   0.00    0.00    0.00    0.00
        75.0          5.89   0.00    0.00    0.00    0.00
        90.0          5.92   0.00    0.00    0.00    0.00
        110.0         5.98   0.00    0.00    0.00    0.00
        120.0         5.80   0.00    0.00    0.00    0.00
        135.0         5.20   0.00    0.00    0.00    0.00
        """
        new_bsps = np.atleast_1d(new_bsps)

        new_bsps = np.asarray_chkfinite(new_bsps)

        if new_bsps.dtype == object:
            raise PolarDiagramException("`new_bsps` is not array_like")

        ws = self._get_indices(ws, _Resolution.WIND_SPEED)
        wa = self._get_indices(wa, _Resolution.WIND_ANGLE)

        wa_len = len(wa) == 1
        ws_len = len(ws) == 1

        # wrong shape can lead to missing assignments
        if wa_len and ws_len:
            correct_shape = new_bsps.shape == (1,)
        elif wa_len:
            correct_shape = new_bsps.shape == (len(ws),)
        elif ws_len:
            correct_shape = new_bsps.shape == (len(wa),)
        else:
            correct_shape = new_bsps.shape == (len(wa), len(ws))

        if not correct_shape:
            raise PolarDiagramException("`new_bsps` has wrong shape")

        mask = np.zeros(self.boat_speeds.shape, dtype=bool)
        for i in wa:
            for j in ws:
                mask[i, j] = True

        self._boat_speeds[mask] = new_bsps.flat

    def get_slices(self, ws=None):
        """Returns given slice of polar diagram.

        The slices are equal to the corresponding columns of the table
        together with `self.wind_angles`.

        Parameters
        ----------
        ws : array_like, default: `self.wind_speeds`
            Slices of the polar diagram table, given as either:
            - a tuple of length 2 specifying an interval of
                considered wind speeds,
            - an iterable containing only elements of `self.wind_speeds`,
            - a single element of `self.wind_speeds`.

        Returns
        -------
        slices : tuple
            Slices of the polar diagram, given as a tuple of length 3,
            consisting of the given wind speeds `ws`, `self.wind_angles`
            (in rad) and an array with the corresponding columns of the table.

        Raises
        ------
        PolarDiagramException
            If at least one element of `ws` is not in `self.wind_speeds`.
        PolarDiagramException
            If the given interval doesn't contain any slices of the
            polar diagram.
        """
        if ws is None:
            ws = self.wind_speeds
        elif isinstance(ws, (int, float)):
            ws = [ws]
        elif isinstance(ws, tuple) and len(ws) == 2:
            ws = [w for w in self.wind_speeds if ws[0] <= w <= ws[1]]

        ws = sorted(list(ws))
        if not ws:
            raise PolarDiagramException("no slices were given")

        ind = self._get_indices(ws, _Resolution.WIND_SPEED)
        wa = np.deg2rad(self.wind_angles)

        bsp = self.boat_speeds[:, ind]
        if max(wa) > np.deg2rad(350) and min(wa) < np.deg2rad(10):
            bsp = np.concatenate([bsp, [bsp[0]]])
            wa = np.concatenate([wa, [wa[0]]])
        return ws, wa, bsp

    def plot_polar(
        self,
        ws=None,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Creates a polar plot of one or more slices of the polar diagram.

        Parameters
        ----------
        ws : array_like, default: `self.wind_speeds`
            Slices of the polar diagram table, given as either:
            - a tuple of length 2 specifying an interval of
                considered wind speeds,
            - an iterable containing only elements of `self.wind_speeds`,
            - a single element of `self.wind_speeds`.
            The slices are then equal to the corresponding
            columns of the table together with `self.wind_angles`.

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

        colors : color_like or sequence of color_likes or (ws, color_like) pairs
            Specifies the colors to be used for the different slices.

            - If a color_like is passed, all slices will be plotted in the
            respective color.
            - If 2 colors are passed, slices will be plotted with a color
            gradient that is determined by the corresponding wind speed.
            - Otherwise the slices will be colored in turn with the specified
            colors or the color `"blue"`, if there are too few colors. The
            order is determined by the corresponding wind speeds.
            - Alternatively one can specify certain slices to be plotted in
            a color out of order by passing a sequence of `(ws, color)` pairs.

            Defaults to `("green", "red")`.

        show_legend : bool, default: `False`
            Specifies wether or not a legend will be shown next to the plot.

            The type of legend depends on the color options.

            If plotted with a color gradient, a `matplotlib.colorbar.Colorbar`
            will be created, otherwise a `matplotlib.legend.Legend` instance.

        Other Parameters
        ----------------
        legend_kw : dict, optional
            Keyword arguments to change position and appearance of the colorbar
            or legend respectively.
            - If 2 colors are passed, a colorbar will be created.
            In this case see `matplotlib.colorbar.Colorbar` for possible
            keywords and their effect.
            - Otherwise, a legend will be created.
            In this case see `matplotlib.legend.Legend` for possible keywords
            and their effect.
        plot_kw : dict, optional
            Keyword arguments to change various appearances of the plot.
            See `matplotlib.axes.Axes.plot` for possible keywords and their
            effects.

        Raises
        ------
        PolarDiagramException
            If at least one element of `ws` is not in `self.wind_speeds`.
        PolarDiagramException
            If the given interval doesn't contain any slices of the
            polar diagram.

        Examples
        --------
        >>> import matplotlib.pyplot as pyplot
        >>> pd = from_csv("src/polar_diagrams/orc/A-35.csv", fmt="orc")
        >>> pd.plot_polar(
        ...     ws=[6, 8], show_legend=True, ls="-", lw=1.5, marker=""
        ... )
        >>> pyplot.show()

        ![logo](https://raw.githubusercontent.com/hrosailing/hrosailing/main/examples/pictures/table_plot_polar.png)
        """
        ws, wa, bsp = self.get_slices(ws)
        bsp = list(bsp.T)
        wa = [wa] * len(bsp)
        plot_polar(
            ws,
            wa,
            bsp,
            ax,
            colors,
            show_legend,
            legend_kw,
            _lines=True,
            **plot_kw,
        )

    def plot_flat(
        self,
        ws=None,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Creates a cartesian plot of one or more slices of the polar diagram.

        Parameters
        ----------
        ws : array_like, default: `self.wind_speeds`
            Slices of the polar diagram table, given as either:
            - a tuple of length 2 specifying an interval of considered
            wind speeds,
            - an iterable containing only elements of `self.wind_speeds`,
            - a single element of `self.wind_speeds`.

            The slices are then equal to the corresponding
            columns of the table together with `self.wind_angles`.
        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.
        colors : color_like or sequence of color_likes or (ws, color_like) pairs
            Specifies the colors to be used for the different slices.
            - If a color_like is passed, all slices will be plotted in the
            respective color.
            - If 2 colors are passed, slices will be plotted with a color
            gradient that is determined by the corresponding wind speed.
            - Otherwise the slices will be colored in turn with the specified
            colors or the color `"blue"`, if there are too few colors. The
            order is determined by the corresponding wind speeds.
            - Alternatively one can specify certain slices to be plotted in
            a color out of order by passing a sequence of `(ws, color)` pairs.

            Defaults to `("green", "red")`.

        show_legend : bool, default: `False`
            Specifies wether or not a legend will be shown next to the plot.

            The type of legend depends on the color options.

            If plotted with a color gradient, a `matplotlib.colorbar.Colorbar`
            will be created, otherwise a `matplotlib.legend.Legend` instance.
        legend_kw : dict, optional
            Keyword arguments to change position and appearance of the colorbar
            or legend respectively.
            - If 2 colors are passed, a colorbar will be created.
            In this case see `matplotlib.colorbar.Colorbar` for possible
            keywords and their effect.
            - Otherwise, a legend will be created.
            In this case see `matplotlib.legend.Legend` for possible keywords
            and their effect.
        plot_kw : dict, optional
            Keyword arguments to change various appearances of the plot.

            See `matplotlib.axes.Axes.plot` for possible keywords and their
            effects.

        Raises
        ------
        PolarDiagramException
            If at least one element of `ws` is not in `self.wind_speeds`.
        PolarDiagramException
            If the given interval doesn't contain any slices of the
            polar diagram.

        Examples
        --------
        >>> import matplotlib.pyplot as pyplot
        >>> pd = from_csv("src/polar_diagrams/orc/A-35.csv", fmt="orc")
        >>> pd.plot_flat(
        ...     ws=[6, 8], show_legend=True, ls="-", lw=1.5, marker=""
        ... )
        >>> pyplot.show()

        ![logo](https://raw.githubusercontent.com/hrosailing/hrosailing/main/examples/pictures/table_plot_flat.png)
        """
        ws, wa, bsp = self.get_slices(ws)
        bsp = list(bsp.T)
        wa = [np.rad2deg(wa)] * len(bsp)
        plot_flat(
            ws,
            wa,
            bsp,
            ax,
            colors,
            show_legend,
            legend_kw,
            _lines=True,
            **plot_kw,
        )

    def plot_3d(self, ax=None, colors=("green", "red")):
        """Creates a 3d plot of the polar diagram.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            Axes instance where the plot will be created.
        colors: tuple of two (2) color_likes, default: `("green", "red")`
            Color pair determining the color gradient with which the
            polar diagram will be plotted.

            Will be determined by the corresponding wind speeds.
        """
        wa = np.deg2rad(self.wind_angles)
        ws, wa = np.meshgrid(self.wind_speeds, wa)
        bsp = self.boat_speeds
        bsp, wa = bsp * np.cos(wa), bsp * np.sin(wa)

        plot_surface(ws, wa, bsp, ax, colors)

    def plot_color_gradient(
        self,
        ax=None,
        colors=("green", "red"),
        marker=None,
        ms=None,
        show_legend=False,
        **legend_kw,
    ):
        """Creates a 'wind speed vs. wind angle' color gradient plot
        of the polar diagram with respect to the corresponding boat speeds.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.
        colors : tuple of two (2) color_likes, default: `("green", "red")`
            Color pair determining the color gradient with which the
            polar diagram will be plotted.

            Will be determined by the corresponding boat speed.
        marker : matplotlib.markers.Markerstyle or equivalent
            Markerstyle for the created scatter plot.
            Defaults to `"o"`.
        ms : array_like, optional
            Marker size in points**2.
        show_legend : bool, default: `False`
            Specifies whether a legend will be shown next
            to the plot.

            Legend will be a `matplotlib.colorbar.Colorbar` instance.
        legend_kw : dict, optional
            Keyword arguments to change position and appearance of the colorbar
            or legend respectively.
            - If 2 colors are passed, a colorbar will be created.
            In this case see `matplotlib.colorbar.Colorbar` for possible
            keywords and their effect.
            - Otherwise, a legend will be created.
            In this case see `matplotlib.legend.Legend` for possible keywords
            and their effect.
        """
        ws, wa = np.meshgrid(self.wind_speeds, self.wind_angles)
        ws = ws.ravel()
        wa = wa.ravel()
        bsp = self.boat_speeds.ravel()
        plot_color_gradient(
            ws, wa, bsp, ax, colors, marker, ms, show_legend, **legend_kw
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
        """Computes the (separate) convex hull of one or more
        slices of the polar diagram and creates a polar plot of them.

        Parameters
        ----------
        ws : array_like, default: `self.wind_speeds`
            Slices of the polar diagram table, given as either:
            - a tuple of length 2 specifying an interval of considered
            wind speeds,
            - an iterable containing only elements of `self.wind_speeds`,
            - a single element of `self.wind_speeds`.

            The slices are then equal to the corresponding
            columns of the table together with `self.wind_angles`.
        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.
        colors : sequence of color_likes or (ws, color_like) pairs, optional
            Specifies the colors to be used for the different slices.

            - If 2 colors are passed, slices will be plotted with a color
            gradient that is determined by the corresponding wind speed.
            - Otherwise the slices will be colored in turn with the specified
            colors or the color `"blue"`, if there are too few colors. The
            order is determined by the corresponding wind speeds.
            - Alternatively one can specify certain slices to be plotted in
            a color out of order by passing a sequence of `(ws, color)` pairs.

            Defaults to `("green", "red")`
        show_legend : bool, default: `False`
            Specifies whether a legend will be shown next to the plot.

            The type of legend depends on the color options.

            If plotted with a color gradient, a `matplotlib.colorbar.Colorbar`
            will be created, otherwise a `matplotlib.legend.Legend`.
        legend_kw : dict, optional
            Keyword arguments to change position and appearance of the colorbar
            or legend respectively.

            - If 2 colors are passed, a colorbar will be created.
            In this case see `matplotlib.colorbar.Colorbar` for possible
            keywords and their effect.
            - Otherwise, a legend will be created.
            In this case see `matplotlib.legend.Legend` for possible keywords
            and their effect.
        plot_kw : dict, optional
            Keyword arguments to change various appearances of the plot.

            See `matplotlib.axes.Axes.plot` for possible keywords and their
            effects.

        Raises
        ------
        PolarDiagramException
            If at least one element of `ws` is not in `self.wind_speeds`.
        PolarDiagramException
            If the given interval doesn't contain any slices of the
            polar diagram.
        """
        ws, wa, bsp = self.get_slices(ws)
        bsp = list(bsp.T)
        wa = [wa] * len(bsp)
        plot_convex_hull(
            ws,
            wa,
            bsp,
            ax,
            colors,
            show_legend,
            legend_kw,
            _lines=True,
            **plot_kw,
        )


class _Resolution(enum.Enum):
    WIND_SPEED = (np.arange(2, 42, 2), 40)
    WIND_ANGLE = (np.arange(0, 360, 5), 360)

    def __init__(self, standard_res, max_value):
        self.standard_res = standard_res
        self.max_value = max_value

    def set_resolution(self, res):
        if res is None:
            return self.standard_res

        if isinstance(res, Iterable):
            return self._custom_iterable_resolution(res)

        return self._custom_stepsize_resolution(res)

    def _custom_iterable_resolution(self, res):
        # NaN's and infinite values can cause problems later on
        res = np.asarray_chkfinite(res)

        if res.dtype == object:
            raise ValueError("`res` is not array_like")

        if not res.size or res.ndim != 1:
            raise ValueError("`res` has incorrect shape")

        if len(set(res)) != len(res):
            warnings.warn(
                "`res` contains duplicate data. "
                "This may lead to unwanted behaviour"
            )

        if self.name == _Resolution.WIND_SPEED.name:
            if np.any((res < 0)):
                raise ValueError("`res` contains negative entries")

        if self.name == _Resolution.WIND_ANGLE.name:
            res %= 360

        return res

    def _custom_stepsize_resolution(self, res):
        if res <= 0:
            raise ValueError("`res` is non-positive")

        if res > self.max_value:
            raise ValueError(
                f"Resolution stepsize ({res}) is bigger than the maximal"
                f" resolution value ({self.max_value})."
            )

        return np.arange(res, self.max_value, res)

    def normalize_wind(self, wind):
        wind = np.atleast_1d(wind)  # allow scalar inputs

        if self.name == _Resolution.WIND_SPEED.name:
            if np.any((wind < 0)):
                raise ValueError("`wind` is negative")

        if self.name == _Resolution.WIND_ANGLE.name:
            wind %= 360

        return set(wind)


def _incompatible_shapes(bsps, ws_resolution, wa_resolution):
    rows, cols = len(wa_resolution), len(ws_resolution)
    return bsps.shape != (rows, cols)


def _sort_table(ws_resolution, wa_resolution, bsps):
    wa_resolution, bsps = zip(
        *sorted(zip(wa_resolution, bsps), key=lambda x: x[0])
    )
    bsps = np.array(bsps, float)

    ws_resolution, bsps = zip(
        *sorted(zip(ws_resolution, bsps.T), key=lambda x: x[0])
    )

    return (
        np.array(ws_resolution, float),
        np.array(wa_resolution, float),
        np.array(bsps, float).T,
    )


def _delete_multiple_180_degree_occurrences(wa_resolution, bsps):
    mid = np.where(wa_resolution == 180)[0][0]
    wa_resolution = np.delete(wa_resolution, mid)
    bsps = np.row_stack((bsps[:mid, :], bsps[mid + 1 :, :]))

    return wa_resolution, bsps


def _delete_multiple_0_degree_occurrences(wa_resolution, bsps):
    return wa_resolution[:-1], bsps[:-1, :]
