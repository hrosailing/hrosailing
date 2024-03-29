# pylint: disable=missing-module-docstring
# pylint: disable=too-many-lines

import csv
import enum
import warnings
from ast import literal_eval
from typing import Iterable

import numpy as np

from hrosailing.core.data import WeightedPoints
from hrosailing.processing import ArithmeticMeanInterpolator, Ball

from ._basepolardiagram import PolarDiagram


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
        Needs to have attributes matching `ws_resolution` and `wa_resolution`.

    Attributes
    ----------
    wind_angles (property) : numpy.ndarray
        Read only version of the wind angle resolution.

    wind_speeds (property) : numpy.ndarray
        Read only version of the wind speed resolution.

    boat_speeds (property) : numpy.ndarray
        Read only version of the boat speed table.

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

    def get_slices(
        self,
        ws=None,
        n_steps=None,
        full_info=False,
        interpolator=ArithmeticMeanInterpolator(params=(50,)),
        **kwargs,
    ):
        """
        Parameters
        ----------------
        interpolator : Interpolator, optional
            The interpolator used to estimate boat speeds
            not covered in the table.

            Defaults to `ArithmeticMeanInterpolator(params=(50,)`

        See also
        -------
        `PolarDiagram.get_slices`
        """
        kwargs["interpolator"] = interpolator
        return super().get_slices(ws, n_steps, full_info, **kwargs)

    def ws_to_slices(
        self,
        ws,
        interpolator=ArithmeticMeanInterpolator(params=(50,)),
        **kwargs,
    ):
        """
        See also
        -------
        `PolarDiagramTable.get_slices`
        `PolarDiagram.ws_to_slices`
        """
        slices = []
        for ws_ in ws:
            if ws_ in self.wind_speeds:
                bsp = self.boat_speeds.T[np.where(self.wind_speeds == ws_)][0]
            else:
                bsp = [
                    self(ws_, wa_, interpolator) for wa_ in self.wind_angles
                ]
            slices.append(
                np.row_stack(
                    [[ws_] * len(self.wind_angles), self.wind_angles, bsp]
                )
            )
        return slices

    def __init__(self, ws_resolution=None, wa_resolution=None, bsps=None):
        ws_resolution = _Resolution_helper.build_wind_speed_resolution(
            ws_resolution
        )
        wa_resolution = _Resolution_helper.build_wind_angle_resolution(
            wa_resolution
        )

        if bsps is None:
            self._create_zero_table(ws_resolution, wa_resolution)
            return

        bsps = np.asarray_chkfinite(bsps, float)

        # sanity checks
        if bsps.dtype is object:
            raise TypeError("`bsps` is not array_like")
        if _incompatible_shapes(bsps, ws_resolution, wa_resolution):
            raise ValueError("`bsps` has incorrect shape")

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

        neighbourhood : hrosailing.pipelinecomponents.Neighbourhood, optional
            Neighbourhood subclass used to determine the grid points
            in the table that will be used in the interpolation.

            Defaults to `hrosailing.pipelinecomponents.Ball(1)`

        Returns
        -------
        bsp : int or float
            Boat speed value as determined above.
        """
        try:
            return self[ws, wa]
        except (TypeError, ValueError):
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

        col = self._get_indices(np.atleast_1d(ws), _Resolution_type.WIND_SPEED)
        row = self._get_indices(np.atleast_1d(wa), _Resolution_type.WIND_ANGLE)
        return float(self.boat_speeds[row, col])

    def _get_indices(self, wind, resolution_type):
        res = (
            self.wind_speeds
            if resolution_type == _Resolution_type.WIND_SPEED
            else self.wind_angles
        )

        if wind is None:
            return range(len(res))

        wind = _Resolution_helper.normalize_wind(wind, resolution_type)

        # sanity checks
        if not wind:
            raise ValueError("empty slice-list was passed")

        if not wind.issubset(set(res)):
            raise ValueError(f"{wind} is not contained in {res}")

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
    def default_slices(self):
        """
        See also
        ---------
        `Polardiagram.default_slices`
        """
        return self.wind_speeds

    @property
    def default_points(self):
        """
        See also
        --------
        `Polardiagram.default_slices`
        """
        x, y = np.meshgrid(self.wind_speeds, self.wind_angles)
        wind = np.array(list(zip(x.ravel(), y.ravel())))
        bsps = self.boat_speeds.ravel()
        return np.column_stack([wind, bsps])

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
        fmt : {"hro", "orc", "opencpn", "array"}, optional
            Format for the .csv file.

            Defaults to "hro".

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
            raise NotImplementedError("`fmt` not implemented")

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
        ws: array_like, optional
            Columns of table given as elements of `self.wind_speed`.

            Defaults to `self.wind_speeds`.

        wa: array_like, optional
            Rows of table given as elements of `self.wind_angles`.

            Defaults to `self.wind_angles`.

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
            raise TypeError("`new_bsps` is not array_like")

        ws = self._get_indices(ws, _Resolution_type.WIND_SPEED)
        wa = self._get_indices(wa, _Resolution_type.WIND_ANGLE)

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
            raise ValueError("`new_bsps` has wrong shape")

        mask = np.zeros(self.boat_speeds.shape, dtype=bool)
        for i in wa:
            for j in ws:
                mask[i, j] = True

        self._boat_speeds[mask] = new_bsps.flat


class _Resolution_type(enum.Enum):
    WIND_SPEED = "WIND_SPEED"
    WIND_ANGLE = "WIND_ANGLE"


class _Resolution_helper:
    WIND_SPEED_STANDARD_RESOLUTION = np.arange(2, 42, 2)
    WIND_SPEED_STANDARD_MAX_VALUE = 40
    WIND_ANGLE_STANDARD_RESOLUTION = np.arange(0, 360, 5)
    WIND_ANGLE_STANDARD_MAX_VALUE = 360

    @staticmethod
    def build_wind_speed_resolution(descriptive_resolution=None):
        return _Resolution_helper._descriptive_resolution_to_ndarray(
            _Resolution_type.WIND_SPEED, descriptive_resolution
        )

    @staticmethod
    def build_wind_angle_resolution(descriptive_resolution=None):
        return _Resolution_helper._descriptive_resolution_to_ndarray(
            _Resolution_type.WIND_ANGLE, descriptive_resolution
        )

    def __len__(self):
        return len(self.resolution)

    def __iter__(self):
        yield from self.resolution

    @staticmethod
    def _get_standard_resolution(resolution_type):
        if resolution_type == _Resolution_type.WIND_SPEED:
            return _Resolution_helper.WIND_SPEED_STANDARD_RESOLUTION

        if resolution_type == _Resolution_type.WIND_ANGLE:
            return _Resolution_helper.WIND_ANGLE_STANDARD_RESOLUTION

        raise TypeError(
            'A resolution has to be one of "WIND_SPEED" or "WIND_ANGLE"'
        )

    @staticmethod
    def _get_standard_max_value(resolution_type):
        if resolution_type == _Resolution_type.WIND_SPEED:
            return _Resolution_helper.WIND_SPEED_STANDARD_MAX_VALUE

        if resolution_type == _Resolution_type.WIND_ANGLE:
            return _Resolution_helper.WIND_ANGLE_STANDARD_MAX_VALUE

        raise TypeError(
            'A resolution has to be one of "WIND_SPEED" or "WIND_ANGLE"'
        )

    @staticmethod
    def _descriptive_resolution_to_ndarray(resolution_type, res):
        if res is None:
            return _Resolution_helper._get_standard_resolution(resolution_type)

        if isinstance(res, Iterable):
            return _Resolution_helper._custom_iterable_resolution(
                resolution_type, res
            )

        return _Resolution_helper._custom_stepsize_resolution(
            resolution_type, res
        )

    @staticmethod
    def _custom_iterable_resolution(resolution_type, res):
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

        if resolution_type == _Resolution_type.WIND_SPEED:
            if np.any((res < 0)):
                raise ValueError("`res` contains negative entries")

        if resolution_type == _Resolution_type.WIND_ANGLE:
            res %= 360

        return res

    @staticmethod
    def _custom_stepsize_resolution(resolution_type, res):
        if res <= 0:
            raise ValueError("`res` is non-positive")

        max_value = _Resolution_helper._get_standard_max_value(resolution_type)

        if res > max_value:
            raise ValueError(
                f"Resolution stepsize ({res}) is bigger than the maximal"
                f" resolution value ({max_value})."
            )

        return np.arange(res, max_value, res)

    @staticmethod
    def normalize_wind(wind, resolution_type):
        wind = np.atleast_1d(wind)  # allow scalar inputs

        if resolution_type == _Resolution_type.WIND_SPEED:
            if np.any((wind < 0)):
                raise ValueError("`wind` is negative")

        if resolution_type == _Resolution_type.WIND_ANGLE:
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
