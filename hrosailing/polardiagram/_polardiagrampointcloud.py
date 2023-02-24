# pylint: disable=missing-module-docstring

import csv
import warnings
from ast import literal_eval

import numpy as np

from hrosailing.processing import (
    ArithmeticMeanInterpolator,
    Ball,
)
from ..core.data import WeightedPoints
from hrosailing.core.computing import convert_apparent_wind_to_true

from ._basepolardiagram import PolarDiagram


class PolarDiagramPointcloud(PolarDiagram):
    """A class to represent, visualize and work with a polar diagram
    given by a point cloud.

    Parameters
    ----------
    points : array_like of shape (n, 3)
        Initial points of the point cloud, given as a sequence of
        points consisting of wind speed, wind angle and
        boat speed.
        Points with negative wind speeds will be ignored.

    apparent_wind : bool, optional
        Specifies if wind data is given in apparent wind.

        If `True`, data will be converted to true wind.

        Defaults to `False`.

    Attributes
    -----------
    wind_speeds (property) : numpy.ndarray
        All unique wind speeds in the point cloud.

    wind_angles (property) : numpy.ndarray
        All unique wind angles in the point cloud.

    boat_speeds (property) :
        All occurring boat speeds in the point cloud
        (including duplicates).

    points (property) :
        Read only version of all points present in the point cloud.

    See also
    ---------
    `PolarDiagram`
    """

    @property
    def default_points(self):
        """
        Returns all points given in the data.

        See also
        -------
        `PolarDiagram.default_points`
        """
        return self.points

    def get_slices(
        self,
        ws=None,
        n_steps=None,
        full_info=False,
        wa_resolution=100,
        range_=1,
        interpolator=None,
        **kwargs,
    ):
        """
        Parameters
        ---------------------
        wa_resolution : int, optional
            The number of wind angles that will be used for estimation if an
            interpolator is given.

            Defaults to 100.

        range_ : int, optional
            If no interpolator is given, a slice consists of all data points
            with a wind speed with an absolute difference of at most `range_`
            to the corresponding wind speed value.

            Defaults to 1

        interpolator : Interpolator, optional
            The interpolation method used to estimate unknown boat speeds.

            Defaults to `None`

        See also
        --------
        `PolarDiagram.get_slices`
        """
        kwargs["wa_resolution"] = wa_resolution
        kwargs["range_"] = range_
        kwargs["interpolator"] = interpolator
        return super().get_slices(ws, n_steps, full_info, **kwargs)

    def ws_to_slices(
        self, ws, wa_resolution=100, range_=1, interpolator=None, **kwargs
    ):
        """
        See also
        -------
        `PolarDiagramPointcloud.get_slices`
        `PolarDiagram.ws_to_slices`
        """
        all_ws = self.points[:, 0]
        if interpolator is not None:
            default_wind_angles = np.linspace(0, 360, wa_resolution)

        slices = []
        for ws_ in ws:
            if interpolator is None:
                slicing = np.where(np.abs(all_ws - ws_) <= range_)
                slices.append(np.atleast_2d(self.points[slicing]).T)
                continue
            slices.append(self(ws_, default_wind_angles))
        return slices

    def __init__(self, points, apparent_wind=False):
        if apparent_wind:
            points = convert_apparent_wind_to_true(points)
        else:
            points = np.asarray_chkfinite(points)
            points = points[np.where(points[:, 0] >= 0)]
            points[:, 1] %= 360

        self._points = points

    def __str__(self):
        table = ["   TWS      TWA     BSP\n", "++++++  +++++++  ++++++\n"]
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

    def __call__(
        self,
        ws,
        wa,
        interpolator=ArithmeticMeanInterpolator(50),
        neighbourhood=Ball(radius=1),
    ):
        """Returns the value of the polar diagram at a given `ws-wa` point.

        If the `ws-wa` point is in the cloud, the corresponding boat speed is
        returned, otherwise the value is interpolated.

        Parameters
        ----------
        ws : scalar
            Wind speed.

        wa : scalar
            Wind angle.

        interpolator : Interpolator, optional
            Interpolator subclass that determines the interpolation
            method used to determine the value at the `ws-wa` point.

            Defaults to `ArithmeticMeanInterpolator(50)`.

        neighbourhood : Neighbourhood, optional
            Neighbourhood subclass used to determine the points in
            the point cloud that will be used in the interpolation.

            Defaults to `Ball(radius=1)`.

        Returns
        -------
        bsp : scalar
            Boat speed value as determined above.
        """
        if np.any((ws <= 0)):
            raise ValueError("`ws` is non-positive")

        wa %= 360

        cloud = self.points
        point = cloud[np.logical_and(cloud[:, 0] == ws, cloud[:, 1] == wa)]
        if point.size:
            return point[2]

        point = np.array([ws, wa])
        weighted_points = WeightedPoints(data=cloud, weights=1)

        considered_points = neighbourhood.is_contained_in(cloud[:, :2] - point)

        return interpolator.interpolate(
            weighted_points[considered_points], point
        )

    @property
    def default_slices(self):
        ws = self.points[:, 0]
        return np.linspace(ws.min(), ws.max(), 20)

    @property
    def wind_speeds(self):
        """Returns all unique wind speeds in the point cloud."""
        return np.array(sorted(list(set(self.points[:, 0]))))

    @property
    def wind_angles(self):
        """Returns all unique wind angles in the point cloud."""
        return np.array(sorted(list(set(self.points[:, 1]))))

    @property
    def boat_speeds(self):
        """Returns all occurring boat speeds in the point cloud
        (including duplicates).
        """
        return self.points[:, 2]

    @property
    def points(self):
        """Returns a read only version of `self._points`."""
        return self._points.copy()

    def to_csv(self, csv_path):
        """Creates a .csv file with delimiter ',' and the
        following format:

            `PolarDiagramPointcloud`
            TWS,TWA,BSP
            `self.points`

        Parameters
        ----------
        csv_path : path-like
            Path to a .csv-file or where a new .csv file will be created.
        """
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file, delimiter=",")
            csv_writer.writerow([self.__class__.__name__])
            csv_writer.writerow(["TWS", "TWA", "BSP"])
            csv_writer.writerows(self.points)

    @classmethod
    def __from_csv__(cls, file):
        csv_reader = csv.reader(file, delimiter=",")
        next(csv_reader)
        points = np.array(
            [[literal_eval(point) for point in row] for row in csv_reader]
        )

        return PolarDiagramPointcloud(points)

    def symmetrize(self):
        """Constructs a symmetric version of the polar diagram,
        by mirroring it at the 0° - 180° axis and returning a new instance.

        Warning
        -------
        Should only be used if all the wind angles of the initial
        polar diagram are on one side of the 0° - 180° axis,
        otherwise this can result in the construction of duplicate points,
        that might overwrite or live alongside old points.
        """
        if not self.points.size:
            return self

        below_180 = [wa for wa in self.wind_angles if wa <= 180]
        above_180 = [wa for wa in self.wind_angles if wa > 180]
        if below_180 and above_180:
            warnings.warn(
                "there are wind angles on both sides of the 0° - 180° axis. "
                "This might result in duplicate data, "
                "which can overwrite or live alongside old data"
            )

        mirrored_points = self.points
        mirrored_points[:, 1] = 360 - mirrored_points[:, 1]
        symmetric_points = np.row_stack((self.points, mirrored_points))

        return PolarDiagramPointcloud(symmetric_points)

    def add_points(self, new_pts, apparent_wind=False):
        """Adds additional points to the point cloud.

        Parameters
        ----------
        new_pts : array_like of shape (n, 3)
            New points to be added to the point cloud given as a sequence
            of points consisting of wind speed, wind angle and
            boat speed.

        apparent_wind : bool, optional
            Specifies if wind data is given in apparent wind.

            If `True`, data will be converted to true wind.

            Defaults to `False`.
        """
        if apparent_wind:
            new_pts = convert_apparent_wind_to_true(new_pts)
        else:
            new_pts = np.asarray_chkfinite(new_pts)
            if np.any((new_pts[:, 0] <= 0)):
                raise ValueError(
                    "`new_pts` has non-positive wind speeds"
                )
            new_pts[:, 1] %= 360

        self._points = np.row_stack((self._points, new_pts))

    def _get_points(self, ws, range_):
        wa = []
        bsp = []
        cloud = self.points
        for w in ws:
            if not isinstance(w, tuple):
                w = (w - range_, w + range_)

            pts = cloud[
                np.logical_and(w[1] >= cloud[:, 0], cloud[:, 0] >= w[0])
            ][:, 1:]
            if not pts.size:
                raise RuntimeError(
                    f"no points with wind speed in range {w} found"
                )

            # sort for wind angles (needed for plotting methods)
            pts = pts[pts[:, 0].argsort()]

            wa.append(np.deg2rad(pts[:, 0]))
            bsp.append(pts[:, 1])

        if not wa:
            raise ValueError(
                "there are no slices in the given range `ws`"
            )

        return wa, bsp
