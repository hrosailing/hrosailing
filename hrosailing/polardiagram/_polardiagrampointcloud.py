# pylint: disable=missing-module-docstring

import csv
import warnings
from ast import literal_eval

import numpy as np

from hrosailing.pipelinecomponents import (
    ArithmeticMeanInterpolator,
    Ball,
    WeightedPoints,
)
from hrosailing.wind import convert_apparent_wind_to_true

from ._basepolardiagram import (
    PolarDiagram,
    PolarDiagramException,
    PolarDiagramInitializationException,
)
from ._plotting import (
    plot3d,
    plot_color_gradient,
    plot_convex_hull,
    plot_flat,
    plot_polar,
)


class PolarDiagramPointcloud(PolarDiagram):
    """A class to represent, visualize and work with a polar diagram
    given by a point cloud

    Parameters
    ----------
    points : array_like of shape (n, 3)
        Initial points of the point cloud, given as a sequence of
        points consisting of wind speed, wind angle and boat speed

    apparent_wind : bool, optional
        Specifies if wind data is given in apparent wind

        If `True`, data will be converted to true wind

        Defaults to `False`
    """

    def __init__(self, points, apparent_wind=False):
        if apparent_wind:
            points = convert_apparent_wind_to_true(points)
        else:
            points = np.asarray_chkfinite(points)
            if np.any((points[:, 0] <= 0)):
                raise PolarDiagramInitializationException(
                    "`points` has nonpositive wind speeds"
                )
            points[:, 1] %= 360

        self._points = points

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

    def __call__(
        self,
        ws,
        wa,
        interpolator=ArithmeticMeanInterpolator(50),
        neighbourhood=Ball(radius=1),
    ):
        """Returns the value of the polar diagram at a given ws-wa point

        If the ws-wa point is in the cloud, the corresponding boat speed is
        returned, otherwise the value is interpolated

        Parameters
        ----------
        ws : scalar
            Wind speed

        wa : scalar
            Wind angle

        interpolator : Interpolator, optional
            Interpolator subclass that determines the interpolation
            method used to determine the value at the ws-wa point

            Defaults to `ArithmeticMeanInterpolator(50)`

        neighbourhood : Neighbourhood, optional
            Neighbourhood subclass used to determine the points in
            the point cloud that will be used in the interpolation

            Defaults to `Ball(radius=1)`

        Returns
        -------
        bsp : scalar
            Boat speed value as determined above
        """
        if np.any((ws <= 0)):
            raise PolarDiagramException("`ws` is nonpositive")

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
    def wind_speeds(self):
        """Returns all unique wind speeds in the point cloud"""
        return np.array(sorted(list(set(self.points[:, 0]))))

    @property
    def wind_angles(self):
        """Returns all unique wind angles in the point cloud"""
        return np.array(sorted(list(set(self.points[:, 1]))))

    @property
    def boat_speeds(self):
        """Returns all occurring boat speeds in the point cloud
        (including duplicates)
        """
        return self.points[:, 2]

    @property
    def points(self):
        """Returns a read only version of `self._points`"""
        return self._points.copy()

    def to_csv(self, csv_path):
        """Creates a .csv file with delimiter ',' and the
        following format:

            PolarDiagramPointcloud
            TWS,TWA,BSP
            `self.points`

        Parameters
        ----------
        csv_path : path-like
            Path to a .csv-file or where a new .csv file will be created
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
            warnings.warn(
                "There are wind angles on both sides of the 0° - 180° axis. "
                "This might result in duplicate data, "
                "which can overwrite or live alongside old data"
            )

        mirrored_points = self.points
        mirrored_points[:, 1] = 360 - mirrored_points[:, 1]
        symmetric_points = np.row_stack((self.points, mirrored_points))

        return PolarDiagramPointcloud(symmetric_points)

    def add_points(self, new_pts, apparent_wind=False):
        """Adds additional points to the point cloud

        Parameters
        ----------
        new_pts: array_like of shape (n, 3)
            New points to be added to the point cloud given as a sequence
            of points consisting of wind speed, wind angle and boat speed

        apparent_wind : bool, optional
            Specifies if wind data is given in apparent_wind

            If `True`, data will be converted to true wind

            Defaults to `False`
        """
        if apparent_wind:
            new_pts = convert_apparent_wind_to_true(new_pts)
        else:
            new_pts = np.asarray_chkfinite(new_pts)
            if np.any((new_pts[:, 0] <= 0)):
                raise PolarDiagramException(
                    "`new_pts` has nonpositive wind speeds"
                )
            new_pts[:, 1] %= 360

        self._points = np.row_stack((self._points, new_pts))

    # TODO Add positivity checks for ws in various cases
    def get_slices(self, ws, stepsize=None, range_=1):
        """For given wind speeds, return the slices of the polar diagram
        corresponding to them

        A slice then consists of all rows in self.wind_speeds whose
        first entry lies in the interval given by w in `ws`

        Parameters
        ----------
        ws : tuple of length 2, iterable , int or float, optional
            Slices of the polar diagram given as either

            - a tuple of length 2 specifying an interval of considered
            wind speeds. The amount of slices taken from that interval are
            determined by the parameter `stepsize`
            - an iterable of tuples of length 2 and int/float values
            which will be interpreted as individual slices. If a w in `ws`
            is an int or float, the given interval will be determined by
            the  parameter `range_`. If it is a tuple, it will be interpreted
            as an inverval as is
            - a single wind speed. The given interval is then determined by
            the parameter `range_`

            If nothing is passed, it will default to
            `(min(self.wind_speeds), max(self.wind_speeds))`

        stepsize : positive int, optional
            Specfies the amount of slices taken from the given
            interval in `ws`

            Will only be used if `ws` is a tuple of length 2

            If nothing is passed it will default to int(round(ws[1] - ws[0]))

        range_ : positive int or float, optional
            Used to convert and int or float `w` in `ws` to the interval
            `(w - range_, w + range_)`

            Will only be used if `ws` is int or float or
            if any `w` in `ws` is an int or float

            Defaults to `1`

        Returns
        -------
        slices : tuple

        Raises
        ------
        PolarDiagramException
            If `stepsize` is nonpositive

            If `range_` is nonpositive
        """
        if ws is None:
            ws = self.wind_speeds

        if isinstance(ws, (int, float)):
            ws = [ws]
        elif isinstance(ws, tuple) and len(ws) == 2:
            if stepsize is None:
                stepsize = int(round(ws[1] - ws[0]))

            if stepsize <= 0:
                raise PolarDiagramException("`stepsize` is nonpositive")

            ws = np.linspace(ws[0], ws[1], stepsize)

        if range_ <= 0:
            raise PolarDiagramException("`range_` is nonpositive")

        wa, bsp = self._get_points(ws, range_)

        ws = [(w[0] + w[1]) / 2 if isinstance(w, tuple) else w for w in ws]
        if len(ws) != len(set(ws)):
            warnings.warn(
                "There are duplicate slices. This might cause "
                "unwanted behaviour"
            )

        return ws, wa, bsp

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
                raise PolarDiagramException(
                    f"No points with wind speed in range {w} found"
                )

            # sort for wind angles (needed for plotting methods)
            pts = pts[pts[:, 0].argsort()]

            wa.append(np.deg2rad(pts[:, 0]))
            bsp.append(pts[:, 1])

        if not wa:
            raise PolarDiagramException(
                "There are no slices in the given range `ws`"
            )

        return wa, bsp

    # pylint: disable=arguments-renamed
    def plot_polar(
        self,
        ws=None,
        stepsize=None,
        range_=1,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Creates a polar plot of one or more slices of the polar diagram

        Parameters
        ----------
        ws : tuple of length 2, iterable , int or float, optional
            Slices of the polar diagram given as either

            - a tuple of length 2 specifying an interval of considered
            wind speeds. The amount of slices taken from that interval are
            determined by the parameter `stepsize`
            - an iterable of tuples of length 2 and int/float values
            which will be interpreted as individual slices. If a `w` in `ws`
            is an int or float, the given interval will be determined by
            the  parameter `range_`. If it is a tuple, it will be interpreted
            as an inverval as is
            - a single wind speed. The given interval is then determined by
            the parameter `range_`

            A slice then consists of all rows in `self.wind_speeds` whose
            first entry lies in the interval given by `w` in `ws`

            If nothing is passed, it will default to
            `(min(self.wind_speeds), max(self.wind_speeds))`

        stepsize : positive int, optional
            Specfies the amount of slices taken from the given
            interval in `ws`

            Will only be used if `ws` is a tuple of length 2

            If nothing is passed it will default to `int(round(ws[1] - ws[0]))`

        range_ : positive scalar, optional
            Used to convert a scalar `w` in `ws` to the interval
            `(w - range_, w + range_)`

            Will only be used if `ws` is scalar or
            if any `w` in `ws` is a scalar

            Defaults to `1`

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be created.

        colors : sequence of color_likes or (ws, color_like) pairs, optional
            Specifies the colors to be used for the different slices

            - If 2 colors are passed, slices will be plotted with a color
            gradient that is determined by the corresponding wind speed
            - Otherwise the slices will be colored in turn with the specified
            colors or the color `"blue"`, if there are too few colors. The
            order is determined by the corresponding wind speeds
            - Alternatively one can specify certain slices to be plotted in
            a color out of order by passing a `(ws, color)` pair

            Defaults to `("green", "red")`

        show_legend : bool, optional
            Specifies wether or not a legend will be shown next to the plot

            The type of legend depends on the color options

            If plotted with a color gradient, a `matplotlib.colorbar.Colorbar`
            will be created, otherwise a `matplotlib.legend.Legend` instance

            Defaults to `False`

        legend_kw : dict, optional
            Keyword arguments to change position and appearence of the legend

            See matplotlib.colorbar.Colorbar and matplotlib.legend.Legend for
            possible keywords and their effects

            Will only be used if show_legend is `True`

        plot_kw : Keyword arguments
            Keyword arguments to change various appearences of the plot

            See matplotlib.axes.Axes.plot for possible keywords and their
            effects

        Raises
        ------
        PolarDiagramException
            If `ws` is given as a single value or a list and there is a
            value `w` in `ws`, such that there are no rows in `self.points`
            whose first entry is equal to `w`
        """
        ws, wa, bsp = self.get_slices(ws, stepsize, range_)
        plot_polar(
            ws,
            wa,
            bsp,
            ax,
            colors,
            show_legend,
            legend_kw,
            _lines=False,
            **plot_kw,
        )

    # pylint: disable=arguments-renamed
    def plot_flat(
        self,
        ws=None,
        stepsize=None,
        range_=1,
        ax=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **plot_kw,
    ):
        """Creates a cartesian plot of one or more slices of the polar diagram

        Parameters
        ----------
        ws : tuple of length 2, iterable , int or float, optional
            Slices of the polar diagram given as either

            - a tuple of length 2 specifying an interval of considered
            wind speeds. The amount of slices taken from that interval are
            determined by the parameter `stepsize`
            - an iterable of tuples of length 2 and int/float values
            which will be interpreted as individual slices. If a `w` in `ws`
            is an int or float, the given interval will be determined by
            the  parameter `range_`. If it is a tuple, it will be interpreted
            as an inverval as is
            - a single wind speed. The given interval is then determined by
            the parameter `range_`

            A slice then consists of all rows in `self.wind_speeds` whose
            first entry lies in the interval given by `w` in `ws`

            If nothing is passed, it will default to
            `(min(self.wind_speeds), max(self.wind_speeds)`

        stepsize : positive int, optional
            Specfies the amount of slices taken from the given
            interval in `ws`

            Will only be used if `ws` is a tuple of length 2

            If nothing is passed it will default to `int(round(ws[1] - ws[0]))`

        range_ : positive scalar, optional
            Used to convert a scalar `w` in `ws` to the interval
            `(w - range_, w + range_)`

            Will only be used if `ws` is scalar or
            if any `w` in `ws` is a scalar

            Defaults to `1`

        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

        colors : sequence of color_likes or (ws, color_like) pairs, optional
            Specifies the colors to be used for the different slices

            - If 2 colors are passed, slices will be plotted with a color
            gradient that is determined by the corresponding wind speed
            - Otherwise the slices will be colored in turn with the specified
            colors or the color `"blue"`, if there are too few colors. The
            order is determined by the corresponding wind speeds
            - Alternatively one can specify certain slices to be plotted in
            a color out of order by passing a `(ws, color)` pair

            Defaults to `("green", "red")`

        show_legend : bool, optional
            Specifies wether or not a legend will be shown next to the plot

            The type of legend depends on the color options

            If plotted with a color gradient, a `matplotlib.colorbar.Colorbar`
            will be created, otherwise a `matplotlib.legend.Legend` instance

            Defaults to `False`

        legend_kw : dict, optional
            Keyword arguments to change position and appearence of the legend

            See matplotlib.colorbar.Colorbar and matplotlib.legend.Legend for
            possible keywords and their effects

            Will only be used if show_legend is `True`

        plot_kw : Keyword arguments
            Keyword arguments to change various appearences of the plot

            See matplotlib.axes.Axes.plot for possible keywords and their
            effects

        Raises
        ------
        PolarDiagramException
            If `ws` is given as a single value or a list and there is a
            value `w` in `ws`, such that there are no rows in `self.points`
            whose first entry is equal to `w`
        """
        ws, wa, bsp = self.get_slices(ws, stepsize, range_)
        wa = [np.rad2deg(a) for a in wa]
        plot_flat(
            ws,
            wa,
            bsp,
            ax,
            colors,
            show_legend,
            legend_kw,
            _lines=False,
            **plot_kw,
        )

    def plot_3d(self, ax=None, colors=("green", "red"), **plot_kw):
        """Creates a 3d plot of the polar diagram

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            Axes instance where the plot will be created.

        colors: tuple of two (2) color_likes, optional
            Color pair determining the color gradient with which the
            polar diagram will be plotted

            Will be determined by the corresponding wind speeds

            Defaults to `("green", "red")`

        plot_kw : Keyword arguments
            Keyword arguments to change various appearences of the plot

            See matplotlib.axes.Axes.plot for possible keywords and their
            effects

        Raises
        ------
        PolarDiagramException
            If there are no points in the point cloud
        """
        if not self.points.size:
            raise PolarDiagramException(
                "Can't create 3d plot of empty point cloud"
            )

        ws, wa, bsp = (self.points[:, 0], self.points[:, 1], self.points[:, 2])

        wa = np.deg2rad(wa)
        bsp, wa = bsp * np.cos(wa), bsp * np.sin(wa)
        plot3d(ws, wa, bsp, ax, colors, **plot_kw)

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
        of the polar diagram with respect to the corresponding boat speeds

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes instance where the plot will be created.

        colors : tuple of two (2) color_likes, optional
            Color pair determining the color gradient with which the
            polar diagram will be plotted

            Will be determined by the corresponding boat speed

            Defaults to `("green", "red")`

        marker : matplotlib.markers.Markerstyle or equivalent, optional
            Markerstyle for the created scatter plot

            Defaults to `"o"`

        ms : float or array_like of fitting shape, optional
            Marker size in points**2

        show_legend : bool, optional
            Specifies wether or not a legend will be shown next
            to the plot

            Legend will be a `matplotlib.colorbar.Colorbar` instance

            Defaults to `False`

        legend_kw : Keyword arguments
            Keyword arguments to change position and appearence of the legend

            See matplotlib.legend.Legend for possible keywords and
            their effects

            Will only be used if show_legend is `True`

        Raises
        ------
        PolarDiagramException
            If there are no points in the point cloud
        """
        if not self.points.size:
            raise PolarDiagramException(
                "Can't create color gradient plot of empty point cloud"
            )

        ws, wa, bsp = (self.points[:, 0], self.points[:, 1], self.points[:, 2])

        plot_color_gradient(
            ws, wa, bsp, ax, colors, marker, ms, show_legend, **legend_kw
        )

    # pylint: disable=arguments-renamed
    def plot_convex_hull(
        self,
        ws=None,
        stepsize=None,
        range_=1,
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
        ws : tuple of length 2, iterable , int or float, optional
            Slices of the polar diagram given as either

            - a tuple of length 2 specifying an interval of considered
            wind speeds. The amount of slices taken from that interval are
            determined by the parameter `stepsize`
            - an iterable of tuples of length 2 and int/float values
            which will be interpreted as individual slices. If a `w` in `ws`
            is an int or float, the given interval will be determined by
            the  parameter `range_`. If it is a tuple, it will be interpreted
            as an inverval as is
            - a single wind speed. The given interval is then determined by
            the parameter `range_`

            A slice then consists of all rows in `self.wind_speeds` whose
            first entry lies in the interval given by `w` in `ws`

            If nothing is passed, it will default to
            `(min(self.wind_speeds), max(self.wind_speeds)`

        stepsize : positive int, optional
            Specfies the amount of slices taken from the given
            interval in `ws`

            Will only be used if `ws` is a tuple of length 2

            Defaults to `int(round(ws[1] - ws[0]))`

        range_ : positive int or float, optional

            Will only be used if `ws` is int or float or
            if any `w` in `ws` is an int or float

            Defaults to `1`

        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes instance where the plot will be create

        colors : sequence of color_likes or (ws, color_like) pairs, optional
            Specifies the colors to be used for the different slices

            - If 2 colors are passed, slices will be plotted with a color
            gradient that is determined by the corresponding wind speed
            - Otherwise the slices will be colored in turn with the specified
            colors or the color `"blue"`, if there are too few colors. The
            order is determined by the corresponding wind speeds
            - Alternatively one can specify certain slices to be plotted in
            a color out of order by passing a `(ws, color)` pair

            Defaults to `("green", "red")`

        show_legend : bool, optional
            Specifies wether or not a legend will be shown next to the plot

            The type of legend depends on the color options

            If plotted with a color gradient, a `matplotlib.colorbar.Colorbar`
            will be created, otherwise a `matplotlib.legend.Legend`

            Defaults to `False`

        legend_kw : dict, optional
            Keyword arguments to change position and appearence of the legend

            See matplotlib.colorbar.Colorbar and matplotlib.legend.Legend for
            possible keywords and their effects

            Will only be used if show_legend is `True`

        plot_kw : Keyword arguments
            Keyword arguments to change various appearences of the plot

            See matplotlib.axes.Axes.plot for possible keywords and their
            effects

        Raises
        ------
        PolarDiagramException
            If `ws` is given as a single value or a list and there is a
            value `w` in `ws`, such that there are no rows in `self.points`
            whose first entry is equal to `w`
        """
        ws, wa, bsp = self.get_slices(ws, stepsize, range_)

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
