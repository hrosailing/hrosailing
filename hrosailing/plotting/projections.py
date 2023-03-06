"""
Contains projections and functions for plotting objects of the hrosailing framework.
Currently, the plot of `PolarDiagram` objects is supported.
Defines the following projections:

- `"hro polar"` : plot and scatter polar diagrams in a polar plot (see `HROPolar`),
- `"hro flat"` : plot and scatter polar diagrams in a euclidean plot (see `HROFlat`),
- `"hro color gradient"` : plot two-dimensional heat maps of a polar diagram (see `HROColorGradient`),
- `"hro 3d"` : scatter or plot the surface of the three-dimensional representation of a polar diagram (see `Axes3D`).

Examples
--------
>>> import matplotlib.pyplot as plt
>>> from hrosailing.polardiagram import from_csv
>>> import hrosailing.plotting
>>>
>>> ax = plt.subplot("hro polar")
>>> pd = from_csv("my_file.pd")
>>> ax.plot(pd)
>>> plt.show()
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
    is_color_like,
    to_rgb,
)
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D as pltAxes3D
from scipy.spatial import ConvexHull

from hrosailing.polardiagram import PolarDiagram


class HROPolar(PolarAxes):
    """
    Projection for plotting polar diagrams in a polar plot.
    """

    name = "hro polar"

    def plot(
        self,
        *args,
        ws=None,
        n_steps=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        use_convex_hull=False,
        **kwargs,
    ):
        """
        Plots the given data as a polar plot.
        If a `PolarDiagram` is given, plots each slice corresponding to `ws` and `n_steps`
        as described in `PolarDiagram.get_slices`.

        Parameters
        ----------
        *args :
            If the first argument is a polar diagram it plots the polar diagram.
            Otherwise, it plots the arguments as usual.

        ws : int, float or iterable thereof, optional
            As in `Polardiagram.get_slices`.

        n_steps : int, optional
            As in `Polardiagram.get_slices`.

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

        show_legend : bool
            Specifies whether a legend will be shown next to the plot.

            The type of legend depends on the color options.

            If plotted with a color gradient, a `matplotlib.colorbar.Colorbar`
            will be created, otherwise a `matplotlib.legend.Legend` instance.

            Defaults to `False`.

        legend_kw : dict, optional
            Keyword arguments to change position and appearance of the colorbar
            or legend respectively.
            - If 2 colors are passed, a colorbar will be created.
            In this case see `matplotlib.colorbar.Colorbar` for possible
            keywords and their effect.
            - Otherwise, a legend will be created.
            In this case see `matplotlib.legend.Legend` for possible keywords
            and their effect.

            Defaults to `None`

        use_convex_hull : bool, optional
            If set to `True`, the convex hull (in polar coordinates) of the slices will be plotted instead of the
            slices itself.
        """
        if not isinstance(args[0], PolarDiagram):
            super().plot(*args, **kwargs)
            return

        labels, slices, info = self._prepare_plot(
            args, ws, n_steps, colors, show_legend, legend_kw, **kwargs
        )
        _plot(self, slices, info, True, use_convex_hull, **kwargs)

    def scatter(
        self,
        *args,
        ws=None,
        n_steps=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        use_convex_hull=False,
        **kwargs,
    ):
        """
        Plots the given data as a polar plot.
        Parameters are identical to `plot`.

        See also
        -------
        `HROPolar.plot`
        """
        if not isinstance(args[0], PolarDiagram):
            super().scatter(*args, **kwargs)
            return

        labels, slices, info = self._prepare_plot(
            args, ws, n_steps, colors, show_legend, legend_kw, **kwargs
        )
        _plot(self, slices, info, True, use_convex_hull, True, **kwargs)

    def _prepare_plot(
        self, args, ws, n_steps, colors, show_legend, legend_kw, **kwargs
    ):
        pd = args[0]
        labels, slices, info = pd.get_slices(ws, n_steps, full_info=True)
        _set_polar_axis(self)
        _configure_axes(self, labels, colors, show_legend, legend_kw, **kwargs)
        return labels, slices, info


class HROFlat(Axes):
    """
    Projection to plot given data in a rectilinear plot.
    """

    name = "hro flat"

    def plot(
        self,
        *args,
        ws=None,
        colors=("green", "red"),
        show_legend=False,
        use_convex_hull=False,
        legend_kw=None,
        **kwargs,
    ):
        """
        Plots the given data in a rectilinear plot.
        Otherwise, it works identical to `HROPolar.plot`.

        See also
        ----------
        `HROPolar.plot`
        """
        if not isinstance(args[0], PolarDiagram):
            super().plot(*args, **kwargs)
            return

        pd = args[0]
        labels, slices, info = pd.get_slices(ws, full_info=True)
        _configure_axes(self, labels, colors, show_legend, legend_kw, **kwargs)
        _plot(self, slices, info, False, use_convex_hull, **kwargs)

    def scatter(
        self,
        *args,
        ws=None,
        colors=("green", "red"),
        show_legend=False,
        use_convex_hull=False,
        legend_kw=None,
        **kwargs,
    ):
        """
        Creates rectilinear scatter plot of a `PolarDiagram` instance.
        Otherwise, it works the same as `HROPolar.plot`

        See also:
        ----------
        `HROPolar.plot`
        """

        if not isinstance(args[0], PolarDiagram):
            super().scatter(*args, **kwargs)
            return

        pd = args[0]
        labels, slices, info = pd.get_slices(ws, full_info=True)
        _configure_axes(self, labels, colors, show_legend, legend_kw, **kwargs)
        _plot(self, slices, info, False, use_convex_hull, True, **kwargs)


class HROColorGradient(Axes):
    """Projection supporting two-dimensional color gradient plotting of polar diagrams."""

    name = "hro color gradient"

    def plot(self, *args, **kwargs):
        """
        Works identical to `scatter`.

        See also
        ----------
        `HROColorGradient.scatter`
        """
        self.scatter(*args, **kwargs)

    def scatter(
        self,
        *args,
        wind=None,
        colors=("green", "red"),
        show_legend=False,
        legend_kw=None,
        **kwargs,
    ):
        """
        Plots the given data as a polar plot.
        If a `PolarDiagram` is given, plots each slice corresponding to `ws` and `n_steps`
        as described in `PolarDiagram.get_slices`.

        Parameters
        ----------
        *args :
            If the first argument is a polar diagram it plots the polar diagram.
            Otherwise, it plots the arguments as usual.

        wind : (2, d) or (d, 2) numpy.ndarray or tuple of 2 (1, d) numpy.ndarray or `None`, optional
            As in `Polardiagram.get_points`.

        colors : color_like or sequence of color_likes or (ws, color_like) pairs
            Specifies the colors to be used for the different slices.
            As in `HROPolar.plot`.

        show_legend : bool, default: `False`
            Specifies whether a legend will be shown next to the plot.
            As in `HROPolar.plot`.

        legend_kw : dict, optional
            Keyword arguments to change position and appearance of the colorbar
            or legend respectively.
            As in `HROPolar.plot`.
        """
        if not isinstance(args[0], PolarDiagram):
            super().scatter(*args, **kwargs)
            return

        if legend_kw is None:
            legend_kw = {}

        pd = args[0]
        points = pd.get_points(wind=wind)
        ws, wa, bsp = points.T

        if show_legend:
            _configure_legend(self, bsp, colors, "Boat Speed", **legend_kw)

        color_gradient = _determine_color_gradient(colors, bsp.ravel())

        self.scatter(ws, wa, c=color_gradient, **legend_kw)


class Axes3D(pltAxes3D):
    """Projection enabling the display of polar diagrams in a three-dimensional plot."""

    name = "hro 3d"

    def scatter(
        self,
        *args,
        wind=None,
        colors=("green", "red"),
        **kwargs,
    ):
        """
        Works identical to `HROColorGradient.scatter`.

        See also
        ---------
        `HROColorGradient.scatter`
        """
        if not isinstance(args[0], PolarDiagram):
            super().scatter(*args, **kwargs)
            return

        pd = args[0]
        x, y, z = self._prepare_points(pd, wind)

        self._plot3d(x, y, z, colors, **kwargs)

    def plot(self, *args, **kwargs):
        """
        Works identical to `plot_surface` on polar diagrams.
        Otherwise, it works like `mpl_toolkits.mplot3d.Axes3D.plot`.

        See also
        --------
        `Axes3D.plot_surface`
        `mpl_toolkits.mplot3d.Axes3D.plot`
        """
        if not isinstance(args[0], PolarDiagram):
            super().plot(*args, **kwargs)
            return

        self.plot_surface(*args, **kwargs)

    def plot_surface(
        self, *args, wind=None, colors=("green", "red"), **kwargs
    ):
        """
        Plots a three-dimensional depiction of the polar diagram as a triangulated surface plot if a
        polar diagram is given. Otherwise, it works as `mpl_toolkits.mplot3d.plot_surface`.
        Parameters are the same as in `HROColorGradient.scatter`.

        See also
        --------
        `HROColorGradient.scatter`
        `mpl_toolkits.mplot3d.Axes3D.plot_surface`
        """
        if not isinstance(args[0], PolarDiagram):
            super().plot_surface(*args, **kwargs)
            return

        pd = args[0]
        x, y, z = self._prepare_points(pd, wind)

        triang = Triangulation(x, y)

        txs = x[triang.triangles]
        tys = y[triang.triangles]
        tzs = z[triang.triangles]

        diffz = [
            (tzs[:, idx1] - tzs[:, idx2]) ** 2
            for idx1, idx2 in itertools.combinations([0, 1, 2], 2)
        ]

        not_too_narrow = np.logical_or(
            diffz[0] > 0.2, diffz[1] > 0.2, diffz[2] > 0.2
        )
        axis_skip = np.logical_or(
            np.sign(txs[:, 0]) != np.sign(txs[:, 1]),
            np.sign(txs[:, 0]) != np.sign(txs[:, 2]),
        )
        northern = np.logical_and(*[tys[:, i] > 0 for i in range(2)])
        no_northern_skip = np.logical_not(np.logical_and(axis_skip, northern))
        mask = np.logical_and(not_too_narrow, no_northern_skip)

        color_map = _create_color_map(colors)

        _set_3d_axis_labels(self)
        _remove_3d_tick_labels_for_polar_coordinates(self)

        self.plot_trisurf(
            x, y, z, triangles=triang.triangles[mask], cmap=color_map, **kwargs
        )

    def _prepare_points(self, pd, wind):
        points = pd.get_points(wind)
        ws, wa, bsp = points.T
        # flip and rotate such that 0° is on top and 90° is right
        # wa = (-wa)%360 - 90
        wa_rad = np.deg2rad(wa)

        x, y = bsp * np.sin(wa_rad), bsp * np.cos(wa_rad)
        return x, y, ws

    def _plot3d(self, x, y, z, colors, **plot_kw):
        _set_3d_axis_labels(self)
        _remove_3d_tick_labels_for_polar_coordinates(self)

        color_map = _create_color_map(colors)

        super().scatter(x, y, z, c=z, cmap=color_map, **plot_kw)


register_projection(HROPolar)
register_projection(HROFlat)
register_projection(HROColorGradient)
register_projection(Axes3D)


def _plot(
    ax,
    slices,
    info,
    use_radians,
    use_convex_hull=False,
    use_scatter=False,
    **kwargs,
):
    def safe_zip(iter1, iter2):
        if iter2 is not None:
            yield from zip(iter1, iter2)
            return
        for entry in iter1:
            yield (entry, None)

    for slice_, info_ in safe_zip(slices, info):
        slice_ = slice_[:, np.argsort(slice_[1])]
        if use_convex_hull:
            ws, wa, bsp, info_ = _get_convex_hull(slice_, info_)
        else:
            ws, wa, bsp = slice_
        if use_radians:
            wa = np.deg2rad(wa)
        if info_ is not None and not use_convex_hull:
            wa, bsp = _alter_with_info(wa, bsp, info_)
        if use_scatter:
            ax.scatter(wa, bsp, **kwargs)
        else:
            ax.plot(wa, bsp, **kwargs)


def _get_info_intervals(info_):
    intervals = {}
    for j, entry in enumerate(info_):
        if entry not in intervals:
            intervals[entry] = []
        intervals[entry].append(j)
    return intervals.values()


def _merge(wa, intervals):
    wa_in_intervals = [
        np.concatenate([wa[interval], [np.NAN]]) for interval in intervals
    ]
    return np.concatenate(wa_in_intervals)[:-1]


def _alter_with_info(wa, bsp, info_):
    intervals = _get_info_intervals(info_)
    wa = _merge(wa, intervals)
    bsp = _merge(bsp, intervals)
    return wa, bsp


def _get_convex_hull(slice_, info_):
    ws, wa, bsp = slice_
    wa_rad = np.deg2rad(wa)
    points = np.column_stack([bsp * np.cos(wa_rad), bsp * np.sin(wa_rad)])
    try:
        vertices = ConvexHull(points).vertices
    except ValueError:
        return ws, wa, bsp, info_
    slice_ = slice_.T[vertices]
    if info_ is not None:
        info_ = [entry for i, entry in enumerate(info_) if i in vertices]
        slice_, info = zip(*sorted(zip(slice_, info_), key=lambda x: x[0][1]))
    else:
        slice_ = sorted(slice_, key=lambda x: x[1])
    slice_ = np.array(slice_).T

    if (slice_[1, 0] == 0 and slice_[1, -1] == 360):
        ws, wa, bsp = slice_
        return ws, wa, bsp, info_

    if slice_[1][-1] - slice_[1][0] < 180:
        ws, wa, bsp = slice_
        return ws, wa, bsp, info_

    if slice_[1, 0] == 0:
        slice_ = np.column_stack([
            slice_, [slice_[0, 0], 360, slice_[2, 0]]
        ])
        if info_ is not None:
            info_ = info_ + [info_[0]]
        ws, wa, bsp = slice_
        return ws, wa, bsp, info_

    if slice_[1, -1] == 360:
        slice_ = np.column_stack([
            [slice_[0, -1], 0, slice_[2, -1]], slice_
        ])
        if info_ is not None:
            info_ = info_ + [info_[-1]]
        ws, wa, bsp = slice_
        return ws, wa, bsp, info_

    # if wind angle difference is big, wrap around
        # estimate bsp value at 0 (360 resp)

    x0 = slice_[2, 0] * np.sin(np.deg2rad(slice_[1, 0]))
    x1 = slice_[2, -1] * np.sin(np.deg2rad(slice_[1, -1]))
    y0 = slice_[2, 0] * np.cos(np.deg2rad(slice_[1, 0]))
    y1 = slice_[2, -1] * np.cos(np.deg2rad(slice_[1, -1]))
    lamb = x0 / (x0 - x1)
    approx_ws = lamb * slice_[0, 0] + (1 - lamb) * slice_[0, -1]
    approx_bsp = lamb * y0 + (1 - lamb) * y1

    slice_ = np.column_stack(
        [[approx_ws, 0, approx_bsp], slice_, [approx_ws, 360, approx_bsp]]
        )
    if info_ is not None:
        info_ = [info_[0]] + info_ + info_[-1]

    ws, wa, bsp = slice_

    return ws, wa, bsp, info_


def plot_polar(*args, **kwargs):
    """
    Creates a single `HROPolar` Axes and calls its `plot` method.
    Useful for simple plots of just one polar diagram.

    See also
    ----------
    `HROPolar.plot`
    """
    ax = plt.subplot(projection="hro polar")
    ax.plot(*args, **kwargs)


def scatter_polar(*args, **kwargs):
    """
    Creates a single `HROPolar` Axes and calls its `scatter` method.
    Useful for simple plots of just one polar diagram.

    See also
    ----------
    `HROPolar.scatter`
    """
    ax = plt.subplot(projection="hro polar")
    ax.scatter(*args, **kwargs)


def plot_flat(*args, **kwargs):
    """
    Creates a single `HROFlat` Axes and calls its `plot` method.
    Useful for simple plots of just one polar diagram.

    See also
    ----------
    `HROFlat.plot`
    """
    ax = plt.subplot(projection="hro flat")
    ax.plot(*args, **kwargs)


def scatter_flat(*args, **kwargs):
    """
    Creates a single `HROFlat` Axes and calls its `scatter` method.
    Useful for simple plots of just one polar diagram.

    See also
    ----------
    `HROFlat.scatter`
    """
    ax = plt.subplot(projection="hro flat")
    ax.scatter(*args, **kwargs)


def plot_color_gradient(*args, **kwargs):
    """
    Creates a single `HROColorGradient` Axes and calls its `plot` method.
    Useful for simple plots of just one polar diagram.

    See also
    ----------
    `HROColorGradient.plot`
    """
    ax = plt.subplot(projection="hro color gradient")
    ax.plot(*args, **kwargs)


def plot_3d(*args, **kwargs):
    """
    Creates a single `Axes3D` Axes and calls its `plot` method.
    Useful for simple plots of just one polar diagram.

    See also
    ----------
    `Axes3D.plot`
    """
    ax = plt.subplot(projection="hro 3d")
    ax.plot(*args, **kwargs)


def scatter_3d(*args, **kwargs):
    """
    Creates a single `Axes3D` Axes and calls its `scatter` method.
    Useful for simple plots of just one polar diagram.

    See also
    ----------
    `Axes3D.scatter`
    """
    ax = plt.subplot(projection="hro 3d")
    ax.scatter(*args, **kwargs)


def _configure_axes(ax, labels, colors, show_legend, legend_kw, **kwargs):
    _configure_colors(ax, labels, colors)
    _check_plot_kw(kwargs, True)
    if show_legend:
        if legend_kw is None:
            legend_kw = {}
        _configure_legend(ax, labels, colors, "True Wind Speed", **legend_kw)


def _set_polar_axis(ax):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")


def _check_plot_kw(plot_kw, lines=True):
    ls = plot_kw.pop("linestyle", None) or plot_kw.pop("ls", None)
    if ls is None:
        plot_kw["ls"] = "-" if lines else ""
    else:
        plot_kw["ls"] = ls

    if plot_kw.get("marker", None) is None and not lines:
        plot_kw["marker"] = "o"


def _configure_colors(ax, ws, colors):
    if _only_one_color(colors):
        ax.set_prop_cycle("color", [colors])
        return

    if _more_colors_than_plots(ws, colors) or _no_color_gradient(colors):
        _set_color_cycle(ax, ws, colors)
        return

    _set_color_gradient(ax, ws, colors)


def _only_one_color(colors):
    return is_color_like(colors)


def _more_colors_than_plots(ws, colors):
    return len(ws) <= len(colors)


def _no_color_gradient(colors):
    all_color_format = all(_has_color_format(c) for c in colors)
    return len(colors) != 2 or not all_color_format


def _has_color_format(obj):
    if isinstance(obj, str):
        return True
    if len(obj) in [3, 4]:
        return True
    return False


def _set_color_cycle(ax, ws, colors):
    color_cycle = ["blue"] * len(ws)
    _configure_color_cycle(color_cycle, colors, ws)

    ax.set_prop_cycle("color", color_cycle)


def _configure_color_cycle(color_cycle, colors, ws):
    if isinstance(colors[0], tuple):
        for w, color in colors:
            i = list(ws).index(w)
            color_cycle[i] = color

        return

    colors = itertools.islice(colors, len(color_cycle))

    for i, color in enumerate(colors):
        color_cycle[i] = color


def _set_color_gradient(ax, ws, colors):
    color_gradient = _determine_color_gradient(colors, ws)
    ax.set_prop_cycle("color", color_gradient)


def _determine_color_gradient(colors, gradient):
    gradient_coeffs = _get_gradient_coefficients(gradient)
    color_gradient = _determine_colors_from_coefficients(
        gradient_coeffs, colors
    )
    return color_gradient


def _get_gradient_coefficients(gradient):
    min_gradient = gradient.min()
    max_gradient = gradient.max()

    return [
        (grad - min_gradient) / (max_gradient - min_gradient)
        for grad in gradient
    ]


def _determine_colors_from_coefficients(coefficients, colors):
    min_color = np.array(to_rgb(colors[0]))
    max_color = np.array(to_rgb(colors[1]))

    return [
        (1 - coeff) * min_color + coeff * max_color for coeff in coefficients
    ]


def _configure_legend(ax, ws, colors, label, **legend_kw):
    if _plot_with_color_gradient(ws, colors):
        _set_colormap(ws, colors, ax, label, **legend_kw)
        return

    if isinstance(colors[0], tuple) and not is_color_like(colors[0]):
        _set_legend_without_wind_speeds(ax, colors, legend_kw)
        return

    _set_legend_with_wind_speeds(ax, colors, ws, legend_kw)


def _plot_with_color_gradient(ws, colors):
    return not _no_color_gradient(colors) and len(ws) > len(colors) == 2


def _set_colormap(ws, colors, ax, label, **legend_kw):
    color_map = _create_color_map(colors)

    label_kw, legend_kw = _extract_possible_text_kw(legend_kw)
    plt.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=min(ws), vmax=max(ws)), cmap=color_map
        ),
        ax=ax,
        **legend_kw,
    ).set_label(label, **label_kw)


def _extract_possible_text_kw(legend_kw):
    return {}, legend_kw


def _set_legend_without_wind_speeds(ax, colors, legend_kw):
    ax.legend(
        handles=[
            Line2D([0], [0], color=color, lw=1, label=f"TWS {ws}")
            for (ws, color) in colors
        ],
        **legend_kw,
    )


def _set_legend_with_wind_speeds(ax, colors, ws, legend_kw):
    slices = zip(ws, colors)

    ax.legend(
        handles=[
            Line2D([0], [0], color=color, lw=1, label=f"TWS {ws}")
            for (ws, color) in slices
        ],
        **legend_kw,
    )


def _set_3d_axis_labels(ax):
    ax.set_zlabel("TWS")
    ax.set_xlabel("Polar plane: TWA / BSP ")


def _remove_3d_tick_labels_for_polar_coordinates(ax):
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])


def _create_color_map(colors):
    return LinearSegmentedColormap.from_list("cmap", list(colors))
