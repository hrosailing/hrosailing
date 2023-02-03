"""
Contains projections and functions for plotting objects of the hrosailing framework.
Currently the plot of `PolarDiagram` objects is supported.
Defines the following projections:

- 'hro polar' : plot polar diagrams in a polar plot (see `HROPolar.plot`)
- 'hro flat' : plot polar diagrams in an euclidean plot (see `HROFlat.plot`)

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
from matplotlib.projections.polar import PolarAxes
from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
    is_color_like,
    to_rgb,
)
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

from hrosailing.polardiagram import PolarDiagram


class HROPolar(PolarAxes):
    """
    Projection for plotting polar diagrams in a polar plot.
    """
    name = "hro polar"

    def plot(self,
             *args,
             ws=None,
             n_steps=None,
             colors=("green", "red"),
             show_legend=False,
             legend_kw=None,
             use_convex_hull=False,
             **kwargs
             ):
        """
        Plots the given data as a polar plot.
        If a `PolarDiagram` is given, plots each slice corresponding to `ws` and `n_steps`
        as described in `PolarDiagram.get_slices`.

        Parameter
        ----------
        *args :
            If the first argument is a polar diagram it plots the polar diagram.
            Otherwise it plots the arguments as usual.

        ws : int, float or iterable thereof, optional
            As in `Polardiagram.get_slices`

        n_steps : int, optional
            As in `Polardiagram.get_slices`

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

            Defaults to `None`

        use_convex_hull : bool, optional
            If set to `True`, the convex hull (in polar coordinates) of the slices will be plotted instead of the
            slices itself.
        """
        if not isinstance(args[0], PolarDiagram):
            super().plot(*args, **kwargs)
            return

        pd = args[0]
        labels, slices, info = pd.get_slices(ws, n_steps, full_info=True)
        _set_polar_axis(self)
        _configure_axes(self, labels, colors, show_legend, legend_kw, **kwargs)
        _plot(self, slices, info, True, use_convex_hull, **kwargs)


class HROFlat(Axes):
    """
    Projection to plot given data in a rectilinear plot.
    API works identical to `HROPolar`.
    """
    name = "hro flat"

    def plot(self,
             *args,
             ws=None,
             colors=("green", "red"),
             show_legend=False,
             use_convex_hull=False,
             legend_kw=None,
             **kwargs
             ):
        """
        Plots the given data in a rectilinear plot.
        Otherwise it works identical to `HROPolar.plot`.

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


class HROColorGradient(Axes):
    """Projection supporting two dimensional color gradient plotting of polar diagrams."""
    name = "hro color gradient"

    def plot(self,
             *args,
             wind = None,
             colors = ("green", "red"),
             show_legend=False,
             legend_kw=None,
             **kwargs
             ):
        """
        Plots the given data as a polar plot.
        If a `PolarDiagram` is given, plots each slice corresponding to `ws` and `n_steps`
        as described in `PolarDiagram.get_slices`.

        Parameter
        ----------
        *args :
            If the first argument is a polar diagram it plots the polar diagram.
            Otherwise it plots the arguments as usual.

        wind : (2, d) or (d, 2) numpy.ndarray or tuple of 2 (1, d) numpy.ndarray or `None`, optional
            As in `Polardiagram.get_points`

        colors : color_like or sequence of color_likes or (ws, color_like) pairs
            Specifies the colors to be used for the different slices.
            As in `HROPolar.plot`.

        show_legend : bool, default: `False`
            Specifies wether or not a legend will be shown next to the plot.
            As in `HROPolar.plot`.

        legend_kw : dict, optional
            Keyword arguments to change position and appearance of the colorbar
            or legend respectively.
            As in `HROPolar.plot`.
        """
        if not isinstance(args[0], PolarDiagram):
            super().plot(*args, **kwargs)
            return

        if legend_kw is None:
            legend_kw = {}

        pd = args[0]
        points = pd.get_points()
        ws, wa, bsp = points.T

        if show_legend:
            _show_legend(self, bsp, colors, "Boat Speed", legend_kw)

        color_gradient = _determine_color_gradient(colors, bsp.ravel())

        self.scatter(ws, wa, c=color_gradient, **legend_kw)


class HRO3D(Axes3D):
    name = "hro 3d"

    def plot(self, *args, colors=("green", "red"), **kwargs):
        if not isinstance(args[0], PolarDiagram):
            super().plot(*args, **kwargs)
            return

        pd = args[0]
        ws, wa, bsp = pd.get_slices()
        lines_ = _check_for_lines(wa)
        if wa.ndim == 1 and ws.ndim == 1:
            wa, ws = np.meshgrid(np.flip(wa), ws)
            bsp = np.flip(bsp, axis=1)
        elif ws.ndim == 1:
            ws = np.array(
                [[ws_ for _ in range(len(wa_))] for ws_, wa_ in
                 zip(ws, wa)]
            )

        y, z = np.cos(np.pi/2-wa) * bsp, np.sin(np.pi/2-wa) * bsp

        if lines_:
            self._plot_surf(ws, y, z, colors, **kwargs)
            return

        self._plot3d(ws, y, z, colors, **kwargs)

    def _plot3d(self, ws, wa, bsp, colors, **plot_kw):
        _set_3d_axis_labels(self)
        _remove_3d_tick_labels_for_polar_coordinates(self)

        color_map = _create_color_map(colors)

        super().scatter(ws, wa, bsp, c=ws, cmap=color_map, **plot_kw)

    def _plot_surf(self, ws, wa, bsp, colors, **plot_kw):
        _set_3d_axis_labels(self)
        _remove_3d_tick_labels_for_polar_coordinates(self)

        color_map = _create_color_map(colors)
        face_colors = _determine_face_colors(color_map, ws)

        super().plot_surface(
            ws, wa, bsp, facecolors=face_colors
        )



register_projection(HROPolar)
register_projection(HROFlat)
register_projection(HROColorGradient)
register_projection(HRO3D)


def _plot(ax, slices, info, use_radians, use_convex_hull, **kwargs):
    def safe_zip(iter1, iter2):
        if iter2 is not None:
            yield from zip(iter1, iter2)
            return
        for entry in iter1:
            yield (entry, None)

    for slice, info_ in safe_zip(slices, info):
        if use_convex_hull:
            ws, wa, bsp, info_ = _get_convex_hull(slice, info_)
        else:
            ws, wa, bsp = slice
        if use_radians:
            wa = np.deg2rad(wa)
        if info_ is not None and not use_convex_hull:
            wa, bsp = _alter_with_info(wa, bsp, info_)
        ax.plot(wa, bsp, **kwargs)


def _get_info_intervals(info_):
    intervals = {}
    for j, entry in enumerate(info_):
        if entry not in intervals:
            intervals[entry] = []
        intervals[entry].append(j)
    return intervals.values()


def _merge(wa, intervals):
    wa_in_intervals = [np.concatenate([wa[interval], [np.NAN]]) for interval in intervals]
    return np.concatenate(wa_in_intervals)[:-1]

def _alter_with_info(wa, bsp, info_):
    intervals = _get_info_intervals(info_)
    wa = _merge(wa, intervals)
    bsp = _merge(bsp, intervals)
    return wa, bsp

def _get_convex_hull(slice, info_):
    ws, wa, bsp = slice
    wa_rad = np.deg2rad(wa)
    points = np.column_stack([
            bsp*np.cos(wa_rad), bsp*np.sin(wa_rad)
    ])
    try:
        vertices = ConvexHull(points).vertices
    except ValueError:
        return ws, wa, bsp, info_
    slice = slice.T[vertices]
    if info_ is not None:
        info_ = [entry for i, entry in enumerate(info_) if i in vertices]
        slice, info = zip(*sorted(zip(slice, info_), key=lambda x: x[0][1]))
    else:
        slice = sorted(slice, key=lambda x: x[1])
    slice = np.array(slice).T

    #if wind angle difference is big, wrap around
    if slice[1][-1] - slice[1][0] > 180:
        #estimate bsp value at 0 (360 resp)
        x0 = slice[2, 0]*np.sin(np.deg2rad(slice[1, 0]))
        x1 = slice[2, -1]*np.sin(np.deg2rad(slice[1, -1]))
        y0 = slice[2, 1]*np.cos(np.deg2rad(slice[1, 0]))
        y1 = slice[2, -1]*np.cos(np.deg2rad(slice[1, -1]))
        lamb = x0/(x0 - x1)
        approx_ws = lamb*slice[0, 0] + (1-lamb)*slice[0, -1]
        approx_bsp = lamb*y0 + (1-lamb)*y1

        slice = np.column_stack([[approx_ws, 0, approx_bsp], slice, [approx_ws, 360, approx_bsp]])

    ws, wa, bsp = slice

    #connect if smaller than 180


    return ws, wa, bsp, info_

def _check_for_lines(wa):
    return wa.ndim == 1

def _get_new_axis(kind):
    return plt.axes(projection=kind)

def _configure_axes(ax, labels, colors, show_legend, legend_kw, **kwargs):
    _configure_colors(ax, labels, colors)
    _check_plot_kw(kwargs, True)
    if show_legend:
        _show_legend(ax, labels, colors, "True Wind Speed", legend_kw)

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


def _show_legend(ax, ws, colors, label, legend_kw):
    _configure_legend(ax, ws, colors, label, **legend_kw)


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
    ax.set_xlabel("TWS")
    ax.set_ylabel("Polar plane: TWA / BSP ")


def _remove_3d_tick_labels_for_polar_coordinates(ax):
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])


def _create_color_map(colors):
    return LinearSegmentedColormap.from_list("cmap", list(colors))


def _determine_face_colors(color_map, ws):
    return color_map((ws - ws.min()) / float((ws - ws.min()).max()))


def plot_convex_hull(
        ws, wa, bsp, ax, colors, show_legend, legend_kw, _lines, **plot_kw
):
    if ax is None:
        ax = _get_new_axis("polar")
    _set_polar_axis(ax)

    _check_plot_kw(plot_kw, _lines)

    wa, bsp = _convex_hull(zip(wa, bsp))

    _plot(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw)


def _convex_hull(slices):
    xs, ys = [], []
    for wa, bsp in slices:
        wa = np.asarray(wa)
        bsp = np.asarray(bsp)

        # convex hull is line between the two points
        # or is equal to one point
        if len(wa) < 3:
            xs.append(wa)
            ys.append(bsp)
            continue

        conv = _convex_hull_in_polar_coordinates(wa, bsp)
        vert = conv.vertices
        x, y = zip(
            *([(wa[i], bsp[i]) for i in vert] + [(wa[vert[0]], bsp[vert[0]])])
        )
        xs.append(list(x))
        ys.append(list(y))

    return xs, ys


def _convex_hull_in_polar_coordinates(wa, bsp):
    polar_points = np.column_stack((bsp * np.cos(wa), bsp * np.sin(wa)))
    return ConvexHull(polar_points)


def plot_convex_hull_multisails(
        ws, wa, bsp, members, ax, colors, show_legend, legend_kw, **plot_kw
):
    if ax is None:
        ax = _get_new_axis("polar")

    _set_polar_axis(ax)

    _check_plot_kw(plot_kw)

    xs, ys, members = _get_convex_hull_multisails(ws, wa, bsp, members)

    if colors is None:
        colors = plot_kw.pop("color", None) or plot_kw.pop("c", None) or []
    colors = dict(colors)
    _set_colors_multisails(ax, members, colors)

    if legend_kw is None:
        legend_kw = {}
    if show_legend:
        _set_legend_multisails(ax, colors, **legend_kw)

    for x, y in zip(list(xs), list(ys)):
        ax.plot(x, y, **plot_kw)


def _get_convex_hull_multisails(ws, wa, bsp, members):
    xs = []
    ys = []
    membs = []
    for s, w, b in zip(ws, wa, bsp):
        w = np.asarray(w)
        b = np.asarray(b)
        conv = _convex_hull_in_polar_coordinates(w, b)
        vert = sorted(conv.vertices)

        x, y, memb = zip(
            *(
                    [(w[i], b[i], members[i]) for i in vert]
                    + [(w[vert[0]], b[vert[0]], members[vert[0]])]
            )
        )
        x = list(x)
        y = list(y)
        memb = list(memb)

        for i in range(len(vert)):
            xs.append(x[i: i + 2])
            ys.append(y[i: i + 2])
            membs.append(memb[i: i + 2] + [s])

    return xs, ys, membs


def _set_colors_multisails(ax, members, colors):
    colorlist = []

    for member in members:
        # check if edge belongs to one or two sails
        # If it belongs to one sail, color it in that sails color
        # else color it in neutral color
        if len(set(member[:2])) == 1:
            color = colors.get(member[0], "blue")
        else:
            color = colors.get("neutral", "gray")

        if is_color_like(color):
            colorlist.append(color)
            continue

        color = dict(color)
        colorlist.append(color.get(member[2], "blue"))

    ax.set_prop_cycle("color", colorlist)


def _set_legend_multisails(ax, colors, **legend_kw):
    handles = []
    for key in colors:
        color = colors.get(key, "blue")

        if is_color_like(color):
            legend = Line2D([0], [0], color=color, lw=1, label=key)
            handles.append(legend)
            continue

        color = dict(color)
        legends = [
            Line2D(
                [0],
                [0],
                color=color.get(ws, "blue"),
                lw=1,
                label=f"{key} at TWS {ws}",
            )
            for ws in color
        ]
        handles.extend(legends)

    if "neutral" not in colors:
        legend = Line2D([0], [0], color="gray", lw=1, label="neutral")
        handles.append(legend)

    ax.legend(handles=handles, **legend_kw)
