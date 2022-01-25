"""
Contains various helper functions for the plot_*-methods() of the
PolarDiagram subclasses
"""

# pylint: disable=missing-function-docstring, missing-module-docstring

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (LinearSegmentedColormap, Normalize,
                               is_color_like, to_rgb)
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull


def plot_polar(
    ws, wa, bsp, ax, colors, show_legend, legend_kw, _lines, **plot_kw
):
    if ax is None:
        ax = _get_new_axis("polar")
    _set_polar_axis(ax)

    _check_plot_kw(plot_kw, _lines)

    _plot(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw)


def _get_new_axis(kind):
    return plt.axes(projection=kind)


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


def _plot(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw):
    _configure_colors(ax, ws, colors)

    if show_legend:
        _show_legend(ax, ws, colors, "True Wind Speed", legend_kw)

    for x, y in zip(wa, bsp):
        ax.plot(x, y, **plot_kw)


def _configure_colors(ax, ws, colors):
    if _only_one_color(colors):
        ax.set_prop_cycle("color", [colors])
        return

    if _more_colors_than_plots(ws, colors):
        ax.set_prop_cycle("color", colors)
        return

    if _no_color_gradient(colors):
        _set_color_cycle(ax, ws, colors)
        return

    _set_color_gradient(ax, ws, colors)


def _only_one_color(colors):
    return is_color_like(colors)


def _more_colors_than_plots(ws, colors):
    return len(ws) <= len(colors)


def _no_color_gradient(colors):
    return len(colors) != 2


def _set_color_cycle(ax, ws, colors):
    color_cycle = ["blue"] * len(ws)
    _configure_color_cycle(color_cycle, colors, ws)

    ax.set_prop_cycle("color", color_cycle)


def _configure_color_cycle(color_cycle, colors, ws):
    if isinstance(colors[0], tuple):
        if _only_one_color(colors[0]):
            for i, c in enumerate(colors):
                color_cycle[i] = c
                return

        for w, c in colors:
            i = list(ws).index(w)
            color_cycle[i] = c
            return

    for i, c in enumerate(colors):
        color_cycle[i] = c


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
    min_gradient = np.min(gradient)
    max_gradient = np.max(gradient)

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
    if legend_kw is None:
        legend_kw = {}

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
    return len(ws) > len(colors) == 2


def _set_colormap(ws, colors, ax, label, **legend_kw):
    color_map = _create_color_map(colors)
    plt.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=min(ws), vmax=max(ws)), cmap=color_map
        ),
        ax=ax,
        **legend_kw,
    ).set_label(label)


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


def plot_flat(
    ws, wa, bsp, ax, colors, show_legend, legend_kw, _lines, **plot_kw
):
    if ax is None:
        ax = _get_new_axis("rectlinear")

    _check_plot_kw(plot_kw, _lines)

    _plot(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw)


def plot_color_gradient(
    ws, wa, bsp, ax, colors, marker, ms, show_legend, **legend_kw
):
    if ax is None:
        ax = _get_new_axis("rectlinear")

    if show_legend:
        _show_legend(ax, bsp, colors, "Boat Speed", legend_kw)

    color_gradient = _determine_color_gradient(colors, bsp)

    ax.scatter(ws, wa, s=ms, marker=marker, c=color_gradient)


def plot3d(ws, wa, bsp, ax, colors, **plot_kw):
    if ax is None:
        ax = _get_new_axis("3d")

    _set_3d_axis_labels(ax)
    _remove_3d_tick_labels_for_polar_coordinates(ax)

    color_map = _create_color_map(colors)

    ax.scatter(ws, wa, bsp, c=ws, cmap=color_map, **plot_kw)


def _set_3d_axis_labels(ax):
    ax.set_xlabel("TWS")
    ax.set_ylabel("Polar plane: TWA / BSP ")


def _remove_3d_tick_labels_for_polar_coordinates(ax):
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabel([])


def _create_color_map(colors):
    return LinearSegmentedColormap.from_list("cmap", list(colors))


def plot_surface(ws, wa, bsp, ax, colors):
    if ax is None:
        ax = _get_new_axis("3d")

    _set_3d_axis_labels(ax)
    _remove_3d_tick_labels_for_polar_coordinates(ax)

    color_map = _create_color_map(colors)
    face_colors = _determine_face_colors(color_map, ws)

    ax.plot_surface(ws, wa, bsp, facecolors=face_colors)


def _determine_face_colors(color_map, ws):
    return color_map((ws - ws.min()) / float((ws - ws.min()).max()))


def plot_convex_hull(
    ws, wa, bsp, ax, colors, show_legend, legend_kw, _lines, **plot_kw
):
    if ax is None:
        ax = _get_new_axis("polar")
    _set_polar_axis(ax)

    _check_plot_kw(plot_kw, _lines)

    wa, bsp = _get_convex_hull(wa, bsp)

    _plot(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw)


def _get_convex_hull(wa, bsp):
    xs = []
    ys = []
    slices = zip(wa, bsp)

    for w, b in slices:
        w = np.asarray(w)
        b = np.asarray(b)

        # convex hull is line between the two points
        # or is equal to one point
        if len(w) < 3:
            xs.append(w)
            ys.append(b)
            continue

        conv = _convex_hull_in_polar_coordinates(w, b)
        vert = sorted(conv.vertices)
        x, y = zip(
            *([(w[i], b[i]) for i in vert] + [(w[vert[0]], b[vert[0]])])
        )
        xs.append(list(x))
        ys.append(list(y))

    return xs, ys


def _convex_hull_in_polar_coordinates(wa, bsp):
    polar_pts = np.column_stack((bsp * np.cos(wa), bsp * np.sin(wa)))
    return ConvexHull(polar_pts)


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
    members = members[0]
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
            xs.append(x[i : i + 2])
            ys.append(y[i : i + 2])
            membs.append(memb[i : i + 2] + [s])

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
