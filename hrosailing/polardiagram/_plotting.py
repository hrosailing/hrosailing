"""
Contains various helper functions for the plot_*-methods() of the 
PolarDiagram subclasses
"""

# Author: Valentin Dannenberg

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (LinearSegmentedColormap, Normalize,
                               is_color_like, to_rgb)
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull


def plot_polar(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw):
    if ax is None:
        ax = plt.axes(projection="polar")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")

    return _plot(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw)


def plot_flat(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw):
    if ax is None:
        ax = plt.gca()

    return _plot(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw)


def plot_color_gradient(
    ws, wa, bsp, ax, colors, marker, ms, show_legend, **legend_kw
):
    if ax is None:
        ax = plt.gca()

    if legend_kw is None:
        legend_kw = {}
    if show_legend:
        _set_legend(ax, bsp, colors, label="Boat Speed", **legend_kw)

    min_color = np.array(to_rgb(colors[0]))
    max_color = np.array(to_rgb(colors[1]))
    min_bsp = np.min(bsp)
    max_bsp = np.max(bsp)

    coeffs = [(b - min_bsp) / (max_bsp - min_bsp) for b in bsp]
    colors = [(1 - c) * min_color + c * max_color for c in coeffs]

    return ax.scatter(ws, wa, s=ms, marker=marker, c=colors)


def plot3d(ws, wa, bsp, ax, colors, **plot_kw):
    if ax is None:
        ax = plt.axes(projection="3d")

    ax.set_xlabel("TWS")
    ax.set_ylabel("Polar plane: TWA / BSP ")

    # remove axis labels since we are using polar-coordinates,
    # which are transformed to cartesian, so the labels
    # would be wrong
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    cmap = LinearSegmentedColormap.from_list("cmap", [colors[0], colors[1]])

    return ax.scatter(ws, wa, bsp, c=ws, cmap=cmap, **plot_kw)


def plot_surface(ws, wa, bsp, ax, colors):
    if ax is None:
        ax = plt.axes(projection="3d")

    ax.set_xlabel("TWS")
    ax.set_ylabel("Polar plane: TWA / BSP ")

    # remove axis labels since we are using polar-coordinates,
    # which are transformed to cartesian, so the labels
    # would be wrong
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    cmap = LinearSegmentedColormap.from_list("cmap", list(colors))
    color = cmap((ws - ws.min()) / float((ws - ws.min()).max()))

    return ax.plot_surface(ws, wa, bsp, facecolors=color)


def plot_convex_hull(
    ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw
):
    if ax is None:
        ax = plt.axes(projection="polar")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")

    _prepare_plot(ax, ws, colors, show_legend, legend_kw, plot_kw)

    xs, ys = _get_convex_hull(wa, bsp)

    for x, y in zip(list(xs), list(ys)):
        ax.plot(x, y, **plot_kw)


def plot_convex_hull_multisails(
    ws, wa, bsp, members, ax, colors, show_legend, legend_kw, **plot_kw
):
    if ax is None:
        ax = plt.axes(projection="polar")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")

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


def _plot(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw):
    _prepare_plot(ax, ws, colors, show_legend, legend_kw, plot_kw)

    for x, y in zip(wa, bsp):
        ax.plot(x, y, **plot_kw)


def _prepare_plot(ax, ws, colors, show_legend, legend_kw, plot_kw):
    if colors is None:
        colors = plot_kw.pop("color", None) or plot_kw.pop("c", None)
    _set_color_cycle(ax, ws, colors)

    if legend_kw is None:
        legend_kw = {}
    if show_legend:
        _set_legend(ax, ws, colors, label="True Wind Speed", **legend_kw)


def _set_color_cycle(ax, ws, colors):
    if is_color_like(colors):
        ax.set_prop_cycle("color", [colors])
        return

    if colors is None:
        colors = ["blue"]

    n_plots = len(ws)
    n_colors = len(colors)
    if n_plots <= n_colors:
        ax.set_prop_cycle("color", colors)
        return

    if n_plots > n_colors != 2:
        colorlist = ["blue"] * n_plots
        _set_colorlist(colors, colorlist, ws)
        ax.set_prop_cycle("color", colorlist)

        return

    # n_colors == 2
    min_color = np.array(to_rgb(colors[0]))
    max_color = np.array(to_rgb(colors[1]))
    min_ws = min(ws)
    max_ws = max(ws)

    coeffs = [(w - min_ws) / (max_ws - min_ws) for w in ws]
    colors = [(1 - c) * min_color + c * max_color for c in coeffs]
    ax.set_prop_cycle("color", colors)


def _set_colorlist(colors, colorlist, ws):
    if isinstance(colors[0], tuple):
        if is_color_like(colors[0]):
            for i, c in enumerate(colors):
                colorlist[i] = c
                return

        for w, c in colors:
            i = list(ws).index(w)
            colorlist[i] = c
            return

    for i, c in enumerate(colors):
        colorlist[i] = c


def _set_legend(ax, ws, colors, label, **legend_kw):
    n_colors = len(colors)
    n_plots = len(ws)

    if n_plots > n_colors == 2:
        _set_colormap(ws, colors, ax, label, **legend_kw)
        return

    if isinstance(colors[0], tuple) and not is_color_like(colors[0]):
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=colors[i][1],
                    lw=1,
                    label=f"TWS {colors[i][0]}",
                )
                for i in range(n_colors)
            ],
            **legend_kw,
        )
        return

    ax.legend(
        handles=[
            Line2D([0], [0], color=colors[i], lw=1, label=f"TWS {ws[i]}")
            for i in range(min(n_plots, n_colors))
        ],
        **legend_kw,
    )


def _set_colormap(ws, colors, ax, label, **legend_kw):
    cmap = LinearSegmentedColormap.from_list("cmap", [colors[0], colors[1]])
    plt.colorbar(
        ScalarMappable(norm=Normalize(vmin=min(ws), vmax=max(ws)), cmap=cmap),
        ax=ax,
        **legend_kw,
    ).set_label(label)


def _get_convex_hull(wa, bsp):
    xs = []
    ys = []

    for w, b in zip(wa, bsp):
        w = np.asarray(w)
        b = np.asarray(b)

        # convex hull is just line between the two points
        # or is equal to just one point
        if len(w) < 3:
            xs.append(w)
            ys.append(b)
            continue

        conv = _convex_hull_polar(w, b)
        vert = sorted(conv.vertices)
        x, y = zip(
            *([(w[i], b[i]) for i in vert] + [(w[vert[0]], b[vert[0]])])
        )
        xs.append(list(x))
        ys.append(list(y))

    return xs, ys


def _get_convex_hull_multisails(ws, wa, bsp, members):
    members = members[0]
    xs = []
    ys = []
    membs = []
    for s, w, b in zip(ws, wa, bsp):
        w = np.asarray(w)
        b = np.asarray(b)
        conv = _convex_hull_polar(w, b)
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


def _convex_hull_polar(wa, bsp):
    polar_pts = np.column_stack((bsp * np.cos(wa), bsp * np.sin(wa)))
    return ConvexHull(polar_pts)


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
