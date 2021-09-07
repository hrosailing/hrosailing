"""
Contains various helper functions to plot polar diagrams in a number of ways
"""

# Author: Valentin F. Dannenberg / Ente

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import (
    to_rgb,
    is_color_like,
    Normalize,
    LinearSegmentedColormap,
)
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull


def plot_polar(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw):
    if ax is None:
        ax = plt.axes(projection="polar")

    _set_polar_directions(ax)
    _prepare_plot(ax, ws, colors, show_legend, legend_kw, plot_kw)
    xs, ys = _sort_data(wa, bsp)
    return _plot(ax, xs, ys, **plot_kw)


def plot_flat(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw):
    if ax is None:
        ax = plt.gca()

    _prepare_plot(ax, ws, colors, show_legend, legend_kw, plot_kw)

    xs, ys = _sort_data(wa, bsp)
    return _plot(ax, xs, ys, **plot_kw)


def plot_color_gradient(
    ws, wa, bsp, ax, colors, marker, ms, show_legend, **legend_kw
):
    if ax is None:
        ax = plt.gca()

    if legend_kw is None:
        legend_kw = {}
    if show_legend:
        _set_legend(ax, bsp, colors, label="Boat Speed", **legend_kw)

    colors = _get_colors(colors, bsp)

    return ax.scatter(ws, wa, s=ms, marker=marker, c=colors)


def plot3d(ws, wa, bsp, ax, colors, **plot_kw):
    if ax is None:
        ax = plt.axes(projection="3d")

    _set_3d_labels(ax)
    print(colors)
    cmap = LinearSegmentedColormap.from_list("cmap", [colors[0], colors[1]])

    return ax.scatter(ws, wa, bsp, c=ws, cmap=cmap, **plot_kw)


def plot_surface(ws, wa, bsp, ax, **plot_kw):
    if ax is None:
        ax = plt.axes(projection="3d")

    _set_3d_labels(ax)

    colors = plot_kw.get("color") or plot_kw.get("c") or ("green", "red")
    if not isinstance(colors, (list, tuple)):
        colors = [colors, colors]

    cmap = LinearSegmentedColormap.from_list("cmap", list(colors))
    color = cmap((ws - ws.min()) / float((ws - ws.min()).max()))

    return ax.plot_surface(ws, wa, bsp, facecolors=color)


def plot_convex_hull(
    ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw
):
    if ax is None:
        ax = plt.axes(projection="polar")

    _set_polar_directions(ax)
    ls = plot_kw.get("linestyle") or plot_kw.get("ls")
    if ls is None:
        plot_kw["ls"] = "-"
    _prepare_plot(ax, ws, colors, show_legend, legend_kw, plot_kw)

    wa, bsp = _sort_data(wa, bsp)
    xs, ys = _get_convex_hull(wa, bsp)

    return _plot(ax, xs, ys, **plot_kw)


def plot_convex_hull_multisails(
    ws, wa, bsp, members, ax, colors, show_legend, legend_kw, **plot_kw
):
    if ax is None:
        ax = plt.axes(projection="polar")

    _set_polar_directions(ax)
    ls = plot_kw.get("linestyle") or plot_kw.get("ls")
    if ls is None:
        plot_kw["ls"] = "-"

    if colors is None:
        colors = plot_kw.pop("color", None) or plot_kw.pop("c", None) or []

    xs, ys, members = _get_convex_hull_multisails(ws, wa, bsp, members)

    colors = dict(colors)
    _set_colors_multisails(ax, ws, members, colors)
    if legend_kw is None:
        legend_kw = {}
    if show_legend:
        _set_legend_multisails(ax, colors, **legend_kw)

    return _plot(ax, xs, ys, **plot_kw)


def _prepare_plot(ax, ws, colors, show_legend, legend_kw, plot_kw):
    _check_keywords(plot_kw)

    if colors is None:
        colors = plot_kw.pop("color", None) or plot_kw.pop("c", None)
    _set_color_cycle(ax, ws, colors)

    if legend_kw is None:
        legend_kw = {}
    if show_legend:
        _set_legend(ax, ws, colors, label="True Wind Speed", **legend_kw)


def _check_keywords(dct):
    ls = dct.pop("linestyle", None) or dct.pop("ls", None)
    if ls is None:
        dct["ls"] = ""
    else:
        dct["ls"] = ls
    marker = dct.get("marker", None)
    if marker is None:
        dct["marker"] = "o"


def _set_polar_directions(ax):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")


def _set_3d_labels(ax):
    ax.set_xlabel("TWS")
    ax.set_ylabel("Polar plane: TWA / BSP ")
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])


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
    ax.set_prop_cycle("color", _get_colors(colors, ws))


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


def _get_colors(colors, grad):
    min_color = np.array(to_rgb(colors[0]))
    max_color = np.array(to_rgb(colors[1]))
    grad_max = max(grad)
    grad_min = min(grad)

    coeffs = [(g - grad_min) / (grad_max - grad_min) for g in grad]
    return [(1 - coeff) * min_color + coeff * max_color for coeff in coeffs]


def _set_colormap(ws, colors, ax, label, **legend_kw):
    cmap = LinearSegmentedColormap.from_list("cmap", [colors[0], colors[1]])
    plt.colorbar(
        ScalarMappable(norm=Normalize(vmin=min(ws), vmax=max(ws)), cmap=cmap),
        ax=ax,
        **legend_kw,
    ).set_label(label)


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


def _sort_data(wa, bsp):
    if not isinstance(wa, list):
        wa = [wa]
    if not isinstance(bsp, list):
        bsp = [bsp]

    xs, ys = zip(
        *(zip(*sorted(zip(w, b), key=lambda x: x[0])) for w, b in zip(wa, bsp))
    )

    return list(xs), list(ys)


def _plot(ax, xs, ys, **plot_kw):
    for x, y in zip(xs, ys):
        ax.plot(x, y, **plot_kw)


def _get_convex_hull(wa, bsp):
    if not isinstance(wa, list):
        wa = [wa]
    if not isinstance(bsp, list):
        bsp = [bsp]
    xs = []
    ys = []
    for w, b in zip(wa, bsp):
        w = np.asarray(w).ravel()
        b = np.asarray(b).ravel()
        if len(w) < 3:
            xs.append(w)
            ys.append(b)
            continue
        conv = _convex_hull_polar(w, b)
        vert = sorted(conv.vertices)
        x, y = zip(*([(w[i], b[i]) for i in vert]))
        x = list(x)
        x.append(x[0])
        y = list(y)
        y.append(y[0])
        xs.append(x)
        ys.append(y)

    return xs, ys


def _get_convex_hull_multisails(ws, wa, bsp, members):
    wa, bsp, members = zip(
        *(
            zip(*sorted(zip(w, b, members), key=lambda tup: tup[0]))
            for w, b in zip(wa, bsp)
        )
    )
    members = members[0]
    xs = []
    ys = []
    membs = []
    for s, w, b in zip(ws, wa, bsp):
        w = np.asarray(w)
        b = np.asarray(b)
        conv = _convex_hull_polar(w, b)
        vert = sorted(conv.vertices)
        x, y, memb = zip(*([(w[i], b[i], members[i]) for i in vert]))
        x = list(x)
        x.append(x[0])
        y = list(y)
        y.append(y[0])
        memb = list(memb)
        memb.append(memb[0])
        for i in range(len(vert)):
            xs.append(x[i : i + 2])
            ys.append(y[i : i + 2])
            m = memb[i : i + 2]
            m.append(s)
            membs.append(m)
    return xs, ys, membs


# TODO Finish color api
def _set_colors_multisails(ax, ws, members, colors):
    colorlist = []

    for member in members:
        if len(set(member[:2])) == 1:
            color = colors.get(member[0], "blue")
            if is_color_like(color):
                colorlist.append(color)
                continue

            color = dict(color)
            colorlist.append(color.get(member[2], "blue"))
            continue

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


def _convex_hull_polar(wa, bsp):
    polar_pts = np.column_stack((bsp * np.cos(wa), bsp * np.sin(wa)))
    return ConvexHull(polar_pts)
