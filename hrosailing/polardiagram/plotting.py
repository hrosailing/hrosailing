"""
Various functions to plot PolarDiagram objects
"""

# Author: Valentin F. Dannenberg / Ente


import matplotlib.pyplot as plt
import numpy as np
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
        ax = plt.gca(projection="polar")
    _set_polar_directions(ax)

    _prepare_plot(ax, ws, wa, colors, show_legend, legend_kw, **plot_kw)

    xs, ys = _sort_data(wa, bsp)
    return _plot(ax, xs, ys, **plot_kw)


def plot_flat(ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw):
    if ax is None:
        ax = plt.gca()

    _prepare_plot(ax, ws, wa, colors, show_legend, legend_kw, **plot_kw)

    xs, ys = _sort_data(wa, bsp)
    return _plot(ax, xs, ys, **plot_kw)


def plot_color_gradient(
    ws, wa, bsp, ax, colors, marker, show_legend, **legend_kw
):
    if ax is None:
        ax = plt.gca()

    if legend_kw is None:
        legend_kw = {}
    if show_legend:
        _set_legend(ax, bsp, colors, label="Boat Speed", **legend_kw)

    colors = _get_colors(colors, bsp)

    return ax.scatter(ws, wa, marker=marker, c=colors)


def plot3d(ws, wa, bsp, ax, **plot_kw):
    _check_keywords(plot_kw)

    if ax is None:
        ax = plt.gca(projection="3d")
    _set_3d_labels(ax)

    return ax.plot(ws, wa, bsp, **plot_kw)


def plot_surface(ws, wa, bsp, ax, colors):
    if ax is None:
        ax = plt.gca(projection="3d")
    _set_3d_labels(ax)

    cmap = LinearSegmentedColormap.from_list("custom_cmap", list(colors))
    color = cmap((ws - ws.min()) / float((ws - ws.min()).max()))

    return ax.plot_surface(ws, wa, bsp, facecolors=color)


def plot_convex_hull(
    ws, wa, bsp, ax, colors, show_legend, legend_kw, **plot_kw
):
    if ax is None:
        ax = plt.gca(projection="polar")
    _set_polar_directions(ax)

    _prepare_plot(ax, ws, wa, colors, show_legend, legend_kw, **plot_kw)

    wa, bsp = _sort_data(wa, bsp)
    xs, ys = _get_convex_hull(wa, bsp)

    return _plot(ax, xs, ys, **plot_kw)


def _prepare_plot(ax, ws, wa, colors, show_legend, legend_kw, **plot_kw):
    _check_keywords(plot_kw)

    if colors is None:
        colors = plot_kw.pop("color", None) or plot_kw.pop("c", None)
    if isinstance(ws, list):
        c = _set_color_cycle(ax, ws, colors)
    else:
        c = _set_color_cycle(ax, wa, colors)
    if c is not None:
        plot_kw["c"] = c

    if legend_kw is None:
        legend_kw = {}
    if show_legend:
        _set_legend(ax, ws, colors, label="True Wind Speed", **legend_kw)

    if isinstance(ws, list):
        c = _set_color_cycle(ax, ws, colors)
    else:
        c = _set_color_cycle(ax, wa, colors)
    if c is not None:
        plot_kw["c"] = c


# TODO: Many problems!!
# def plot_convex_surface(ws, wa, bsp, ax, color):
#     if ax is None:
#         ax = plt.gca(projection="3d")
#     _set_3d_labels(ax)
#
#     xs, ys, zs = _get_convex_hull_3d(ws, wa, bsp)
#
#     return ax.plot_surface(xs, ys, zs, rstride=1, cstride=1)


def _check_keywords(dct):
    ls = dct.get("linestyle") or dct.get("ls")
    if ls is None:
        dct["ls"] = ""
    marker = dct.get("marker")
    if marker is None:
        dct["marker"] = "o"


def _set_3d_labels(ax):
    ax.set_xlabel("True Wind Speed")
    ax.set_ylabel("True Wind Angle")
    ax.set_zlabel("Boat Speed")


def _set_polar_directions(ax):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")


def _set_color_cycle(ax, ws, colors):
    if is_color_like(colors):
        return colors
    if colors is None:
        return "blue"

    if not isinstance(ws, list):
        ws = [ws]

    n_plots = len(ws)
    n_colors = len(colors)
    if n_plots <= n_colors:
        ax.set_prop_cycle("color", colors)
        return

    if n_plots > n_colors != 2:
        color_list = ["blue"] * n_plots
        if isinstance(colors[0], tuple):
            if is_color_like(colors[0]):
                for i, c in enumerate(colors):
                    color_list[i] = c
            for w, c in colors:
                i = list(ws).index(w)
                color_list[i] = c
        else:
            for i, c in enumerate(colors):
                color_list[i] = c

        ax.set_prop_cycle("color", color_list)
        return

    ax.set_prop_cycle("color", _get_colors(colors, ws))


def _get_colors(colors, grad):
    min_color = np.array(to_rgb(colors[0]))
    max_color = np.array(to_rgb(colors[1]))
    grad_max = max(grad)
    grad_min = min(grad)

    coeffs = [(g - grad_min) / (grad_max - grad_min) for g in grad]

    return [(1 - coeff) * min_color + coeff * max_color for coeff in coeffs]


def _set_colormap(ws, colors, ax, label, **legend_kw):
    min_color = colors[0]
    max_color = colors[1]
    ws_min = min(ws)
    ws_max = max(ws)

    cmap = LinearSegmentedColormap.from_list(
        "custom_map", [min_color, max_color]
    )

    plt.colorbar(
        ScalarMappable(norm=Normalize(vmin=ws_min, vmax=ws_max), cmap=cmap),
        ax=ax,
        **legend_kw,
    ).set_label(label)


def _set_legend(ax, ws, colors, label, **legend_kw):
    n_colors = len(colors)
    n_plots = len(ws)

    if n_plots == 1:
        ax.legend(
            handles=[
                Line2D([0], [0], color=colors, lw=1, label=f"TWS {ws[0]}")
            ]
        )

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
            ]
        )
        return

    ax.legend(
        handles=[
            Line2D([0], [0], color=colors[i], lw=1, label=f"TWS {ws[i]}")
            for i in range(min(n_plots, n_colors))
        ]
    )


def _sort_data(wa, bsp):
    if not isinstance(wa, list):
        wa = [wa]
    if not isinstance(bsp, list):
        bsp = [bsp]

    xs, ys = list(
        zip(
            *(
                zip(*sorted(zip(wa, bsp), key=lambda x: x[0]))
                for wa, bsp in zip(wa, bsp)
            )
        )
    )

    return xs, ys


# TODO: Is there a better way?
def _plot(ax, xs, ys, **plot_kw):
    for x, y in zip(xs, ys):
        x, y = np.asarray(x), np.asarray(y)
        ax.plot(x, y, **plot_kw)


def _get_convex_hull(wa, bsp):
    if not isinstance(wa, list):
        wa = [wa]
    if not isinstance(bsp, list):
        bsp = [bsp]
    xs = ys = []
    for w, b in zip(wa, bsp):
        w, b = np.asarray(w).ravel(), np.asarray(b).ravel()
        vert = sorted(_convex_hull_polar(np.column_stack((w, b))).vertices)
        x, y = list(zip(*([(w[i], b[i]) for i in vert])))
        x.append(x[0])
        y.append(y[0])
        xs.append(x)
        ys.append(y)

    return xs, ys


def _convex_hull_polar(pts):
    polar_pts = np.column_stack(
        (pts[:, 1] * np.cos(pts[:, 0]), pts[:, 1] * np.sin(pts[:, 0]))
    )
    return ConvexHull(polar_pts)


# TODO: Merge with _get_convex_hull()?
# def _get_convex_hull_3d(ws, wa, bsp):
#     ws, wa, bsp = ws.ravel(), wa.ravel(), bsp.ravel()
#     vert = sorted(ConvexHull(np.column_stack((ws, wa, bsp))).vertices)
#
#     # maybe not list compr. but a for loop?
#     xs = [ws[i] for i in vert]
#     ys = [wa[i] for i in vert]
#     zs = [bsp[i] for i in vert]
#     xs.append(xs[0])
#     ys.append(ys[0])
#     zs.append(zs[0])
#
#     xs = np.asarray(xs).reshape(-1, 1)
#     ys = np.asarray(ys).reshape(-1, 1)
#     zs = np.asarray(zs).reshape(-1, 1)
#     return xs, ys, zs
