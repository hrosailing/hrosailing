"""
Various functions to plot PolarDiagram objects
"""

# Author: Valentin F. Dannenberg / Ente


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import to_rgb, Normalize, \
    LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

from utils import convex_hull_polar


def plot_polar(wa, bsp, ax, **plot_kw):
    _check_keywords(plot_kw)
    if ax is None:
        ax = plt.gca(projection='polar')
    _set_polar_directions(ax)

    xs, ys = _sort_data([wa], [bsp])
    return ax.plot(xs, ys, **plot_kw)


def plot_flat(wa, bsp, ax, **plot_kw):
    _check_keywords(plot_kw)
    if ax is None:
        ax = plt.gca()

    xs, ys = _sort_data([wa], [bsp])
    return ax.plot(xs, ys, **plot_kw)


def plot_polar_range(ws_list, wa_list, bsp_list,
                     ax, colors, show_legend,
                     legend_kw, **plot_kw):
    _check_keywords(plot_kw)
    plot_kw.pop('color', None) or plot_kw.pop('c', None)
    if ax is None:
        ax = plt.gca(projection='polar')
    _set_polar_directions(ax)
    if legend_kw is None:
        legend_kw = {}
    if show_legend:
        _set_legend(ax, ws_list, colors,
                    label="True Wind Speed",
                    **legend_kw)
    _set_color_cycle(ax, ws_list, colors)

    xs, ys = _sort_data(wa_list, bsp_list)
    return _plot_multiple(ax, xs, ys, **plot_kw)


def plot_flat_range(ws_list, wa_list, bsp_list,
                    ax, colors, show_legend,
                    legend_kw, **plot_kw):
    _check_keywords(plot_kw)
    plot_kw.pop('color', None) or plot_kw.pop('c', None)
    if ax is None:
        ax = plt.gca()
    if legend_kw is None:
        legend_kw = {}
    if show_legend:
        _set_legend(ax, ws_list, colors,
                    label="True Wind Speed",
                    **legend_kw)
    _set_color_cycle(ax, ws_list, colors)

    xs, ys = _sort_data(wa_list, bsp_list)

    return _plot_multiple(ax, xs, ys, **plot_kw)


def plot_color(ws, wa, bsp, ax, colors,
               marker, show_legend,
               **legend_kw):
    if ax is None:
        ax = plt.gca()

    if legend_kw is None:
        legend_kw = {}
    if show_legend:
        _set_legend(
            ax, bsp, colors,
            label="Boat Speed",
            **legend_kw)

    colors = _get_colors(colors, bsp)

    return ax.scatter(ws, wa, marker=marker,
                      c=colors)


def plot3d(ws, wa, bsp, ax, **plot_kw):
    _check_keywords(plot_kw)

    if ax is None:
        ax = plt.gca(projection='3d')
    _set_3d_labels(ax)

    return ax.plot(ws, wa, bsp, **plot_kw)


def plot_surface(ws, wa, bsp, ax, colors):
    if ax is None:
        ax = plt.gca(projection='3d')
    _set_3d_labels(ax)

    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", list(colors))
    color = cmap((ws - ws.min())
                 / float((ws - ws.min()).max()))

    return ax.plot_surface(ws, wa, bsp,
                           facecolors=color)


def plot_convex_hull(wa, bsp, ax, **plot_kw):
    if ax is None:
        ax = plt.gca(projection='polar')
    _set_polar_directions(ax)

    wa, bsp = _sort_data([wa], [bsp])
    xs, ys = _get_convex_hull(wa, bsp)
    return ax.plot(xs, ys, **plot_kw)


def _check_keywords(plot_kw):
    ls = (plot_kw.get('linestyle')
          or plot_kw.get('ls'))
    if ls is None:
        plot_kw["ls"] = ''
    marker = plot_kw.get('marker')
    if marker is None:
        plot_kw["marker"] = 'o'


def _set_3d_labels(ax):
    ax.set_xlabel("True Wind Speed")
    ax.set_ylabel("True Wind Angle")
    ax.set_zlabel("Boat Speed")


def _set_polar_directions(ax):
    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')


def _set_color_cycle(ax, ws_list, colors):
    no_plots = len(ws_list)
    no_colors = len(colors)
    if no_plots == no_colors or no_plots < no_colors:
        ax.set_prop_cycle('color', colors)
        return

    if no_plots > no_colors != 2:
        color_list = ['blue'] * no_plots
        if isinstance(colors[0], str):
            for c, i in enumerate(colors):
                color_list[i] = c

        if isinstance(colors[0], tuple):
            for ws, c in colors:
                i = list(ws_list).index(ws)
                color_list[i] = c

        ax.set_prob_cycle('color', color_list)
        return

    ax.set_prop_cycle(
        'color',
        _get_colors(colors, ws_list))
    return


def _get_colors(colors, scal_list):
    min_color = np.array(to_rgb(colors[0]))
    max_color = np.array(to_rgb(colors[1]))
    scal_max = max(scal_list)
    scal_min = min(scal_list)

    coeffs = [(scal - scal_min) / (scal_max - scal_min)
              for scal in scal_list]

    return [(1 - coeff) * min_color
            + coeff * max_color
            for coeff in coeffs]


def _set_colormap(ws_list, colors, ax, label,
                  **legend_kw):
    min_color = colors[0]
    max_color = colors[1]
    ws_min = min(ws_list)
    ws_max = max(ws_list)

    cmap = LinearSegmentedColormap.from_list(
        "custom_map", [min_color, max_color])

    plt.colorbar(
        ScalarMappable(norm=Normalize(
            vmin=ws_min, vmax=ws_max), cmap=cmap),
        ax=ax, **legend_kw).set_label(label)


def _set_legend(ax, ws_list, colors, label,
                **legend_kw):
    no_colors = len(colors)
    no_plots = len(ws_list)

    if no_plots > no_colors == 2:
        _set_colormap(ws_list, colors, ax,
                      label, **legend_kw)
        return
    if isinstance(colors[0], tuple):
        ax.legend(handles=[Line2D(
            [0], [0], color=colors[i][1], lw=1,
            label=f"TWS {colors[i][0]}")
            for i in range(no_colors)])
        return
    ax.legend(handles=[Line2D(
        [0], [0], color=colors[i], lw=1,
        label=f"TWS {ws_list[i]}")
        for i in range(min(no_colors, no_plots))])
    return


# TODO Is there a better way?
def _sort_data(wa_list, bsp_list):
    sorted_lists = list(zip(
        *(zip(*sorted(zip(wa, bsp), key=lambda x: x[0]))
          for wa, bsp in zip(wa_list, bsp_list))))

    return sorted_lists[0], sorted_lists[1]


# TODO Is there a better way?
def _plot_multiple(ax, xs, ys,
                   **plot_kw):
    for x, y in zip(xs, ys):
        x, y = np.asarray(x), np.asarray(y)
        ax.plot(x, y, **plot_kw)


def _get_convex_hull(wa, bsp):
    wa, bsp = np.array(wa), np.array(bsp)

    vert = sorted(convex_hull_polar(
        np.column_stack((bsp.copy(), wa.copy()))).vertices)
    xs = []
    ys = []
    for i in vert:
        xs.append(wa[i])
        ys.append(bsp[i])
    xs.append(xs[0])
    ys.append(ys[0])

    return xs, ys
