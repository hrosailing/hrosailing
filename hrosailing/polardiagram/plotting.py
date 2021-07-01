"""
Various functions to plot PolarDiagram objects
"""

# Author: Valentin F. Dannenberg / Ente


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import (
    to_rgb,
    Normalize,
    LinearSegmentedColormap,
)
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull


def plot_polar(wa, bsp, ax, **plot_kw):
    _check_keywords(plot_kw)
    if ax is None:
        ax = plt.gca(projection='polar')
    _set_polar_directions(ax)

    colors = plot_kw.pop('color', None) \
        or plot_kw.pop('c', None)
    _set_color_cycle(ax, wa, colors)

    xs, ys = _sort_data(wa, bsp)
    return _plot(ax, xs, ys, **plot_kw)


def plot_flat(wa, bsp, ax, **plot_kw):
    _check_keywords(plot_kw)
    if ax is None:
        ax = plt.gca()

    colors = plot_kw.pop('color', None) \
        or plot_kw.pop('c', None)
    _set_color_cycle(ax, wa, colors)

    xs, ys = _sort_data(wa, bsp)
    return _plot(ax, xs, ys, **plot_kw)


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
    return _plot(ax, xs, ys, **plot_kw)


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

    return _plot(ax, xs, ys, **plot_kw)


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

    wa, bsp = _sort_data(wa, bsp)

    xs, ys = _get_convex_hull(wa, bsp)
    return ax.plot(xs, ys, **plot_kw)


# TODO: Many problems!!
def plot_convex_surface(ws, wa, bsp, ax, color):
    if ax is None:
        ax = plt.gca(projection='3d')
    _set_3d_labels(ax)

    xs, ys, zs = _get_convex_hull_3d(ws, wa, bsp)

    return ax.plot_surface(xs, ys, zs, rstride=1,
                           cstride=1)


def _check_keywords(dct):
    ls = dct.get('linestyle') or dct.get('ls')
    if ls is None:
        dct['ls'] = ''
    marker = dct.get('marker')
    if marker is None:
        dct['marker'] = 'o'
    color = dct.get('color') or dct.get('c')
    if color is None:
        dct['color'] = 'blue'


def _set_3d_labels(ax):
    ax.set_xlabel("True Wind Speed")
    ax.set_ylabel("True Wind Angle")
    ax.set_zlabel("Boat Speed")


def _set_polar_directions(ax):
    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')


def _set_color_cycle(ax, ws_list, colors):
    if not isinstance(ws_list, list):
        ws_list = [ws_list]
    if not isinstance(colors, tuple):
        colors = (colors,)
    no_plots = len(ws_list)
    no_colors = len(colors)

    if no_plots == no_colors or no_plots < no_colors:
        ax.set_prop_cycle('color', colors)
        return

    if no_plots > no_colors != 2:
        color_list = ['blue'] * no_plots
        if isinstance(colors[0], str):
            for i, c in enumerate(colors):
                print(i)
                print(c)
                color_list[i] = c

        if isinstance(colors[0], tuple):
            for ws, c in colors:
                i = list(ws_list).index(ws)
                color_list[i] = c

        ax.set_prop_cycle('color', color_list)
        return

    ax.set_prop_cycle(
        'color',
        _get_colors(colors, ws_list))


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


# TODO: Is there a better way?
#       Write it for 2 and 3 dim?
def _sort_data(wa_list, bsp_list):
    if not isinstance(wa_list, list):
        wa_list = [wa_list]
    if not isinstance(bsp_list, list):
        bsp_list = [bsp_list]

    sorted_ = list(zip(
        *(zip(*sorted(zip(wa, bsp), key=lambda x: x[0]))
          for wa, bsp in zip(wa_list, bsp_list))))

    return sorted_[0], sorted_[1]


# TODO: Is there a better way?
def _plot(ax, xs, ys,
          **plot_kw):
    for x, y in zip(xs, ys):
        x, y = np.asarray(x), np.asarray(y)
        ax.plot(x, y, **plot_kw)


# TODO: Write it for 2 and 3 dim?
def _get_convex_hull(wa, bsp):
    wa, bsp = np.asarray(wa).ravel(), np.asarray(bsp).ravel()
    vert = sorted(convex_hull_polar(
        np.column_stack((wa, bsp))).vertices)

    # maybe not list compr. but a for loop?
    xs = [wa[i] for i in vert]
    ys = [bsp[i] for i in vert]
    xs.append(xs[0])
    ys.append(ys[0])

    return xs, ys


# TODO: Merge with _get_convex_hull()?
def _get_convex_hull_3d(ws, wa, bsp):
    ws, wa, bsp = ws.ravel(), wa.ravel(), bsp.ravel()
    vert = sorted(ConvexHull(
        np.column_stack((ws, wa, bsp))).vertices)

    # maybe not list compr. but a for loop?
    xs = [ws[i] for i in vert]
    ys = [wa[i] for i in vert]
    zs = [bsp[i] for i in vert]
    xs.append(xs[0])
    ys.append(ys[0])
    zs.append(zs[0])

    xs = np.asarray(xs).reshape(-1, 1)
    ys = np.asarray(ys).reshape(-1, 1)
    zs = np.asarray(zs).reshape(-1, 1)
    return xs, ys, zs


def convex_hull_polar(points):
    converted_points = polar_to_kartesian(points)
    return ConvexHull(converted_points)


def polar_to_kartesian(arr):
    return np.column_stack(
        (arr[:, 1] * np.cos(arr[:, 0]),
         arr[:, 1] * np.sin(arr[:, 0])))
