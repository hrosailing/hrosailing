import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from _utils import convex_hull_polar


def _check_keywords(plot_kw):
    ls = plot_kw.get('linestyle') or plot_kw.get('ls')
    if ls is None:
        plot_kw["ls"] = ''
    marker = plot_kw.get('marker')
    if marker is None:
        plot_kw["marker"] = 'o'

    return plot_kw


# V: Soweit in Ordnung
def plot_polar(wa, bsp, ax, **plot_kw):
    plot_kw = _check_keywords(plot_kw)
    if ax is None:
        ax = plt.gca(projection='polar')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    xs, ys = zip(*sorted(zip(wa, bsp), key=lambda x: x[0]))
    return ax.plot(xs, ys, **plot_kw)


# V: Soweit in Ordnung
def plot_flat(wa, bsp, ax, **plot_kw):
    plot_kw = _check_keywords(plot_kw)
    if ax is None:
        ax = plt.gca()
    # ax.set_xlabel("True Wind Angle")
    # ax.set_ylabel("Boat Speed")

    xs, ys = zip(*sorted(zip(wa, bsp), key=lambda x: x[0]))
    return ax.plot(xs, ys, **plot_kw)


def _get_color_cycle(ws_list, colors):
    no_plots = len(ws_list)
    no_colors = len(colors)
    if no_plots == no_colors or no_plots < no_colors:
        return colors

    if no_plots > no_colors != 2:
        if len(colors[0]) == 1:
            return list(colors) + ['blue'] * (no_plots - no_colors)

        if len(colors[0]) == 2:
            color_list = ['blue'] * no_plots
            for ws, c in colors:
                i = list(ws_list).index(ws)
                color_list[i] = c
            return color_list

    if no_colors == 2:
        ws_max = max(ws_list)
        ws_min = min(ws_list)
        min_color = np.array(to_rgb(colors[0]))
        max_color = np.array(to_rgb(colors[1]))
        coeffs = [(ws - ws_min) / (ws_max - ws_min)
                  for ws in ws_list]
        return [(1 - coeff) * min_color + coeff * max_color
                for coeff in coeffs]


def _get_legend(ws_list, colors):
    no_colors = len(colors)
    no_plots = len(ws_list)
    print(len(colors[0]))
    if isinstance(colors[0], tuple):
        return [Line2D(
            [0], [0], color=colors[i][1], lw=1,
            label=f"TWS {colors[i][0]}")
            for i in range(no_colors)]

    return [Line2D(
        [0], [0], color=colors[i], lw=1,
        label=f"TWS {ws_list[i]}")
        for i in range(min(no_colors, no_plots))]


def _set_colormap(ws_list, colors, ax, label, **legend_kw):
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


# V: In Arbeit
def plot_polar_range(ws_list, wa_list, bsp_list,
                     ax, colors, show_legend, legend_kw, **plot_kw):
    ls = plot_kw.get('linestyle') or plot_kw.get('ls')
    if ls is None:
        plot_kw["ls"] = ''
    marker = plot_kw.get('marker')
    if marker is None:
        plot_kw["marker"] = 'o'
    _ = plot_kw.pop('color', None) or plot_kw.pop('c', None)

    if ax is None:
        ax = plt.gca(projection='polar')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')

    if legend_kw is None:
        legend_kw = {}
    if show_legend:
        if len(ws_list) > len(colors) == 2:
            _set_colormap(
                ws_list, colors, ax,
                label="True Wind Speed", **legend_kw)
        else:
            legend = _get_legend(ws_list, colors)
            ax.legend(handles=legend, **legend_kw)

    color_cycle = _get_color_cycle(ws_list, colors)
    ax.set_prop_cycle('color', color_cycle)

    wa_list, bsp_list = zip(*sorted(zip(wa_list, bsp_list),
                                    key=lambda x: x[0]))
    xs = np.column_stack(wa_list)
    ys = np.column_stack(bsp_list)
    return ax.plot(xs, ys, **plot_kw)


# V: In Arbeit
def plot_flat_range(ws_list, wa_list, bsp_list,
                    ax, colors, show_legend, legend_kw, **plot_kw):
    ls = plot_kw.get('linestyle') or plot_kw.get('ls')
    if ls is None:
        plot_kw["ls"] = ''
    marker = plot_kw.get('marker')
    if marker is None:
        plot_kw["marker"] = 'o'
    _ = plot_kw.pop('color', None) or plot_kw.pop('c', None)

    if ax is None:
        ax = plt.gca()
    # ax.set_xlabel("True Wind Angle")
    # ax.set_ylabel("Boat Speed")

    if legend_kw is None:
        legend_kw = {}
    if show_legend:
        if len(ws_list) > len(colors) == 2:
            _set_colormap(
                ws_list, colors, ax,
                label="True Wind Speed", **legend_kw)
        else:
            legend = _get_legend(ws_list, colors)
            ax.legend(handles=legend, **legend_kw)

    color_cycle = _get_color_cycle(ws_list, colors)
    ax.set_prop_cycle('color', color_cycle)

    wa_list, bsp_list = zip(*sorted(zip(wa_list, bsp_list),
                                    key=lambda x: x[0]))
    xs = np.column_stack(wa_list)
    ys = np.column_stack(bsp_list)
    return ax.plot(xs, ys, **plot_kw)


# V: Soweit in Ordnung
def plot_color(ws, wa, bsp, ax, colors,
               marker, show_legend, **legend_kw):
    if ax is None:
        ax = plt.gca()

    # ax.set_xlabel("True Wind Speed")
    # ax.set_ylabel("True Wind Angle")

    color = _get_color_cycle(bsp, colors)

    if legend_kw is None:
        legend_kw = {}
    if show_legend:
        _set_colormap(bsp, colors, ax,
                      label="Boat Speed", **legend_kw)

    return ax.scatter(ws, wa, c=color, marker=marker)


# V: In Arbeit
def plot3d(ws, wa, bsp, ax, **plot_kw):
    plot_kw = _check_keywords(plot_kw)
    if ax is None:
        ax = plt.gca(projection='3d')

    ax.set_xlabel("True Wind Speed")
    ax.set_ylabel("True Wind Angle")
    ax.set_zlabel("Boat Speed")
    return ax.plot(ws, wa, bsp, **plot_kw)


# V: In Arbeit
def plot_surface(ws, wa, bsp, ax, colors):
    if ax is None:
        ax = plt.gca(projection='3d')

    ax.set_xlabel("True Wind Speed")
    ax.set_ylabel("True Wind Angle")
    ax.set_zlabel("Boat Speed")
    cmap = LinearSegmentedColormap.from_list("custom_cmap", list(colors))
    color = cmap((ws - ws.min()) / float((ws - ws.min()).max()))
    return ax.plot_surface(ws, wa, bsp, facecolors=color)


# V: Soweit in Ordnung
def plot_convex_hull(wa, bsp, ax, **plot_kw):
    if ax is None:
        ax = plt.gca(projection='polar')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')

    wa, bsp = zip(*sorted(zip(wa, bsp), key=lambda x: x[0]))
    wa, bsp = np.array(wa), np.array(bsp).reshape(-1,)
    vert = sorted(convex_hull_polar(bsp.copy(), wa.copy()).vertices)
    xs = []
    ys = []
    for i in vert:
        xs.append(wa[i])
        ys.append(bsp[i])
    xs.append(xs[0])
    ys.append(ys[0])
    return ax.plot(xs, ys, **plot_kw)
