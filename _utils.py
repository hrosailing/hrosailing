import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import Iterable
from matplotlib.colors import to_rgb, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.interpolate import bisplrep, bisplev, griddata, \
    SmoothBivariateSpline
from scipy.spatial import ConvexHull
from _exceptions import PolarDiagramException
from _sailing_units import *


# V: Soweit in Ordnung
def polar_to_kartesian(rad, ang):
    return np.column_stack((rad * np.cos(ang),
                            rad * np.sin(ang)))


# V: Soweit in Ordnung
def convex_hull_polar(points_rad, points_ang):
    converted_points = polar_to_kartesian(points_rad, points_ang)
    return ConvexHull(converted_points)


# V: In Arbeit
def read_table(csv_reader):
    next(csv_reader)
    ws_res = [eval(ws) for ws in next(csv_reader)]
    next(csv_reader)
    wa_res = [eval(wa) for wa in next(csv_reader)]
    next(csv_reader)
    data = []
    for row in csv_reader:
        data.append([eval(bsp) for bsp in row])

    return ws_res, wa_res, data


# V: In Arbeit
def read_pointcloud(csv_reader):
    data = []
    next(csv_reader)
    for row in csv_reader:
        data.append([eval(entry) for entry in row])

    return np.array(data)


# V: In Arbeit
def read_extern_format(csv_path, fmt):
    if fmt == 'orc':
        return read_orc_csv(csv_path)
    if fmt == 'array':
        return read_array_csv(csv_path)
    if fmt == 'opencpn':
        return read_opencpn_csv(csv_path)


# V: In Arbeit
def read_orc_csv(csv_path):
    with open(csv_path, 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=';', quotechar='"')
        ws_res = [eval(s) for s in next(csv_reader)[1:]]
        wa_res = []
        data = []
        next(csv_reader)
        for row in csv_reader:
            wa_res.append(eval(row[0]))
            data.append([eval(bsp) for bsp in row[1:]])
        return ws_res, wa_res, data


# V: In Arbeit
def read_array_csv(csv_path):
    file_data = np.genfromtxt(csv_path, delimiter="\t")
    return file_data[0, 1:], file_data[1:, 0], file_data[1:, 1:]


# V: In Arbeit
def read_opencpn_csv(csv_path):
    with open(csv_path, 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=',', quotechar='"')
        ws_res = [eval(ws) for ws in next(csv_reader)[1:]]
        wa_res = []
        data = []
        next(csv_reader)
        for row in csv_reader:
            ws_res.append(eval(row[0]))
            data.append([eval(bsp) if bsp != '' else 0 for bsp in row[1:]])

        return ws_res, wa_res, data


# V: In Arbeit
def convert_wind(w_dict, tw):
    if tw:
        return w_dict

    aws = w_dict.get("wind_speed")
    awa = w_dict.get("wind_angle")
    bsp = w_dict.get("boat_speed")
    tws, twa = apparent_wind_to_true(aws, awa, bsp)
    return {"wind_speed": tws, "wind_angle": twa}


# V: In Arbeit
def speed_resolution(ws_res):
    if ws_res is None:
        return np.array(np.arange(2, 42, 2))

    if not isinstance(ws_res, (Iterable, int, float)):
        raise PolarDiagramException(
            "ws_res is neither Iterable, int or float")

    if isinstance(ws_res, Iterable):
        return np.array(list(ws_res))

    return np.array(np.arange(ws_res, 40, ws_res))


# V: In Arbeit
def angle_resolution(wa_res):
    if wa_res is None:
        return np.array(np.arange(0, 360, 5))

    if not isinstance(wa_res, (Iterable, int, float)):
        raise PolarDiagramException(
            "wa_res is neither Iterable, int or float")

    if isinstance(wa_res, Iterable):
        return np.array(list(wa_res))

    return np.array(np.arange(wa_res, 360, wa_res))


# V: In Arbeit
def get_indices(w_list, res_list):
    if w_list is None:
        return list(range(len(res_list)))

    if not isinstance(w_list, Iterable):
        try:
            ind = list(res_list).index(w_list)
            return [ind]
        except ValueError:
            raise PolarDiagramException(
                f"{w_list} is not in resolution")

    if not set(w_list).issubset(set(res_list)):
        raise PolarDiagramException(
            f"{w_list} is not in resolution")

    ind_list = [i for i in range(len(res_list)) if res_list[i] in w_list]
    return ind_list


# V: Soweit in Ordnung
def plot_polar(wa, bsp, ax, **plot_kw):
    ls = plot_kw.get('linestyle') or plot_kw.get('ls')
    if ls is None:
        plot_kw["ls"] = ''
    marker = plot_kw.get('marker')
    if marker is None:
        plot_kw["marker"] = 'o'

    if ax is None:
        ax = plt.gca(projection='polar')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')

    xs, ys = zip(*sorted(zip(wa, bsp), key=lambda x: x[0]))
    return ax.plot(xs, ys, **plot_kw)


# V: Soweit in Ordnung
def plot_flat(wa, bsp, ax, **plot_kw):
    ls = plot_kw.get('linestyle') or plot_kw.get('ls')
    if ls is None:
        plot_kw["ls"] = ''
    marker = plot_kw.get('marker')
    if marker is None:
        plot_kw["marker"] = 'o'

    if ax is None:
        ax = plt.gca()
    # ax.set_xlabel("True Wind Angle")
    # ax.set_ylabel("Boat Speed")

    xs, ys = zip(*sorted(zip(wa, bsp), key=lambda x: x[0]))
    return ax.plot(xs, ys, **plot_kw)


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

    no_plots = len(ws_list)
    no_colors = len(colors)
    if no_plots == no_colors or no_plots < no_colors:
        ax.set_prop_cycle('color', colors)

        if show_legend:
            legend = [Line2D(
                [0], [0], color=colors[i], lw=1,
                label=f"TWS {ws_list[i]}")
                for i in range(no_plots)]
            ax.legend(handles=legend, **legend_kw)
    elif no_plots > no_colors != 2:
        if len(colors[0]) == 1:
            color_list = list(colors) + ['blue'] * (no_plots - no_colors)
            ax.set_prop_cycle('color', color_list)

            if show_legend:
                legend = [Line2D(
                    [0], [0], color=colors[i], lw=1,
                    label=f"TWS {ws_list[i]}")
                    for i in range(no_colors)]
                ax.legend(handles=legend, **legend_kw)

        if len(colors[0]) == 2:
            color_list = ['blue'] * no_plots
            for ws, c in colors:
                i = list(ws_list).index(ws)
                color_list[i] = c

            ax.set_prop_cycle('color', color_list)

            if show_legend:
                legend = [Line2D(
                    [0], [0], color=colors[i][1], lw=1,
                    label=f"TWS {colors[i][0]}")
                    for i in range(no_colors)]
                ax.legend(handles=legend, **legend_kw)

    elif no_colors == 2:
        ws_max = max(ws_list)
        ws_min = min(ws_list)
        min_color = np.array(to_rgb(colors[0]))
        max_color = np.array(to_rgb(colors[1]))
        coeffs = [(ws - ws_min) / (ws_max - ws_min)
                  for ws in ws_list]
        color_list = [(1 - coeff) * min_color + coeff * max_color
                      for coeff in coeffs]
        ax.set_prop_cycle('color', color_list)

        if show_legend:
            cmap = LinearSegmentedColormap.from_list(
                "custom_map", [min_color, max_color])
            plt.colorbar(
                ScalarMappable(norm=Normalize(
                    vmin=ws_min, vmax=ws_max), cmap=cmap),
                ax=ax, **legend_kw).set_label("True Wind Speed")

    wa_list, bsp_list = zip(*sorted(zip(wa_list, bsp_list),
                                    key=lambda x: x[0]))
    xs = np.column_stack(wa_list)
    ys = np.column_stack(bsp_list)
    return ax.plot(xs, ys, **plot_kw)


# V: In Arbeit
def flat_plot_range(ws_list, wa_list, bsp_list,
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

    no_plots = len(ws_list)
    no_colors = len(colors)
    if no_plots == no_colors or no_plots < no_colors:
        ax.set_prop_cycle('color', colors)

        if show_legend:
            legend = [Line2D(
                [0], [0], color=colors[i], lw=1,
                label=f"TWS {ws_list[i]}")
                for i in range(no_plots)]
            ax.legend(handles=legend, **legend_kw)
    elif no_plots > no_colors != 2:
        if len(colors[0]) == 1:
            color_list = list(colors) + ['blue'] * (no_plots - no_colors)
            ax.set_prop_cycle('color', color_list)

            if show_legend:
                legend = [Line2D(
                    [0], [0], color=colors[i], lw=1,
                    label=f"TWS {ws_list[i]}")
                    for i in range(no_colors)]
                ax.legend(handles=legend, **legend_kw)

        if len(colors[0]) == 2:
            color_list = ['blue'] * no_plots
            for ws, c in colors:
                i = list(ws_list).index(ws)
                color_list[i] = c

            ax.set_prop_cycle('color', color_list)

            if show_legend:
                legend = [Line2D(
                    [0], [0], color=colors[i][1], lw=1,
                    label=f"TWS {colors[i][0]}")
                    for i in range(no_colors)]
                ax.legend(handles=legend, **legend_kw)
    elif no_colors == 2:
        ws_max = max(ws_list)
        ws_min = min(ws_list)
        min_color = np.array(to_rgb(colors[0]))
        max_color = np.array(to_rgb(colors[1]))
        coeffs = [(ws - ws_min) / (ws_max - ws_min)
                  for ws in ws_list]
        color_list = [(1 - coeff) * min_color + coeff * max_color
                      for coeff in coeffs]
        ax.set_prop_cycle('color', color_list)

        if show_legend:
            cmap = LinearSegmentedColormap.from_list(
                "custom_cmap", [min_color, max_color])
            plt.colorbar(
                ScalarMappable(norm=Normalize(
                    vmin=ws_min, vmax=ws_max), cmap=cmap),
                ax=ax, **legend_kw).set_label("True Wind Speed")

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

    if legend_kw is None:
        legend_kw = {}

    bsp_max = max(bsp)
    bsp_min = min(bsp)
    min_color = np.array(to_rgb(colors[0]))
    max_color = np.array(to_rgb(colors[1]))
    coeffs = [(b-bsp_min) / (bsp_max-bsp_min)
              for b in bsp]
    color = [(1-coeff) * min_color + coeff*max_color
             for coeff in coeffs]
    if show_legend:
        cmap = LinearSegmentedColormap.from_list(
            "custom_cmap", [min_color, max_color])
        plt.colorbar(
            ScalarMappable(norm=Normalize(
                vmin=bsp_min, vmax=bsp_max), cmap=cmap),
            ax=ax, **legend_kw).set_label("Boat Speed")

    return ax.scatter(ws, wa, c=color, marker=marker)


# V: In Arbeit
def plot3d(ws, wa, bsp, ax, **plot_kw):
    ls = plot_kw.get('linestyle') or plot_kw.get('ls')
    if ls is None:
        plot_kw["ls"] = ''
    marker = plot_kw.get('marker')
    if marker is None:
        plot_kw["marker"] = 'o'

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
    vert = sorted(convex_hull_polar(bsp.copy(), wa.copy()).vertices)
    xs = []
    ys = []
    for i in vert:
        xs.append(wa[i])
        ys.append(bsp[i])
    xs.append(xs[0])
    ys.append(ys[0])
    return ax.plot(xs, ys, **plot_kw)


def bound_filter(weights, upper, lower, strict):
    if strict:
        return (weights > lower) == (weights < upper)

    return (weights >= lower) == (weights <= upper)


def percentile_filter(weights, per):
    per = 1 - per / 100
    num = len(weights) * per
    if int(num) == num:
        return weights >= (weights[int(num)] + weights[int(num) + 1]) / 2

    return weights >= weights[int(np.ceil(num))]


def spline_interpolation(points, w_res):
    ws, wa, bsp = np.hsplit(points, 3)
    ws_res, wa_res = w_res
    spl = SmoothBivariateSpline(ws, wa, bsp)
    # spl = bisplrep(ws, wa, bsp)
    # return bisplev(ws_res, wa_res, spl).T
    # d_points, val = np.hsplit(points, [2])
    ws_res, wa_res = np.meshgrid(ws_res, wa_res)
    ws_res = ws_res.reshape(-1, )
    wa_res = wa_res.reshape(-1, )
    # return griddata(d_points, val, (ws_res, wa_res), 'nearest').T
    return spl.ev(ws_res, wa_res)
