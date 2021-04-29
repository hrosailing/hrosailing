import csv
import matplotlib.pyplot as plt
from collections import Iterable
from matplotlib.colors import to_rgb, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.interpolate import bisplrep, bisplev
from scipy.spatial import ConvexHull
from _exceptions import PolarDiagramException
from _sailing_units import *


# V: Soweit in Ordnung
def polar_to_kartesian(radians, angles):
    return np.column_stack((radians * np.cos(angles),
                            radians * np.sin(angles)))


# V: Soweit in Ordnung
def convex_hull_polar(points_radians, points_angles):
    converted_points = polar_to_kartesian(points_radians, points_angles)
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
def read_curve(csv_reader):
    f = next(csv_reader)[1]
    rad = next(csv_reader)[1]
    params = next(csv_reader)[1:]
    return f, rad, params


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
def convert_wind(w_dict, tws, twa):
    if tws and twa:
        return w_dict
    if not tws:
        w_dict["wind_speed"] = apparent_wind_speed_to_true(
            w_dict["wind_speed"])
    if not twa:
        w_dict["wind_angle"] = apparent_wind_angle_to_true(
            w_dict["wind_angle"])

    return w_dict


# V: In Arbeit
def speed_resolution(ws_res):
    if ws_res is not None:
        if isinstance(ws_res, Iterable):
            return np.array(list(ws_res))
        elif isinstance(ws_res, (int, float)):
            return np.array(np.arange(ws_res, 40, ws_res))
        else:
            raise PolarDiagramException("Wrong resolution", type(ws_res))
    else:
        return np.array(np.arange(2, 42, 2))


# V: In Arbeit
def angle_resolution(wa_res):
    if wa_res is not None:
        if isinstance(wa_res, Iterable):
            return np.array(list(wa_res))
        elif isinstance(wa_res, (int, float)):
            return np.array(np.arange(wa_res, 360, wa_res))
        else:
            raise PolarDiagramException("Wrong resolution", type(wa_res))
    else:
        return np.array(np.arange(0, 360, 5))


# V: In Arbeit
def get_indices(w_list, res_list):
    if not isinstance(w_list, Iterable):
        try:
            ind = list(res_list).index(w_list)
            return ind
        except ValueError:
            raise PolarDiagramException(
                "Not in resolution",
                w_list,
                res_list)

    if not set(w_list).issubset(set(res_list)):
        raise PolarDiagramException(
            "Not in resolution",
            w_list,
            res_list)

    ind_list = [i for i in range(len(res_list)) if res_list[i] in w_list]
    return ind_list


# V: Soweit in Ordnung
def plot_polar(wa, bsp, ax, **plot_kw):
    if "linestyle" not in plot_kw and "ls" not in plot_kw:
        plot_kw["ls"] = ''
    if "marker" not in plot_kw:
        plot_kw["marker"] = 'o'

    if not ax:
        ax = plt.gca(projection='polar')

    xs, ys = zip(*sorted(zip(wa, bsp), key=lambda x: x[0]))

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    return ax.plot(xs, ys, **plot_kw)


# V: Soweit in Ordnung
def plot_flat(wa, bsp, ax, **plot_kw):
    if "linestyle" not in plot_kw and "ls" not in plot_kw:
        plot_kw["ls"] = ''
    if "marker" not in plot_kw:
        plot_kw["marker"] = 'o'

    if not ax:
        ax = plt.gca()

    xs, ys = zip(*sorted(zip(wa, bsp), key=lambda x: x[0]))

    ax.set_xlabel("True Wind Angle")
    ax.set_ylabel("Boat Speed")
    return ax.plot(xs, ys, **plot_kw)


# V: In Arbeit
def plot_polar_range(ws_list, wa_list, bsp_list,
                     ax, colors, show_legend, legend_kw, **plot_kw):
    if "linestyle" not in plot_kw and "ls" not in plot_kw:
        plot_kw["ls"] = ''
    if "marker" not in plot_kw:
        plot_kw["marker"] = 'o'
    if "color" in plot_kw or "c" in plot_kw:
        try:
            del plot_kw["color"]
        except KeyError:
            del plot_kw["c"]

    if not ax:
        ax = plt.gca(projection='polar')

    wa_list, bsp_list = zip(*sorted(zip(wa_list, bsp_list),
                                    key=lambda x: x[0]))
    xs = np.column_stack(wa_list)
    ys = np.column_stack(bsp_list)

    no_plots = len(ws_list)
    no_colors = len(colors)

    if no_plots == no_colors or no_plots < no_colors:
        ax.set_prop_cycle('color', colors)
        if show_legend:
            legend = [Line2D(
                [0], [0], color=colors[i], lw=1,
                label=f"TWS {ws_list[i]}")
                      for i in range(no_plots)]
            if legend_kw is None:
                legend_kw = {}

            ax.legend(handles=legend, **legend_kw)
    elif no_plots > no_colors != 2:
        if len(colors[0]) == 1:
            if show_legend:
                legend = [Line2D(
                    [0], [0], color=colors[i], lw=1,
                    label=f"TWS {ws_list[i]}")
                          for i in range(no_colors)]
                if legend_kw is None:
                    legend_kw = {}

                ax.legend(handles=legend, **legend_kw)

            color_list = list(colors) + ['blue']*(no_plots - no_colors)
            ax.set_prop_cycle('color', color_list)

        if len(colors[0]) == 2:
            if show_legend:
                legend = [Line2D(
                    [0], [0], color=colors[i][1], lw=1,
                    label=f"TWS {colors[i][0]}")
                          for i in range(no_colors)]
                if legend_kw is None:
                    legend_kw = {}

                ax.legend(handles=legend, **legend_kw)

            color_list = ['blue'] * no_plots
            for ws, c in colors:
                i = list(ws_list).index(ws)
                color_list[i] = c

            ax.set_prop_cycle('color', color_list)

    elif no_colors == 2:
        ws_max = max(ws_list)
        ws_min = min(ws_list)
        min_color = np.array(to_rgb(colors[0]))
        max_color = np.array(to_rgb(colors[1]))
        coeffs = [(ws - ws_min) / (ws_max - ws_min)
                  for ws in ws_list]
        color_list = [(1-coeff)*min_color + coeff*max_color
                      for coeff in coeffs]
        ax.set_prop_cycle('color', color_list)

        if show_legend:
            if legend_kw is None:
                legend_kw = {}

            cmap = LinearSegmentedColormap.from_list(
                "custom_map", [min_color, max_color])
            plt.colorbar(
                ScalarMappable(norm=Normalize(
                    vmin=ws_min, vmax=ws_max), cmap=cmap),
                ax=ax, **legend_kw).set_label("True Wind Speed")

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    return ax.plot(xs, ys, **plot_kw)


# V: In Arbeit
def flat_plot_range(ws_list, wa_list, bsp_list,
                    ax, colors, show_legend, legend_kw, **plot_kw):
    if "linestyle" not in plot_kw and "ls" not in plot_kw:
        plot_kw["ls"] = ''
    if "marker" not in plot_kw:
        plot_kw["marker"] = 'o'
    if "color" in plot_kw or "c" in plot_kw:
        try:
            del plot_kw["color"]
        except KeyError:
            del plot_kw["c"]

    if not ax:
        ax = plt.gca()

    wa_list, bsp_list = zip(*sorted(zip(wa_list, bsp_list),
                                    key=lambda x: x[0]))
    xs = np.column_stack(wa_list)
    ys = np.column_stack(bsp_list)

    no_plots = len(ws_list)
    no_colors = len(colors)

    if no_plots == no_colors or no_plots < no_colors:
        ax.set_prop_cycle('color', colors)
        if show_legend:
            legend = [Line2D(
                [0], [0], color=to_rgb(colors[i]), lw=1,
                label=f"TWS {ws_list[i]}")
                      for i in range(len(ws_list))]

            if legend_kw is None:
                legend_kw = {}

            ax.legend(handles=legend, **legend_kw)
    elif no_plots > no_colors != 2:
        if len(colors[0]) == 1:
            if show_legend:
                legend = [Line2D(
                    [0], [0], color=colors[i], lw=1,
                    label=f"TWS {ws_list[i]}")
                          for i in range(no_colors)]
                if legend_kw is None:
                    legend_kw = {}

                ax.legend(handles=legend, **legend_kw)

            color_list = list(colors) + ['blue']*(no_plots - no_colors)
            ax.set_prop_cycle('color', color_list)

        if len(colors[0]) == 2:
            if show_legend:
                legend = [Line2D(
                    [0], [0], color=colors[i][1], lw=1,
                    label=f"TWS {colors[i][0]}")
                          for i in range(no_colors)]
                if legend_kw is None:
                    legend_kw = {}

                ax.legend(handles=legend, **legend_kw)

            color_list = ['blue'] * no_plots
            for ws, c in colors:
                i = list(ws_list).index(ws)
                color_list[i] = c

            ax.set_prop_cycle('color', color_list)
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
            if legend_kw is None:
                legend_kw = {}

            cmap = LinearSegmentedColormap.from_list(
                "custom_cmap", [min_color, max_color])
            plt.colorbar(
                ScalarMappable(norm=Normalize(
                    vmin=ws_min, vmax=ws_max), cmap=cmap),
                ax=ax, **legend_kw).set_label("True Wind Speed")

    ax.set_xlabel("True Wind Angle")
    ax.set_ylabel("Boat Speed")
    return ax.plot(xs, ys, **plot_kw)


# V: Soweit in Ordnung
def plot_color(ws, wa, bsp, ax, colors,
               marker, show_legend, **legend_kw):
    if not ax:
        ax = plt.gca()

    bsp_max = max(bsp)
    bsp_min = min(bsp)
    min_color = np.array(to_rgb(colors[0]))
    max_color = np.array(to_rgb(colors[1]))
    coeffs = [(b - bsp_min)/(bsp_max - bsp_min) for b in bsp]
    color = [(1-coeff)*min_color + coeff*max_color for coeff in coeffs]
    ax.set_xlabel("True Wind Speed")
    ax.set_ylabel("True Wind Angle")

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
    if "linestyle" not in plot_kw and "ls" not in plot_kw:
        plot_kw["ls"] = ''
    if "marker" not in plot_kw:
        plot_kw["marker"] = 'o'

    if not ax:
        ax = plt.gca(projection='3d')

    ax.set_xlabel("True Wind Speed")
    ax.set_ylabel("True Wind Angle")
    ax.set_zlabel("Boat Speed")
    return ax.plot(ws, wa, bsp, **plot_kw)


# V: In Arbeit
def plot_surface(ws, wa, bsp, ax, colors):
    if not ax:
        ax = plt.gca(projection='3d')

    ax.set_xlabel("True Wind Speed")
    ax.set_ylabel("True Wind Angle")
    ax.set_zlabel("Boat Speed")
    cmap = LinearSegmentedColormap.from_list("custom_cmap", list(colors))
    color = cmap((ws - ws.min()) / float((ws - ws.min()).max()))
    return ax.plot_surface(ws, wa, bsp, facecolors=color)


# V: Soweit in Ordnung
def plot_convex_hull(wa, bsp, ax, **plot_kw):
    if not ax:
        ax = plt.gca(projection='polar')

    wa, bsp = zip(*sorted(zip(wa, bsp), key=lambda x: x[0]))
    vert = sorted(convex_hull_polar(bsp.copy(), wa.copy()).vertices)
    xs = []
    ys = []
    for i in vert:
        xs.append(wa[i])
        ys.append(bsp[i])
    xs.append(xs[0])
    ys.append(ys[0])

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    return ax.plot(xs, ys, **plot_kw)


def bound_filter(weights, upper, lower, strict):
    if strict:
        f_arr_l = weights > lower
        f_arr_u = weights < upper
    else:
        f_arr_l = weights >= lower
        f_arr_u = weights <= upper

    return f_arr_l == f_arr_u


def percentile_filter(weights, per):
    per = 1 - per/100
    num = len(weights) * per
    if int(num) == num:
        bound = (weights[num] + weights[num + 1]) / 2
    else:
        bound = weights[np.ceil(num)]

    return weights >= bound


def spline_interpolation(points, w_res):
    ws, wa, bsp = np.hsplit(points, 3)
    ws_res, wa_res = np.hsplit(w_res, 2)
    ws_res, wa_res = ws_res.reshape(-1,), wa_res.reshape(-1)
    spline = bisplrep(ws, wa, bsp)
    return bisplev(ws_res, wa_res, spline)

