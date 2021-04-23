import csv
import matplotlib.pyplot as plt
from collections import Iterable
from matplotlib.colors import to_rgb, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from _exceptions import *
from _sailing_units import *


# V: Soweit in Ordnung
def polar_to_kartesian(radians, angles):
    return np.column_stack((radians * np.cos(angles), radians * np.sin(angles)))


# V: Soweit in Ordnung
def convex_hull_polar(points_radians, points_angles):
    converted_points = polar_to_kartesian(points_radians, points_angles)
    return ConvexHull(converted_points)


# V: In Arbeit
def read_table(csv_reader):
    data = []
    next(csv_reader)
    wind_speed_resolution = [eval(s) for s in next(csv_reader)]
    if len(wind_speed_resolution) == 1:
        wind_speed_resolution = wind_speed_resolution[0]
    next(csv_reader)
    wind_angle_resolution = [eval(a) for a in next(csv_reader)]
    if len(wind_angle_resolution) == 1:
        wind_angle_resolution = wind_angle_resolution[0]
    next(csv_reader)
    for row in csv_reader:
        data.append([eval(entry) for entry in row])

    return wind_speed_resolution, wind_angle_resolution, data


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
        wind_speed_resolution = [eval(s) for s in next(csv_reader)[1:]]
        wind_angle_resolution = []
        data = []
        next(csv_reader)
        for row in csv_reader:
            wind_angle_resolution.append(eval(row[0]))
            data.append([eval(d) for d in row[1:]])
        return wind_speed_resolution, wind_angle_resolution, data


# V: In Arbeit
def read_array_csv(csv_path):
    file_data = np.genfromtxt(csv_path, delimiter="\t")
    return file_data[0, 1:], file_data[1:, 0], file_data[1:, 1:]


# V: In Arbeit
def read_opencpn_csv(csv_path):
    with open(csv_path, 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=',', quotechar='"')
        wind_speed_resolution = [eval(s) for s in next(csv_reader)[1:]]
        wind_angle_resolution = []
        data = []
        next(csv_reader)
        for row in csv_reader:
            wind_angle_resolution.append(eval(row[0]))
            data.append([eval(d) if d != '' else 0 for d in row[1:]])

        return wind_speed_resolution, wind_angle_resolution, data


# V: In Arbeit
def create_wind_dict(points):
    wind_dict = {"wind_speed": points[:, 0],
                 "wind_angle": points[:, 1]}
    return wind_dict


# V: In Arbeit
def convert_wind(wind_dict, tws, twa):
    if tws and twa:
        return wind_dict
    if not twa:
        wind_dict["wind_speed"] = apparent_wind_speed_to_true(wind_dict["wind_speed"])
    if not tws:
        wind_dict["wind_angle"] = apparent_wind_angle_to_true(wind_dict["wind_angle"])

    return wind_dict


# V: In Arbeit
def speed_resolution(wind_speed_resolution):
    if wind_speed_resolution is not None:
        if isinstance(wind_speed_resolution, Iterable):
            return np.array(list(wind_speed_resolution))
        elif isinstance(wind_speed_resolution, (int, float)):
            return np.array(np.arange(wind_speed_resolution, 40, wind_speed_resolution))
        else:
            raise PolarDiagramException("Wrong resolution", type(wind_speed_resolution))
    else:
        return np.array(np.arange(2, 42, 2))


# V: In Arbeit
def angle_resolution(wind_angle_resolution):
    if wind_angle_resolution is not None:
        if isinstance(wind_angle_resolution, Iterable):
            return np.array(list(wind_angle_resolution))
        elif isinstance(wind_angle_resolution, (int, float)):
            return np.array(np.arange(wind_angle_resolution, 360, wind_angle_resolution))
        else:
            raise PolarDiagramException("Wrong resolution", type(wind_angle_resolution))
    else:
        return np.array(np.arange(0, 360, 5))


# V: In Arbeit
def get_indices(wind_list, resolution):
    if not isinstance(wind_list, Iterable):
        try:
            ind = list(resolution).index(wind_list)
            return ind
        except ValueError:
            raise PolarDiagramException("Not in resolution", wind_list, resolution)

    if not set(wind_list).issubset(set(resolution)):
        raise PolarDiagramException("Not in resolution", wind_list, resolution)

    ind_list = [i for i in range(len(resolution))
                if resolution[i] in wind_list]
    return ind_list


# V: Soweit in Ordnung
def plot_polar(wind_angles, boat_speeds, ax, **plot_kw):
    if "linestyle" not in plot_kw and "ls" not in plot_kw:
        plot_kw["ls"] = ''
    if "marker" not in plot_kw:
        plot_kw["marker"] = 'o'

    if not ax:
        ax = plt.gca(projection='polar')

    wind_angles, boat_speeds = zip(*sorted(zip(wind_angles, boat_speeds),
                                           key=lambda x: x[0]))

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    return ax.plot(wind_angles, boat_speeds, **plot_kw)


# V: Soweit in Ordnung
def plot_flat(wind_angles, boat_speeds, ax, **plot_kw):
    if "linestyle" not in plot_kw and "ls" not in plot_kw:
        plot_kw["ls"] = ''
    if "marker" not in plot_kw:
        plot_kw["marker"] = 'o'

    if not ax:
        ax = plt.gca()

    wind_angles, boat_speeds = zip(*sorted(zip(wind_angles, boat_speeds),
                                           key=lambda x: x[0]))

    ax.set_xlabel("True Wind Angle")
    ax.set_ylabel("Boat Speed")
    return ax.plot(wind_angles, boat_speeds, **plot_kw)


# V: In Arbeit
def plot_polar_range(wind_speeds_list, wind_angles_list, boat_speeds_list,
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

    wind_angles_list, boat_speeds_list = zip(*sorted(zip(wind_angles_list, boat_speeds_list),
                                                     key=lambda x: x[0]))
    wind_angles = np.column_stack(wind_angles_list)
    boat_speeds = np.column_stack(boat_speeds_list)

    no_plots = len(wind_speeds_list)
    no_colors = len(colors)

    if no_plots == no_colors or no_plots < no_colors:
        ax.set_prop_cycle('color', colors)
        if show_legend:
            legend = [Line2D([0], [0], color=colors[i], lw=1,
                             label=f"TWS {wind_speeds_list[i]}")
                      for i in range(no_plots)]
            if legend_kw is None:
                legend_kw = {}

            ax.legend(handles=legend, **legend_kw)
    elif no_plots > no_colors != 2 or len(colors[0]) == 2:
        if len(colors[0]) == 1:
            if show_legend:
                legend = [Line2D([0], [0], color=colors[i], lw=1,
                                 label=f"TWS {wind_speeds_list[i]}")
                          for i in range(no_colors)]
                if legend_kw is None:
                    legend_kw = {}

                ax.legend(handles=legend, **legend_kw)

            ax.set_prop_cycle('color', colors + ['blue'] * (no_plots - no_colors))

        if len(colors[0]) == 2:
            if show_legend:
                legend = [Line2D([0], [0], color=colors[i][1], lw=1,
                                 label=f"TWS {colors[i][0]}")
                          for i in range(no_colors)]
                if legend_kw is None:
                    legend_kw = {}

                ax.legend(handles=legend, **legend_kw)

            color_list = ['blue'] * no_plots
            for wind_speed, color in colors:
                i = list(wind_speeds_list).index(wind_speed)
                color_list[i] = color

            ax.set_prop_cycle('color', color_list)

    elif no_colors == 2:
        w_max = max(wind_speeds_list)
        w_min = min(wind_speeds_list)
        min_color = np.array(to_rgb(colors[0]))
        max_color = np.array(to_rgb(colors[1]))
        coeffs = [(w_val - w_min) / (w_max - w_min) for w_val in wind_speeds_list]
        color_list = [(1-coeff) * min_color + coeff * max_color for coeff in coeffs]
        ax.set_prop_cycle('color', color_list)

        if show_legend:
            if legend_kw is None:
                legend_kw = {}

            cmap = LinearSegmentedColormap.from_list("custom_map", [min_color, max_color])
            plt.colorbar(ScalarMappable(norm=Normalize(vmin=w_min, vmax=w_max), cmap=cmap),
                         ax=ax, **legend_kw).set_label("True Wind Speed")

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    return ax.plot(wind_angles, boat_speeds, **plot_kw)


# V: In Arbeit
def flat_plot_range(wind_speeds_list, wind_angles_list, boat_speeds_list,
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

    wind_angles_list, boat_speeds_list = zip(*sorted(zip(wind_angles_list, boat_speeds_list),
                                                     key=lambda x: x[0]))
    wind_angles = np.column_stack(wind_angles_list)
    boat_speeds = np.column_stack(boat_speeds_list)

    no_plots = len(wind_speeds_list)
    no_colors = len(colors)

    if no_plots == no_colors or no_plots < no_colors:
        ax.set_prop_cycle('color', colors)
        if show_legend:
            legend = [Line2D([0], [0], color=to_rgb(colors[i]), lw=1,
                             label=f"TWS {wind_speeds_list[i]}")
                      for i in range(len(wind_speeds_list))]

            if legend_kw is None:
                legend_kw = {}

            ax.legend(handles=legend, **legend_kw)
    elif no_plots > no_colors != 2 or len(colors[0]) == 2:
        if len(colors[0]) == 1:
            if show_legend:
                legend = [Line2D([0], [0], color=colors[i], lw=1,
                                 label=f"TWS {wind_speeds_list[i]}")
                          for i in range(no_colors)]
                if legend_kw is None:
                    legend_kw = {}

                ax.legend(handles=legend, **legend_kw)

            ax.set_prop_cycle('color', colors + ['blue'] * (no_plots - no_colors))

        if len(colors[0]) == 2:
            if show_legend:
                legend = [Line2D([0], [0], color=colors[i][1], lw=1,
                                 label=f"TWS {colors[i][0]}")
                          for i in range(no_colors)]
                if legend_kw is None:
                    legend_kw = {}

                ax.legend(handles=legend, **legend_kw)

            color_list = ['blue'] * no_plots
            for wind_speed, color in colors:
                i = list(wind_speeds_list).index(wind_speed)
                color_list[i] = color

            ax.set_prop_cycle('color', color_list)
    elif no_colors == 2:
        w_max = max(wind_speeds_list)
        w_min = min(wind_speeds_list)
        min_color = np.array(to_rgb(colors[0]))
        max_color = np.array(to_rgb(colors[1]))
        coeffs = [(w_val - w_min) / (w_max - w_min) for w_val in wind_speeds_list]
        color_list = [(1 - coeff) * min_color + coeff * max_color for coeff in coeffs]
        ax.set_prop_cycle('color', color_list)

        if show_legend:
            if legend_kw is None:
                legend_kw = {}

            cmap = LinearSegmentedColormap.from_list("custom_cmap", [min_color, max_color])
            plt.colorbar(ScalarMappable(norm=Normalize(vmin=w_min, vmax=w_max), cmap=cmap),
                         ax=ax, **legend_kw).set_label("True Wind Speed")

    ax.set_xlabel("True Wind Angle")
    ax.set_ylabel("Boat Speed")
    return ax.plot(wind_angles, boat_speeds, **plot_kw)


# V: Soweit in Ordnung
def plot_color(wind_speeds, wind_angles, boat_speeds, ax, colors,
               marker, show_legend, **legend_kw):
    if not ax:
        ax = plt.gca()

    b_max = max(boat_speeds)
    b_min = min(boat_speeds)
    min_color = np.array(to_rgb(colors[0]))
    max_color = np.array(to_rgb(colors[1]))
    coeffs = [(b_val - b_min)/(b_max - b_min) for b_val in boat_speeds]
    color = [(1-coeff) * min_color + coeff*max_color for coeff in coeffs]
    ax.set_xlabel("True Wind Speed")
    ax.set_ylabel("True Wind Angle")

    if show_legend:
        cmap = LinearSegmentedColormap.from_list("custom_cmap", [min_color, max_color])
        plt.colorbar(ScalarMappable(norm=Normalize(vmin=b_min, vmax=b_max), cmap=cmap),
                     ax=ax, **legend_kw).set_label("Boat Speed")

    return ax.scatter(wind_speeds, wind_angles, c=color, marker=marker)


# V: In Arbeit
def plot3d(wind_speeds, wind_angles, boat_speeds, ax, **plot_kw):
    if "linestyle" not in plot_kw and "ls" not in plot_kw:
        plot_kw["ls"] = ''
    if "marker" not in plot_kw:
        plot_kw["marker"] = 'o'

    if not ax:
        ax = plt.gca(projection='3d')

    ax.set_xlabel("True Wind Speed")
    ax.set_ylabel("True Wind Angle")
    ax.set_zlabel("Boat Speed")
    return ax.plot(wind_speeds, wind_angles, boat_speeds, **plot_kw)


# V: In Arbeit
def plot_surface(wind_speeds, wind_angles, boat_speeds, ax, colors):
    if not ax:
        ax = plt.gca(projection='3d')

    ax.set_xlabel("True Wind Speed")
    ax.set_ylabel("True Wind Angle")
    ax.set_zlabel("Boat Speed")
    cmap = LinearSegmentedColormap.from_list("custom_cmap", list(colors))
    color = cmap((wind_speeds - wind_speeds.min())/float((wind_speeds - wind_speeds.min()).max()))
    return ax.plot_surface(wind_speeds, wind_angles, boat_speeds, facecolors=color)


# V: Soweit in Ordnung
def plot_convex_hull(angles, speeds, ax, **plot_kw):
    if not ax:
        ax = plt.gca(projection='polar')

    angles, speeds = zip(*sorted(zip(angles, speeds),
                                 key=lambda x: x[0]))
    vert = sorted(convex_hull_polar(speeds.copy(), angles.copy()).vertices)
    wind_angles = []
    boat_speeds = []
    for i in vert:
        wind_angles.append(angles[i])
        boat_speeds.append(speeds[i])
    wind_angles.append(wind_angles[0])
    boat_speeds.append(boat_speeds[0])

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    return ax.plot(wind_angles, boat_speeds, **plot_kw)
