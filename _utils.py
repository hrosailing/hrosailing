import csv
import matplotlib.pyplot as plt
from collections import Iterable
from matplotlib.colors import to_rgb
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
def plot_polar(wind_angles, boat_speeds, ax, **kwargs):
    if "linestyle" not in kwargs and "ls" not in kwargs:
        kwargs["ls"] = ''
    if "marker" not in kwargs:
        kwargs["marker"] = 'o'

    if not ax:
        ax = plt.gca(projection='polar')

    wind_angles, boat_speeds = zip(*sorted(zip(wind_angles, boat_speeds),
                                           key=lambda x: x[0]))

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    return ax.plot(wind_angles, boat_speeds, **kwargs)


# V: Soweit in Ordnung
def plot_flat(wind_angles, boat_speeds, ax, **kwargs):
    if "linestyle" not in kwargs and "ls" not in kwargs:
        kwargs["ls"] = ''
    if "marker" not in kwargs:
        kwargs["marker"] = 'o'

    if not ax:
        ax = plt.gca()

    wind_angles, boat_speeds = zip(*sorted(zip(wind_angles, boat_speeds),
                                           key=lambda x: x[0]))

    ax.set_xlabel("True Wind Angle")
    ax.set_ylabel("Boat Speed")
    return ax.plot(wind_angles, boat_speeds, **kwargs)


# V: In Arbeit
def plot_polar_range(wind_speeds_list, wind_angles_list, boat_speeds_list,
                     ax, colors, **kwargs):
    if "linestyle" not in kwargs and "ls" not in kwargs:
        kwargs["ls"] = ''
    if "marker" not in kwargs:
        kwargs["marker"] = 'o'
    if "color" in kwargs or "c" in kwargs:
        try:
            del kwargs["color"]
        except KeyError:
            del kwargs["c"]

    if not ax:
        ax = plt.gca(projection='polar')

    wind_angles_list, boat_speeds_list = zip(*sorted(zip(wind_angles_list, boat_speeds_list),
                                                     key=lambda x: x[0]))
    wind_angles = np.column_stack(wind_angles_list)
    boat_speeds = np.column_stack(boat_speeds_list)

    if len(wind_angles_list) == len(colors) or (len(wind_angles_list) > len(colors) > 2):
        ax.set_prop_cycle('color', colors)
    elif len(wind_angles_list) < len(colors):
        ax.set_prob_cycle('color', colors[:len(wind_angles_list)])
    elif len(colors) == 2:
        w_max = max(wind_speeds_list)
        w_min = min(wind_speeds_list)
        min_color = np.array(to_rgb(colors[0]))
        max_color = np.array(to_rgb(colors[1]))
        coeffs = [(w_val - w_min) / (w_max - w_min) for w_val in wind_speeds_list]
        color_list = [(1-coeff) * min_color + coeff * max_color for coeff in coeffs]
        ax.set_prop_cycle('color', color_list)
    elif len(colors) == 1:
        kwargs["c"] = to_rgb(colors[0])

    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    return ax.plot(wind_angles, boat_speeds, **kwargs)


# V: In Arbeit
def flat_plot_range(wind_speeds_list, wind_angles_list, boat_speeds_list,
                    ax, colors, **kwargs):
    if "linestyle" not in kwargs and "ls" not in kwargs:
        kwargs["ls"] = ''
    if "marker" not in kwargs:
        kwargs["marker"] = 'o'
    if "color" in kwargs or "c" in kwargs:
        try:
            del kwargs["color"]
        except KeyError:
            del kwargs["c"]

    if not ax:
        ax = plt.gca()

    wind_angles_list, boat_speeds_list = zip(*sorted(zip(wind_angles_list, boat_speeds_list),
                                                     key=lambda x: x[0]))
    wind_angles = np.column_stack(wind_angles_list)
    boat_speeds = np.column_stack(boat_speeds_list)

    if len(wind_angles_list) == len(colors) or (len(wind_angles_list) > len(colors) > 2):
        ax.set_prop_cycle('color', colors)
    elif len(wind_angles_list) < len(colors):
        ax.set_prob_cycle('color', colors[:len(wind_angles_list)])
    elif len(colors) == 2:
        w_max = max(wind_speeds_list)
        w_min = min(wind_speeds_list)
        min_color = np.array(to_rgb(colors[0]))
        max_color = np.array(to_rgb(colors[1]))
        coeffs = [(w_val - w_min) / (w_max - w_min) for w_val in wind_speeds_list]
        color_list = [(1-coeff) * min_color + coeff * max_color for coeff in coeffs]
        ax.set_prop_cycle('color', color_list)
    elif len(colors) == 1:
        kwargs["c"] = to_rgb(colors[0])

    ax.set_xlabel("True Wind Angle")
    ax.set_ylabel("Boat Speed")
    return ax.plot(wind_angles, boat_speeds, **kwargs)


# V: In Arbeit
def plot_color(wind_speeds, wind_angles, boat_speeds, ax, min_color, max_color, marker):
    if not ax:
        ax = plt.gca()

    z_max = max(boat_speeds)
    z_min = min(boat_speeds)
    min_color = np.array(to_rgb(min_color))
    max_color = np.array(to_rgb(max_color))
    coeffs = [(z_val - z_min)/(z_max - z_min) for z_val in boat_speeds]
    color = [(1-coeff) * min_color + coeff*max_color for coeff in coeffs]
    ax.set_xlabel("True Wind Speed")
    ax.set_ylabel("True Wind Angle")
    return ax.scatter(wind_speeds, wind_angles, c=color, marker=marker)


# V: Soweit in Ordnung
def plot_convex_hull(angles, speeds, ax, **kwargs):
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
    return ax.plot(wind_angles, boat_speeds, **kwargs)
