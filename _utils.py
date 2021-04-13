import csv
import matplotlib.pyplot as plt
from collections import Iterable
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
            eval_data = [eval(d) for d in row[1:]]
            data.append(eval_data)
        return wind_speed_resolution, wind_angle_resolution, data


# V: In Arbeit
def read_array_csv(csv_path):
    file_data = np.genfromtxt(csv_path, delimiter="\t")
    return file_data[0, 1:], file_data[1:, 0], file_data[1:, 1:]


# V: In Arbeit
def create_wind_dict(points):
    wind_dict = {"wind_speed": points[:, 0],
                 "wind_angle": points[:, 1]}
    return wind_dict


# V: In Arbeit
def convert_dict(wind_dict):
    if "wind_speed" in wind_dict or "wind_angle" in wind_dict:
        new_wind_dict = {"wind_speed": wind_dict["wind_speed"] if "wind_speed" in wind_dict
                                                                  else None,
                         "wind_angle": wind_dict["wind_angle"] if "wind_angle" in wind_dict
                                                                  else None}
        return new_wind_dict

    elif "wind_speed_resolution" in wind_dict or "wind_angle_resolution" in wind_dict:
        new_wind_dict = {"wind_speed": wind_dict["wind_speed_resolution"]
                         if "wind_speed_resolution" in wind_dict
                            else None,
                         "wind_angle": wind_dict["wind_angle_resolution"]
                         if "wind_angle_resolution" in wind_dict
                            else None}

        return new_wind_dict
    else:
        return {"wind_speed": None, "wind_angle": None}


# V: In Arbeit
def convert_wind(wind_dict, tws, twa):
    wind_dict = convert_dict(wind_dict)
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
        return np.array(np.arange(5, 365, 5))


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
def plot_polar(wind_angles, boat_speeds, **kwargs):
    if "linestyle" not in kwargs and "ls" not in kwargs:
        kwargs["ls"] = ''
    if "marker" not in kwargs:
        kwargs["marker"] = 'o'

    boat_speeds = [x for _, x in sorted(zip(wind_angles, boat_speeds),
                                        key=lambda pair: pair[0])]
    wind_angles.sort()
    polar_plot = plt.subplot(1, 1, 1, projection='polar')
    polar_plot.set_theta_zero_location('N')
    polar_plot.set_theta_direction('clockwise')
    return polar_plot.plot(wind_angles, boat_speeds, **kwargs)


# V: Soweit in Ordnung
def plot_flat(wind_angles, boat_speeds, **kwargs):
    if "linestyle" not in kwargs and "ls" not in kwargs:
        kwargs["ls"] = ''
    if "marker" not in kwargs:
        kwargs["marker"] = 'o'

    boat_speeds = [x for _, x in sorted(zip(wind_angles, boat_speeds),
                                        key=lambda pair: pair[0])]
    wind_angles.sort()
    flat_plot = plt.subplot(1, 1, 1)
    plt.xlabel("True Wind Angle")
    plt.ylabel("Boat Speed")
    return flat_plot.plot(wind_angles, boat_speeds, **kwargs)


# V: In Arbeit
def plot_color_g(wind_speeds, wind_angles, boat_speeds, min_color, max_color):
    z_max = max(boat_speeds)
    z_min = min(boat_speeds)
    coeffs = [(z_val - z_min)/(z_max - z_min) for z_val in boat_speeds]
    color = [coeff * min_color + (1-coeff)*max_color for coeff in coeffs]
    gradient_plot = plt.subplot(1, 1, 1)
    return gradient_plot.scatter(wind_speeds, wind_angles, c=color)


# V: Soweit in Ordnung
def plot_convex_hull(angles, speeds, **kwargs):
    speeds = [x for _, x in sorted(zip(angles, speeds),
                                   key=lambda pair: pair[0])]
    angles.sort()
    vert = sorted(convex_hull_polar(speeds.copy(), angles.copy()).vertices)
    wind_angles = []
    boat_speeds = []
    for i in vert:
        wind_angles.append(angles[i])
        boat_speeds.append(speeds[i])

    wind_angles.append(wind_angles[0])
    boat_speeds.append(boat_speeds[0])
    convex_hull_plot = plt.subplot(1, 1, 1, projection='polar')
    convex_hull_plot.set_theta_zero_location('N')
    convex_hull_plot.set_theta_direction('clockwise')
    return convex_hull_plot.plot(wind_angles, boat_speeds, **kwargs)


