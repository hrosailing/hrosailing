import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
# from _exceptions import *
from _sailing_units import apparent_wind_angle_to_true, apparent_wind_speed_to_true


# V: Soweit in Ordnung
def polar_to_kartesian(radians, angles):
    return np.column_stack((radians * np.cos(angles), radians * np.sin(angles)))


# V: Soweit in Ordnung
def convex_hull_polar(points_radians, points_angles):
    converted_points = polar_to_kartesian(points_radians, points_angles)
    return ConvexHull(converted_points)


# V: In Arbeit
def convert_wind(keyword_dict):
    if "true_wind_speed" in keyword_dict:
        true_wind_speeds = keyword_dict["true_wind_speed"]
    elif "apparent_wind_speed" in keyword_dict:
        apparent_wind_speeds = keyword_dict["apparent_wind_speed"]
        true_wind_speeds = apparent_wind_speed_to_true(apparent_wind_speeds)
    else:
        true_wind_speeds = None

    if "true_wind_angle" in keyword_dict:
        true_wind_angles = keyword_dict["true_wind_angle"]
    elif "apparent_wind_angle" in keyword_dict:
        apparent_wind_angle = keyword_dict["apparent_wind_angle"]
        true_wind_angles = apparent_wind_angle_to_true(apparent_wind_angle)
    else:
        true_wind_angles = None

    return true_wind_speeds, true_wind_angles


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
