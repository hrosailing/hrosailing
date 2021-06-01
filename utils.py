import numpy as np
from collections import Iterable
from scipy.spatial import ConvexHull
from exceptions import PolarDiagramException
from windconversion import apparent_wind_to_true


def polar_to_kartesian(arr):
    return np.column_stack(
        (arr[:, 1] * np.cos(arr[:, 0]),
         arr[:, 1] * np.sin(arr[:, 0]))
    )


def convex_hull_polar(points):
    converted_points = polar_to_kartesian(points)
    return ConvexHull(converted_points)


def convert_wind(w_dict, tw):
    if tw:
        return w_dict

    aws = w_dict.get("wind_speed")
    awa = w_dict.get("wind_angle")
    bsp = w_dict.get("boat_speed")

    tws, twa = apparent_wind_to_true(aws, awa, bsp)

    return {"wind_speed": tws, "wind_angle": twa}


def speed_resolution(ws_res):
    if ws_res is None:
        return np.array(np.arange(2, 42, 2))

    if not isinstance(ws_res, (Iterable, int, float)):
        raise PolarDiagramException(
            "ws_res is neither Iterable, int or float")

    if isinstance(ws_res, Iterable):
        return np.array(list(ws_res))

    return np.array(np.arange(ws_res, 40, ws_res))


def angle_resolution(wa_res):
    if wa_res is None:
        return np.array(np.arange(0, 360, 5))

    if not isinstance(wa_res, (Iterable, int, float)):
        raise PolarDiagramException(
            "wa_res is neither Iterable, int or float")

    if isinstance(wa_res, Iterable):
        return np.array(list(wa_res))

    return np.array(np.arange(wa_res, 360, wa_res))


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
