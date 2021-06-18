"""
Small utility functions used throughout the module
"""

# Author: Valentin F. Dannenberg / Ente


import numpy as np

from collections.abc import Iterable
from scipy.spatial import ConvexHull

from exceptions import PolarDiagramException
from windconversion import apparent_wind_to_true


def polar_to_kartesian(arr):
    return np.column_stack(
        (arr[:, 1] * np.cos(arr[:, 0]),
         arr[:, 1] * np.sin(arr[:, 0])))


def euclidean_norm(vec):
    return np.linalg.norm(vec, axis=1)


def convex_hull_polar(points):
    converted_points = polar_to_kartesian(points)
    return ConvexHull(converted_points)


def convert_wind(wind_arr, tw):
    if tw:
        return wind_arr

    return apparent_wind_to_true(wind_arr)


# TODO: Make it cleaner!
def speed_resolution(ws_res):
    if ws_res is None:
        return np.array(np.arange(2, 42, 2))

    if not isinstance(ws_res, (Iterable, int, float)):
        raise PolarDiagramException(
            f"{ws_res} is neither Iterable, int or float")

    if isinstance(ws_res, Iterable):
        if isinstance(ws_res, np.ndarray):
            if not ws_res.size:
                raise PolarDiagramException("Empty ws_res was passed")
        else:
            if not ws_res:
                raise PolarDiagramException("Empty ws_res was passed")

        return np.asarray(ws_res)

    if ws_res <= 0:
        raise PolarDiagramException("")

    return np.array(np.arange(ws_res, 40, ws_res))


# TODO: Make it cleaner
def angle_resolution(wa_res):
    if wa_res is None:
        return np.array(np.arange(0, 360, 5))

    if not isinstance(wa_res, (Iterable, int, float)):
        raise PolarDiagramException(
            f"{wa_res} is neither Iterable, int or float")

    if isinstance(wa_res, Iterable):
        if isinstance(wa_res, np.ndarray):
            if not wa_res.size:
                raise PolarDiagramException("Empty ws_res was passed")
        else:
            if not wa_res:
                raise PolarDiagramException("Empty ws_res was passed")

        return np.asarray(wa_res)

    if wa_res <= 0:
        raise PolarDiagramException("")

    return np.array(np.arange(wa_res, 360, wa_res))


# TODO: Make it cleaner!
def get_indices(w_list, res_list):
    if w_list is None:
        return list(range(len(res_list)))

    if not isinstance(w_list, Iterable):
        try:
            ind = list(res_list).index(w_list)
            return [ind]
        except ValueError:
            raise PolarDiagramException(
                f"{w_list} is not contained "
                f"in {res_list}")

    if isinstance(w_list, np.ndarray):
        if not w_list.size:
            raise PolarDiagramException("")
    else:
        if not w_list:
            raise PolarDiagramException("")

    if not set(w_list).issubset(set(res_list)):
        raise PolarDiagramException(
            f"{w_list} is not a subset of {res_list}")

    ind_list = [i for i, w in enumerate(res_list) if w in w_list]
    return ind_list
