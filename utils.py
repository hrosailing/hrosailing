"""
Small utility functions used throughout the module
"""

# Author: Valentin F. Dannenberg / Ente


import numpy as np

from collections.abc import Iterable
from scipy.spatial import ConvexHull

from windconversion import apparent_wind_to_true

# TODO: Error checks?


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


def speed_resolution(ws_res):

    if ws_res is None:
        return np.arange(2, 42, 2)

    if not isinstance(ws_res, (Iterable, int, float)):
        raise ValueError(
            f"{ws_res} is neither Iterable, "
            f"int or float")

    if isinstance(ws_res, Iterable):
        ws_res = np.asarray(ws_res)
        if not ws_res.size:
            # TODO: Also just return
            #       default res?
            raise ValueError(
                "Empty ws_res was passed")
        return ws_res

    if ws_res <= 0:
        raise ValueError(
            "Negative resolution stepsize"
            "is not supported")

    return np.arange(ws_res, 40, ws_res)


def angle_resolution(wa_res):
    if wa_res is None:
        return np.arange(0, 360, 5)

    if not isinstance(wa_res, (Iterable, int, float)):
        raise ValueError(
            f"{wa_res} is neither Iterable, "
            f"int or float")

    if isinstance(wa_res, Iterable):
        wa_res = np.asarray(wa_res)
        if not wa_res.size:
            # TODO: Also just return
            #       default res?
            raise ValueError(
                "Empty wa_res was passed")
        return wa_res

    if wa_res <= 0:
        raise ValueError(
            "Negative resolution stepsize"
            "is not supported")

    return np.arange(wa_res, 360, wa_res)
