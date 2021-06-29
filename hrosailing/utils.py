"""
Small utility functions used throughout the module
"""

# Author: Valentin F. Dannenberg / Ente


import numpy as np

from typing import Iterable

# TODO: Error checks?


def polar_to_kartesian(arr):
    return np.column_stack(
        (arr[:, 1] * np.cos(arr[:, 0]),
         arr[:, 1] * np.sin(arr[:, 0])))


def euclidean_norm(vec):
    return np.linalg.norm(vec, axis=1)


def speed_resolution(ws_res):
    if ws_res is None:
        return np.arange(2, 42, 2)

    if not isinstance(ws_res, (Iterable, int, float)):
        raise ValueError(
            f"{ws_res} is neither Iterable, "
            f"int or float")

    # TODO: Maybe not all iterables valid?
    if isinstance(ws_res, Iterable):
        # TODO: Check if elements of
        #       iterable are of type
        #       Number

        # Edge case, if ws_res is a set or dict
        # since numpy.ndarrays don't behave
        # as desired, when constructed from
        # a set or dict
        if isinstance(ws_res, (set, dict)):
            raise ValueError("")

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

    # TODO: Maybe not all Iterables valid?
    if isinstance(wa_res, Iterable):
        # TODO: Check if elements of
        #       iterable are of type
        #       Number

        # Edge case, if wa_res is a set or dict
        # since numpy.ndarrays don't behave
        # as desired, when constructed from
        # a set or dict
        if isinstance(wa_res, (set, dict)):
            raise ValueError("")

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
