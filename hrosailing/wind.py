"""
Functions to convert wind from apparent to true and vice versa
"""

# Author: Valentin F. Dannenberg / Ente

import numpy as np

from typing import Iterable


class WindException(Exception):
    pass


def apparent_wind_to_true(wind_arr):
    """Converts apparent wind to true wind

    Parameters
    ----------
    wind_arr : array_like
        Wind data given as a sequence of points consisting of wind speed,
        wind angle and boat speed, where the wind speed and wind angle are
        measured as apparent wind

    Returns
    -------
    out : numpy.ndarray of shape (n, 3)
        Array containing the same data as wind_arr, but the wind speed
        and wind angle now measured as true wind

    Raises a WindException
        - if wind_arr is an empty sequence
        - if some values in wind_arr are NaN or
        not finite
    """

    wind_arr = np.asarray(wind_arr)
    if wind_arr.dtype == "object":
        raise WindException("wind_arr is not array_like")
    if not wind_arr.size:
        raise WindException("Empty array was passed. Conversion not possible")
    try:
        wind_arr = wind_arr.reshape(-1, 3)
    except ValueError:
        raise WindException(
            "wind_arr could not be broadcasted to an array of shape (n,3)"
        )
    if not np.isfinite(wind_arr):
        raise WindException("All values have to be finite and not NaN")

    # TODO Maybe not split the array here?
    aws, awa, bsp = np.hsplit(wind_arr, 3)

    awa_above_180 = awa > 180
    awa = np.deg2rad(awa)

    tws = np.sqrt(
        np.square(aws) + np.square(bsp) - 2 * aws * bsp * np.cos(awa)
    )

    temp = (aws * np.cos(awa) - bsp) / tws
    # Account for computer error
    # Why necessary?
    temp[temp > 1] = 1
    temp[temp < -1] = -1

    twa = np.arccos(temp)
    twa[awa_above_180] = 360 - np.rad2deg(twa[awa_above_180])
    twa[np.logical_not(awa_above_180)] = np.rad2deg(
        twa[np.logical_not(awa_above_180)]
    )

    return np.column_stack((tws, twa, bsp))


def true_wind_to_apparent(wind_arr):
    """Converts true wind to apparent wind

        Parameters
        ----------
        wind_arr : array_like
            Wind data given as a sequence of points consisting of wind speed,
            wind angle and boat speed, where the wind speed and wind angle are
            measured as true wind

        Returns
        -------
        out : numpy.ndarray of shape (n, 3)
            Array containing the same data as wind_arr, but the wind speed
            and wind angle now measured as apparent wind

        Raises a WindException
            - if wind_arr is an empty sequence
            - if some values in wind_arr are NaN or
            not finite
        """

    wind_arr = np.asarray(wind_arr)
    if wind_arr.dtype == "object":
        raise WindException("wind_arr is not array_like")
    if not wind_arr.size:
        raise WindException("Empty array passed. Conversion not possible")
    try:
        wind_arr = wind_arr.reshape(-1, 3)
    except ValueError:
        raise WindException(
            "wind_arr could not be broadcasted to an array of shape (n,3)"
        )
    if not np.isfinite(wind_arr):
        raise WindException("All values have to be finite and not NaN")

    # TODO Maybe not split the array here?
    tws, twa, bsp = np.hsplit(wind_arr, 3)

    twa_above_180 = twa > 180
    twa_rad = np.deg2rad(twa)

    aws = np.sqrt(pow(tws, 2) + pow(bsp, 2) + 2 * tws * bsp * np.cos(twa_rad))

    temp = (tws * np.cos(twa_rad) + bsp) / aws
    # Account for computer error
    # Why necessary?
    temp[temp > 1] = 1
    temp[temp < -1] = -1

    awa = np.arccos(temp)
    awa[twa_above_180] = 360 - np.rad2deg(awa[twa_above_180])
    awa[np.logical_not(twa_above_180)] = np.rad2deg(
        awa[np.logical_not(twa_above_180)]
    )

    return np.column_stack((tws, twa, bsp))


def speed_resolution(ws_res):
    if ws_res is None:
        return np.arange(2, 42, 2)

    # Iterable-test really necessary?
    if not isinstance(ws_res, (Iterable, int, float)):
        raise WindException(f"{ws_res} is neither array_like, int or float")

    if isinstance(ws_res, Iterable):
        # TODO: Check if contents of
        #       array are numbers?
        ws_res = np.asarray(ws_res)

        if ws_res.dtype == object:
            raise WindException(f"{ws_res} is not array_like")
        if not ws_res.size:
            raise WindException("Empty ws_res was passed")

        return ws_res

    if ws_res <= 0:
        raise WindException("Nonpositive resolution stepsize")

    return np.arange(ws_res, 40, ws_res)


def angle_resolution(wa_res):
    if wa_res is None:
        return np.arange(0, 360, 5)

    # Iterable-test really necessary?
    if not isinstance(wa_res, (Iterable, int, float)):
        raise WindException(f"{wa_res} is neither array_like, int or float")

    if isinstance(wa_res, Iterable):
        # TODO: Check if contents of
        #       array are numbers?
        wa_res = np.asarray(wa_res)

        if wa_res.dtype == object:
            raise WindException(f"{wa_res} is not array_like")
        if not wa_res.size:
            raise WindException("Empty wa_res was passed")
        return wa_res

    if wa_res <= 0:
        raise WindException("Nonpositive resolution stepsize")

    return np.arange(wa_res, 360, wa_res)
