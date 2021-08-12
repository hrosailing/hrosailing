"""
Functions to convert wind from apparent to true and vice versa
"""

# Author: Valentin F. Dannenberg / Ente


from typing import Iterable

import numpy as np


class WindException(Exception):
    """"""
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
    twa = np.deg2rad(twa)

    aws = np.sqrt(
        np.square(tws, 2) + np.square(bsp, 2) + 2 * tws * bsp * np.cos(twa)
    )

    temp = (tws * np.cos(twa) + bsp) / aws
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


def set_resolution(res, speed_or_angle):
    if speed_or_angle not in {"speed", "angle"}:
        raise WindException("")

    b = speed_or_angle == "speed"

    if res is None:
        return np.arange(2, 42, 2) if b else np.arange(0, 360, 5)

    # Iterable-test really necessary?
    if not isinstance(res, (Iterable, int, float)):
        raise WindException(f"{res} is neither array_like, int or float")

    if isinstance(res, Iterable):
        # TODO: Check if contents of
        #       array are numbers?
        res = np.asarray(res)

        if res.dtype == object:
            raise WindException(f"{res} is not array_like")
        if not res.size:
            raise WindException("Empty ws_res was passed")

        return res

    if res <= 0:
        raise WindException("Nonpositive resolution stepsize")

    return np.arange(res, 40, res) if b else np.arange(res, 360, res)
