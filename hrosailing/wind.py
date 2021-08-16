"""
Functions to convert wind from apparent to true and vice versa
"""

# Author: Valentin F. Dannenberg / Ente


from typing import Iterable

import numpy as np


class WindException(Exception):
    """"""

    pass


def apparent_wind_to_true(wind):
    """Converts apparent wind to true wind

    Parameters
    ----------
    wind : array_like
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
    return convert_wind(wind, -1, False)


def true_wind_to_apparent(wind):
    """Converts true wind to apparent wind

        Parameters
        ----------
        wind : array_like
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
    return convert_wind(wind, 1, False)


def convert_wind(wind, sign, tw):
    wind = _sanity_checks(wind)
    if tw:
        return wind

    ws, wa, bsp = np.hsplit(wind, 3)
    wa_above_180 = wa > 180
    wa = np.deg2rad(wa)

    cws = np.sqrt(
        np.square(ws) + np.square(bsp) + sign * 2 * ws * bsp * np.cos(wa)
    )

    temp = (ws * np.cos(wa) + sign * bsp) / cws
    # Account for computer error
    temp[temp > 1] = 1
    temp[temp < -1] = -1

    cwa = np.arccos(temp)
    cwa[wa_above_180] = 360 - np.rad2deg(cwa[wa_above_180])
    cwa[np.logical_not(wa_above_180)] = np.rad2deg(
        cwa[np.logical_not(wa_above_180)]
    )

    return np.column_stack((cws, cwa, bsp))


def _sanity_checks(wind):
    try:
        wind = np.asarray_chkfinite(wind)
    except ValueError as ve:
        raise WindException(
            "array should only contain finite and non-NaN values"
        ) from ve

    if wind.dtype is object:
        raise WindException("array is not array_like")
    if not wind.size:
        raise WindException("Empty array was passed")
    try:
        wind = wind.reshape(-1, 3)
    except ValueError:
        raise WindException(
            "array could not be broadcasted to an array of shape (n, 3)"
        )

    return wind


def set_resolution(res, speed_or_angle):
    if speed_or_angle not in {"speed", "angle"}:
        raise WindException("")

    b = speed_or_angle == "speed"

    if res is None:
        return np.arange(2, 42, 2) if b else np.arange(0, 360, 5)

    # TODO Iterable-test really necessary?
    if not isinstance(res, (Iterable, int, float)):
        raise WindException(f"{res} is neither array_like, int or float")

    if isinstance(res, Iterable):
        # TODO: Check if contents of array are numbers?
        try:
            res = np.asarray_chkfinite(res)
        except ValueError as ve:
            raise WindException(
                f"{res} should only have finite and non-NaN entries"
            ) from ve

        if res.dtype is object:
            raise WindException(f"{res} is not array_like")
        if not res.size:
            raise WindException("Empty res was passed")

        if len(set(res)) != len(res):
            print(
                "Warning: Wind resolution contains duplicate data."
                "This may lead to unwanted behaviour"
            )

        return res

    if res <= 0:
        raise WindException("Nonpositive resolution stepsize")

    return np.arange(res, 40, res) if b else np.arange(res, 360, res)
