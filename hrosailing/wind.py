"""
Functions to convert wind from apparent to true and vice versa
"""

# Author: Valentin Dannenberg


import warnings
from collections.abc import Iterable
from typing import Iterable as Iter
from typing import Optional, Union

import numpy as np


class WindConversionException(Exception):
    """Exception raised if an error occurs during wind conversion"""


def apparent_wind_to_true(wind):
    """Converts apparent wind to true wind

    Parameters
    ----------
    wind : array_like of shape (n, 3)
        Wind data given as a sequence of points consisting of wind speed,
        wind angle and boat speed, where the wind speed and wind angle are
        measured as apparent wind

    Returns
    -------
    converted : numpy.ndarray of shape (n, 3)
        Array containing the same data as wind_arr, but the wind speed
        and wind angle now measured as true wind
    """
    return _convert_wind(wind, -1, tw=False, _check_finite=True)


def true_wind_to_apparent(wind):
    """Converts true wind to apparent wind

    Parameters
    ----------
    wind : array_like of shape (n, 3)
        Wind data given as a sequence of points consisting of wind speed,
        wind angle and boat speed, where the wind speed and wind angle are
        measured as true wind

    Returns
    -------
    converted : numpy.ndarray of shape (n, 3)
        Array containing the same data as wind_arr, but the wind speed
        and wind angle now measured as apparent wind
    """
    return _convert_wind(wind, 1, tw=False, _check_finite=True)


def _convert_wind(wind, sign, tw, _check_finite=True):
    # Only check for NaNs and infinite values, if wanted
    if _check_finite:
        # NaNs and infinite values can't be handled
        wind = np.asarray_chkfinite(wind)
    else:
        wind = np.asarray(wind)

    if wind.dtype == object:
        raise WindConversionException("`wind` is not array_like")

    if wind.ndim != 2 or wind.shape[1] != 3:
        raise WindConversionException("`wind` has incorrect shape")

    if tw:
        return wind

    ws, wa, bsp = np.hsplit(wind, 3)
    wa_above_180 = wa > 180
    wa = np.deg2rad(wa)

    cws = np.sqrt(
        np.square(ws) + np.square(bsp) + sign * 2 * ws * bsp * np.cos(wa)
    )

    temp = (ws * np.cos(wa) + sign * bsp) / cws

    # account for computer error
    temp[temp > 1] = 1
    temp[temp < -1] = -1

    cwa = np.arccos(temp)

    # standardize angles to [0, 360) inverval after conversion
    cwa[wa_above_180] = 360 - np.rad2deg(cwa[wa_above_180])
    cwa[np.invert(wa_above_180)] = np.rad2deg(cwa[np.invert(wa_above_180)])

    return np.column_stack((cws, cwa, bsp))


def _set_resolution(res: Optional[Union[Iter, int, float]], soa):
    soa = soa == "s"

    if res is None:
        return np.arange(2, 42, 2) if soa else np.arange(0, 360, 5)

    if isinstance(res, Iterable):
        # NaN's and infinite values can't be handled
        res = np.asarray_chkfinite(res)

        if res.dtype == object:
            raise ValueError("`res` is not array_like")

        if not res.size or res.ndim != 1:
            raise ValueError("`res` has incorrect shape")

        if len(set(res)) != len(res):
            warnings.warn(
                "`res` contains duplicate data. "
                "This may lead to unwanted behaviour"
            )

        return res

    if res <= 0:
        raise ValueError("`res` is nonpositive")

    return np.arange(res, 40, res) if soa else np.arange(res, 360, res)
