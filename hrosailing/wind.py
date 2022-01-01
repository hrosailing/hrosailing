"""
Functions to convert wind from apparent to true and vice versa
"""

import warnings
from collections.abc import Iterable
from typing import Iterable as Iter
from typing import Optional, Union

import numpy as np


class WindConversionException(Exception):
    """Exception raised if an error occurs during wind conversion"""


def apparent_wind_to_true(apparent_wind):
    """Converts apparent wind to true wind

    Parameters
    ----------
    apparent_wind : array_like of shape (n, 3)
        Wind data given as a sequence of points consisting of wind speed,
        wind angle and boat speed, where the wind speed and wind angle are
        measured as apparent wind

    Returns
    -------
    true_wind : numpy.ndarray of shape (n, 3)
        Array containing the same data as wind_arr, but the wind speed
        and wind angle now measured as true wind

    Raises
    ------
    WindConversionException
    """
    return _convert_wind(apparent_wind, sign=-1)


def true_wind_to_apparent(true_wind):
    """Converts true wind to apparent wind

    Parameters
    ----------
    true_wind : array_like of shape (n, 3)
        Wind data given as a sequence of points consisting of wind speed,
        wind angle and boat speed, where the wind speed and wind angle are
        measured as true wind

    Returns
    -------
    true_wind : numpy.ndarray of shape (n, 3)
        Array containing the same data as wind_arr, but the wind speed
        and wind angle now measured as apparent wind

    Raises
    ------
    WindConversionException
    """
    return _convert_wind(true_wind, sign=1)


def _convert_wind(wind, sign):
    # NaNs and infinite values will cause problems later on
    wind = np.asarray_chkfinite(wind)
    
    if wind.dtype == object:
        raise WindConversionException("`wind` is not array_like")
    if wind.ndim != 2 or wind.shape[1] != 3:
        raise WindConversionException("`wind` has incorrect shape")

    ws, wa, bsp = np.hsplit(wind, 3)
    wa_above_180 = wa > 180
    wa = np.deg2rad(wa)

    converted_ws = _convert_wind_speed(ws, wa, bsp)

    converted_wa = _convert_wind_angle(converted_ws, ws, wa, bsp, sign)
    _standardize_converted_angles(converted_wa, wa_above_180)

    return np.column_stack((converted_ws, converted_wa, bsp))


def _convert_wind_speed(ws, wa, bsp, sign):
    return np.sqrt(np.square(ws) + np.square(bsp) + 2 * sign * ws * bsp * np.cos(wa))


def _convert_wind_angle(converted_ws, ws, wa, bsp, sign):
    temp = (ws * np.cos(wa) + sign * bsp) / converted_ws

    # account for floating point limitations and errors
    temp[temp > 1] = 1
    temp[temp < -1] = -1

    return np.arccos(temp)


def _standardize_converted_angles(converted_wa, wa_above_180):
    converted_wa[wa_above_180] = 360 - np.rad2deg(converted_wa[wa_above_180])
    converted_wa[~wa_above_180] = np.rad2deg(converted[~wa_above_180])


def _set_resolution(res: Optional[Union[Iter, int, float]], soa):
    # check if wind or angle resolution should be set
    soa = soa == "s"

    if res is None:
        return _standard_resolution(soa)

    if isinstance(res, Iterable):
        return _custom_iterable_resolution(res)
        
    return _custom_stepsize_resolution(res)


def _standard_resolution(soa):
    return np.arange(2, 42, 2) if soa else np.arange(0, 360, 5)


def _custom_iterable_resolution(res):
    # NaN's and infinite values cause problems later on
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


def _custom_stepsize_resolution(res):
    if res <= 0:
        raise ValueError("`res` is nonpositive")

    return np.arange(res, 40, res) if soa else np.arange(res, 360, res)

