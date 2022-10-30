"""Functions to convert wind from apparent to true and vice versa."""


import enum

import numpy as np


class WindConversionException(Exception):
    """Exception raised if an error occurs during wind conversion"""


class _Wind(enum.Enum):
    TO_TRUE = -1
    TO_APPARENT = 1

    def convert_wind(self, wind):
        wind = np.asarray_chkfinite(wind)

        if wind.dtype == object:
            raise WindConversionException("`wind` is not array_like")
        if wind.ndim != 2 or wind.shape[1] != 3:
            raise WindConversionException("`wind` has incorrect shape")

        ws, wa, bsp = np.hsplit(wind, 3)
        if np.any((ws < 0)):
            raise WindConversionException("`wind` has negative wind speeds")

        wa %= 360  # normalize wind angles
        wa_above_180 = wa > 180
        wa = np.deg2rad(wa)

        converted_ws = self._convert_wind_speed(ws, wa, bsp)
        converted_wa = self._convert_wind_angle(
            converted_ws, ws, wa, bsp, wa_above_180
        )

        return np.column_stack((converted_ws, converted_wa, bsp))

    def _convert_wind_speed(self, ws, wa, bsp):
        return np.sqrt(
            np.square(ws)
            + np.square(bsp)
            + 2 * self.value * ws * bsp * np.cos(wa)
        )

    def _convert_wind_angle(self, converted_ws, ws, wa, bsp, wa_above_180):
        temp = (ws * np.cos(wa) + self.value * bsp) / converted_ws

        # account for floating point errors
        temp[temp > 1] = 1
        temp[temp < -1] = -1

        converted_wa = np.arccos(temp)
        converted_wa[wa_above_180] = 360 - np.rad2deg(
            converted_wa[wa_above_180]
        )
        converted_wa[~wa_above_180] = np.rad2deg(converted_wa[~wa_above_180])

        return converted_wa


def convert_apparent_wind_to_true(apparent_wind):
    """Convert apparent wind to true wind.

    Parameters
    ----------
    apparent_wind : array_like of shape (n, 3)
        Wind data given as a sequence of points consisting of wind speed,
        wind angle and boat speed, where the wind speed and wind angle are
        measured as apparent wind.

    Returns
    -------
    true_wind : numpy.ndarray of shape (n, 3)
        Array containing the same data as `apparent_wind`, but the wind speed
        and wind angle now measured as true wind.

    Raises
    ------
    WindConversionException
    """
    return _Wind.TO_TRUE.convert_wind(apparent_wind)


def convert_true_wind_to_apparent(true_wind):
    """Convert true wind to apparent wind.

    Parameters
    ----------
    true_wind : array_like of shape (n, 3)
        Wind data given as a sequence of points consisting of wind speed,
        wind angle and boat speed, where the wind speed and wind angle are
        measured as true wind.

    Returns
    -------
    apparent_wind : numpy.ndarray of shape (n, 3)
        Array containing the same data as `true_wind`, but the wind speed
        and wind angle now measured as apparent wind.

    Raises
    ------
    WindConversionException
    """
    return _Wind.TO_APPARENT.convert_wind(true_wind)
