"""
Functions to convert wind from apparent to true and vice versa
"""

# Author: Valentin F. Dannenberg / Ente


import numpy as np

from exceptions import PolarDiagramException


def apparent_wind_to_true(wind_arr):

    wind_arr = np.asarray(wind_arr)
    if not wind_arr.size:
        raise PolarDiagramException("")

    aws, awa, bsp = np.hsplit(wind_arr, 3)

    awa_above_180 = awa > 180
    awa = np.deg2rad(awa)

    tws = np.sqrt(np.square(aws) + np.square(bsp)
                  - 2 * aws * bsp * np.cos(awa))

    temp = (aws * np.cos(awa) - bsp) / tws
    temp[temp > 1] = 1
    temp[temp < -1] = -1

    twa = np.arccos(temp)
    twa[awa_above_180] = 360 - np.rad2deg(twa[awa_above_180])
    twa[np.logical_not(awa_above_180)] = np.rad2deg(
        twa[np.logical_not(awa_above_180)])

    return np.column_stack((tws, twa, bsp))


def true_wind_to_apparent(wind_arr):

    wind_arr = np.asarray(wind_arr)
    if not wind_arr.size:
        raise PolarDiagramException("")

    tws, twa, bsp = np.hsplit(wind_arr, 3)

    twa_above_180 = twa > 180
    twa_rad = np.deg2rad(twa)

    aws = np.sqrt(pow(tws, 2) + pow(bsp, 2)
                  + 2 * tws * bsp * np.cos(twa_rad))

    temp = (tws * np.cos(twa_rad) + bsp) / aws
    temp[temp > 1] = 1
    temp[temp < -1] = -1

    awa = np.arccos(temp)
    awa[twa_above_180] = 360 - np.rad2deg(awa[twa_above_180])
    awa[np.logical_not(twa_above_180)] = np.rad2deg(
        awa[np.logical_not(twa_above_180)])

    return np.column_stack((tws, twa, bsp))
