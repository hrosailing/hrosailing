"""
Functions to convert wind from apparent to true and vice versa
"""

# Author: Valentin F. Dannenberg / Ente


import logging.handlers
import numpy as np


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='logging/windconversion.log')
LOG_FILE = "logging/windconversion.log"

logger = logging.getLogger(__name__)
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when='midnight')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


def apparent_wind_to_true(aws, awa, bsp):
    logger.info("Function 'apparent_wind_to_true(aws, awa, bsp)' called")

    if any(x is None for x in (aws, awa, bsp)):
        return None, None
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

    return tws, twa


def true_wind_to_apparent(tws, twa, bsp):
    logger.info("Function 'true_wind_to_apparent(tws, twa, bsp)' called")

    if any(x is None for x in (tws, twa, bsp)):
        return None, None

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

    return aws, awa
