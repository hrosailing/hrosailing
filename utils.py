"""
Small utility functions used throughout the module
"""

# Author: Valentin F. Dannenberg / Ente


import logging.handlers
import numpy as np

from collections.abc import Iterable
from scipy.spatial import ConvexHull

from exceptions import PolarDiagramException
from windconversion import apparent_wind_to_true

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='logging/utils.log')
LOG_FILE = "logging/utils.log"

logger = logging.getLogger(__name__)
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when='midnight')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


def polar_to_kartesian(arr):
    logger.info("Function 'polar_to_kartesian(arr)' called")

    return np.column_stack(
        (arr[:, 1] * np.cos(arr[:, 0]),
         arr[:, 1] * np.sin(arr[:, 0])))


def euclidean_norm(vec):
    return np.linalg.norm(vec, axis=1)


def convex_hull_polar(points):
    logger.info("Function 'convex_hull_polar(points)' called")

    converted_points = polar_to_kartesian(points)
    return ConvexHull(converted_points)


def convert_wind(wind_arr, tw):
    logger.info(f"Function 'convert_wind(wind_arr, tw={tw})' called")

    if tw:
        return wind_arr

    logger.info("""Internal function 'windconversion.apparent_wind_to_true(
                 wind_arr)' called""")
    return apparent_wind_to_true(wind_arr)


# TODO: Make it cleaner!
def speed_resolution(ws_res):
    logger.info(f"Function 'speed_resolution(ws_res={ws_res})' called")

    if ws_res is None:
        return np.array(np.arange(2, 42, 2))

    if not isinstance(ws_res, (Iterable, int, float)):
        logger.error("Error occured when checking "
                     "ws_res for 'Iterability'")
        raise PolarDiagramException(
            "ws_res is neither Iterable, int or float")

    if isinstance(ws_res, Iterable):
        if isinstance(ws_res, np.ndarray):
            if not ws_res.size:
                logger.error("")
                raise PolarDiagramException("Empty ws_res was passed")
        else:
            if not ws_res:
                logger.error("")
                raise PolarDiagramException("Empty ws_res was passed")

        return np.asarray(ws_res)

    if not ws_res:
        logger.error("")
        raise PolarDiagramException("")

    return np.array(np.arange(ws_res, 40, ws_res))


# TODO: Make it cleaner!
def angle_resolution(wa_res):
    logger.info(f"Function 'angle_resolution(wa_res={wa_res})' called")

    if wa_res is None:
        return np.array(np.arange(0, 360, 5))

    if not isinstance(wa_res, (Iterable, int, float)):
        logger.error("Error occured when checking "
                     "wa_res for 'Iterability'")
        raise PolarDiagramException(
            "wa_res is neither Iterable, int or float")

    if isinstance(wa_res, Iterable):
        if isinstance(wa_res, np.ndarray):
            if not wa_res.size:
                logger.error("")
                raise PolarDiagramException("Empty ws_res was passed")
        else:
            if not wa_res:
                logger.error("")
                raise PolarDiagramException("Empty ws_res was passed")

        return np.asarray(wa_res)

    if not wa_res:
        logger.error("")
        raise PolarDiagramException("")

    return np.array(np.arange(wa_res, 360, wa_res))


# TODO: Make it cleaner!
def get_indices(w_list, res_list):
    logger.info(f"""Function 'get_indices(w_list={w_list},
                 res_list={res_list})' called""")

    if w_list is None:
        return list(range(len(res_list)))

    if not isinstance(w_list, Iterable):
        try:
            ind = list(res_list).index(w_list)
            return [ind]
        except ValueError:
            logger.error("""Error occured when checking 
                        if w_list is contained in 
                        res_list""")
            raise PolarDiagramException(
                f"{w_list} is not in resolution")

    if isinstance(w_list, np.ndarray):
        if not w_list.size:
            logger.error("")
            raise PolarDiagramException("")
    else:
        if not w_list:
            logger.error("")
            raise PolarDiagramException("")

    if not set(w_list).issubset(set(res_list)):
        logger.error("""Error occured when checking if w_list is
                    contained in res_list""")
        raise PolarDiagramException(
            f"{w_list} is not in resolution")

    ind_list = [i for i, w in enumerate(res_list) if w in w_list]
    return ind_list
