import logging
import logging.handlers
import numpy as np
import sys
from collections import Iterable
from scipy.spatial import ConvexHull
from exceptions import PolarDiagramException
from windconversion import apparent_wind_to_true

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO)
LOG_FILE = "utils.log"

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when='midnight')
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)


def polar_to_kartesian(arr):
    logger.debug("Function 'polar_to_kartesian(arr)' called")

    return np.column_stack(
        (arr[:, 1] * np.cos(arr[:, 0]),
         arr[:, 1] * np.sin(arr[:, 0]))
    )


def convex_hull_polar(points):
    logger.debug("Function 'convex_hull_polar(points)' called")

    converted_points = polar_to_kartesian(points)
    logger.debug("""Extern function 'scipy.spatial.ConvexHull(
                 converted_points)' called""")
    return ConvexHull(converted_points)


def convert_wind(w_dict, tw):
    logger.debug(f"Function 'convert_wind(w_dict, tw={tw})' called")

    if tw:
        return w_dict

    aws = w_dict.get("wind_speed")
    awa = w_dict.get("wind_angle")
    bsp = w_dict.get("boat_speed")

    logger.debug("""Internal function 'windconversion.apparent_wind_to_true(
                 aws, awa, bsp)' called""")
    tws, twa = apparent_wind_to_true(aws, awa, bsp)

    return {"wind_speed": tws, "wind_angle": twa}


def speed_resolution(ws_res):
    logger.debug(f"Function 'speed_resolution(ws_res={ws_res})' called")

    if ws_res is None:
        return np.array(np.arange(2, 42, 2))

    if not isinstance(ws_res, (Iterable, int, float)):
        logger.info("Error occured when checking ws_res for 'Iterability'")
        raise PolarDiagramException(
            "ws_res is neither Iterable, int or float")

    if isinstance(ws_res, Iterable):
        return np.asarray(ws_res)

    return np.array(np.arange(ws_res, 40, ws_res))


def angle_resolution(wa_res):
    logger.debug(f"Function 'angle_resolution(wa_res={wa_res})' called")

    if wa_res is None:
        return np.array(np.arange(0, 360, 5))

    if not isinstance(wa_res, (Iterable, int, float)):
        logger.info("Error occured when checking wa_res for 'Iterability'")
        raise PolarDiagramException(
            "wa_res is neither Iterable, int or float")

    if isinstance(wa_res, Iterable):
        return np.asarray(wa_res)

    return np.array(np.arange(wa_res, 360, wa_res))


def get_indices(w_list, res_list):
    logger.debug(f"""Function 'get_indices(w_list={w_list},
                 res_list={res_list})' called""")

    if w_list is None:
        return list(range(len(res_list)))

    if not isinstance(w_list, Iterable):
        try:
            ind = list(res_list).index(w_list)
            return [ind]
        except ValueError:
            logger.info("""Error occured when checking if w_list is
                        contained in res_list""")
            raise PolarDiagramException(
                f"{w_list} is not in resolution")

    if not set(w_list).issubset(set(res_list)):
        logger.info("""Error occured when checking if w_list is
                    contained in res_list""")
        raise PolarDiagramException(
            f"{w_list} is not in resolution")

    ind_list = [i for i in range(len(res_list)) if res_list[i] in w_list]
    return ind_list
