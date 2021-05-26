import numpy as np


# V: In Arbeit
def convert_wind(w_dict, tw):
    if tw:
        return w_dict

    aws = w_dict.get("wind_speed")
    awa = w_dict.get("wind_angle")
    bsp = w_dict.get("boat_speed")

    tws, twa = apparent_wind_to_true(aws, awa, bsp)

    return {"wind_speed": tws, "wind_angle": twa}


# V: In Arbeit
def apparent_wind_to_true(aws, awa, bsp):
    if any(x is None for x in (aws, awa, bsp)):
        return None, None
    awa_above_180 = awa > 180
    awa = np.deg2rad(awa)

    tws = np.sqrt(np.square(aws) + np.square(bsp)
                  - 2 * aws * bsp * np.cos(awa))
    twa = np.arccos((aws * np.cos(awa) - bsp) / tws)
    twa[awa_above_180] = 360 - np.rad2deg(twa[awa_above_180])
    twa[np.logical_not(awa_above_180)] = np.rad2deg(
        twa[np.logical_not(awa_above_180)])

    return tws, twa


def true_wind_to_appearent(tws, twa, bsp):

    if any(x is None for x in (tws, twa, bsp)):
        return None, None

    twa_above_180 = twa > 180
    twa_rad = np.deg2rad(twa)

    aws = np.sqrt(pow(tws, 2) + pow(bsp, 2)
                  + 2 * tws * bsp * np.cos(twa_rad))
    awa = np.arccos((tws * np.cos(twa_rad) + bsp) / aws)
    awa[twa_above_180] = 360 - np.rad2deg(awa[twa_above_180])
    awa[np.logical_not(twa_above_180)] = np.rad2deg(
        awa[np.logical_not(twa_above_180)])

    return aws, awa
