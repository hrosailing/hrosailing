from numpy.ma import arccos, cos, sqrt
from numpy import deg2rad, rad2deg


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

    awa_rad = deg2rad(awa)
    tws = sqrt(pow(aws, 2) + pow(bsp, 2) + 2 * aws * bsp * cos(awa_rad))
    twa = arccos((aws * cos(awa_rad) - bsp) / tws)
    twa = rad2deg(twa)

    return tws, twa


def true_wind_to_appearent(tws, twa, bsp):

    if any(x is None for x in (tws, twa, bsp)):
        return None, None

    twa_rad = deg2rad(twa)
    aws = sqrt(pow(tws, 2) + pow(bsp, 2) + 2 * tws * bsp * cos(twa_rad))
    awa = arccos((tws * cos(twa_rad) + bsp) / aws)
    awa = rad2deg(awa)

    return aws, awa
