from numpy.ma import arccos, cos, sqrt
from numpy import deg2rad, rad2deg


# V: In Arbeit
def apparent_wind_to_true(aws, awa, bsp):
    if aws is None or awa is None or bsp is None:
        return None, None

    awa_rad = deg2rad(awa)
    tws = sqrt(pow(aws, 2) + pow(bsp, 2) + 2 * aws * bsp * cos(awa_rad))
    twa = arccos((aws * cos(awa_rad) - bsp) / tws)
    twa = rad2deg(twa)

    return tws, twa

