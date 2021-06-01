import numpy as np

from models2d import *


def tws_concave_downturn_twa(tws, twa, *args):
    return concave_function(tws, args[0], args[1], args[2]) \
           + args[3] * twa - np.square(twa - args[4]) * args[5] \
           + args[6] * twa * tws


def derive_tws_concave_downturn_twa():
    pass


def tws_and_twa_s_shaped_downturn(tws, twa, *args):
    return s_shaped(tws, args[0], args[1], args[2], args[3], downturn=True) \
           + s_shaped(twa, args[4], args[5], args[6], args[7], downturn=True) \
           + args[8] * tws * twa


def derive_tws_and_twa_s_shaped_downturn():
    pass


def tws_s_shaped_downturn_twa_gaussian(tws, twa, *args):
    return s_shaped(tws, args[0], args[1], args[2], args[3], downturn=True) \
           + gaussian_model(twa, args[4], args[5], args[6])


def derive_tws_s_shaped_downturn_twa_gaussian():
    pass


def tws_s_shaped_downturn_twa_gaussian_comb(tws, twa, *args):
    return s_shaped(tws, args[0], args[1], args[2], args[3], downturn=True) \
           + gaussian_model(twa, args[4], args[5], args[6]) \
           + args[7] * tws * twa


def derive_tws_s_shaped_downturn_twa_gaussian_comb():
    pass


def tws_s_shaped_twa_gaussian(tws, twa, *args):
    return s_shaped(tws, args[0], args[1], args[2], args[3]) \
           + gaussian_model(twa, args[4], args[5], args[6])


def derive_tws_s_shaped_twa_gaussian():
    pass
