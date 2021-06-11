import numpy as np

from data_analysis.models2d import *


def tws_times_twa(tws, twa, scal):
    return scal * tws * twa


def tws_concave_downturn_twa(tws, twa, *args):
    return concave_function(tws, args[0], args[1], args[2]) \
           + inverted_shifted_parabola(twa, args[3], args[4], args[5]) \
           + tws_times_twa(tws, twa, args[6])


def derive_tws_concave_downturn_twa():
    pass


def tws_and_twa_s_shaped_downturn(tws, twa, *args):
    return s_shaped(tws, args[0], args[1], args[2], args[3], downturn=True) \
           + s_shaped(twa, args[4], args[5], args[6], args[7], downturn=True) \
           + tws_times_twa(tws, twa, args[8])


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
           + tws_times_twa(tws, twa, args[7])


def derive_tws_s_shaped_downturn_twa_gaussian_comb():
    pass


def tws_s_shaped_twa_gaussian(tws, twa, *args):
    return s_shaped(tws, args[0], args[1], args[2], args[3]) \
           + gaussian_model(twa, args[4], args[5], args[6])


def derive_tws_s_shaped_twa_gaussian():
    pass
