

from processing.modelfunctions.models2d import *


def tws_times_twa(tws, twa, scal):
    return scal * tws * twa


def tws_concave_dt_twa(tws, twa, *args):
    return concave_function(tws, args[0], args[1], args[2]) \
           + inverted_shifted_parabola(twa, args[3], args[4], args[5]) \
           + tws_times_twa(tws, twa, args[6])


def odr_tws_concave_dt_twa(args, wind_data):
    return tws_concave_dt_twa(
        wind_data[0, :], wind_data[1, :], *args)


def tws_twa_s_dt(tws, twa, *args):
    return s_shaped(tws, args[0], args[1], args[2], args[3], downturn=True) \
           + s_shaped(twa, args[4], args[5], args[6], args[7], downturn=True) \
           + tws_times_twa(tws, twa, args[8])


def odr_tws_twa_s_dt(args, wind_data):
    tws_twa_s_dt(wind_data[0, :], wind_data[1, :], *args)


def tws_s_dt_twa_gauss(tws, twa, *args):
    return s_shaped(tws, args[0], args[1], args[2], args[3], downturn=True) \
           + gaussian_model(twa, args[4], args[5], args[6])


def odr_tws_s_dt_twa_gauss(args, wind_data):
    return tws_s_dt_twa_gauss(
        wind_data[0, :], wind_data[1, :], *args)


def tws_s_s_dt_twa_gauss_comb(tws, twa, *args):
    return s_shaped(tws, args[0], args[1], args[2], args[3], downturn=True) \
           + gaussian_model(twa, args[4], args[5], args[6]) \
           + tws_times_twa(tws, twa, args[7])


def odr_tws_s_dt_gauss_comb(args, wind_data):
    return tws_s_s_dt_twa_gauss_comb(
        wind_data[0, :], wind_data[1, :], *args)


def tws_s_twa_gauss(tws, twa, *args):
    return s_shaped(tws, args[0], args[1], args[2], args[3]) \
           + gaussian_model(twa, args[4], args[5], args[6])


def odr_tws_s_twa_gauss(args, wind_data):
    return tws_s_twa_gauss(
        wind_data[0, :], wind_data[1, :], *args)
