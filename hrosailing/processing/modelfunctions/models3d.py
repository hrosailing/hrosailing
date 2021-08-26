"""

"""

# Author: Valentin F. Dannenberg / Ente


from hrosailing.processing.modelfunctions.models2d import *


def tws_times_twa(tws, twa, scal):
    tws = np.asarray(tws)
    twa = np.asarray(twa)
    return scal * tws * twa


def tws_concave_dt_twa(tws, twa, *args):
    tws = np.asarray(tws)
    twa = np.asarray(twa)
    return (
        concave_function(tws, args[0], args[1], args[2])
        + inverted_shifted_parabola(twa, args[3], args[4], args[5])
        + tws_times_twa(tws, twa, args[6])
    )


def tws_twa_s_dt(tws, twa, *args):
    tws = np.asarray(tws)
    twa = np.asarray(twa)
    return (
        s_shaped(tws, args[0], args[1], args[2], args[3], downturn=True)
        + s_shaped(twa, args[4], args[5], args[6], args[7], downturn=True)
        + tws_times_twa(tws, twa, args[8])
    )


def tws_s_dt_twa_gauss(tws, twa, *args):
    tws = np.asarray(tws)
    twa = np.asarray(twa)
    return s_shaped(
        tws, args[0], args[1], args[2], args[3], downturn=True
    ) + gaussian_model(twa, args[4], args[5], args[6])


def tws_s_s_dt_twa_gauss_comb(tws, twa, *args):
    tws = np.asarray(tws)
    twa = np.asarray(twa)
    return (
        s_shaped(tws, args[0], args[1], args[2], args[3], downturn=True)
        + gaussian_model(twa, args[4], args[5], args[6])
        + tws_times_twa(tws, twa, args[7])
    )


def tws_s_twa_gauss(tws, twa, *args):
    tws = np.asarray(tws)
    twa = np.asarray(twa)
    return s_shaped(tws, args[0], args[1], args[2], args[3]) + gaussian_model(
        twa, args[4], args[5], args[6]
    )
