"""
Model functions that can be used with the Regressor class to
model certain ship behaviours
"""

# Author: Valentin F. Dannenberg / Ente


from ._models2d import *


def ws_times_wa(ws, wa, scal):
    ws = np.asarray(ws)
    wa = np.asarray(wa)
    return scal * ws * wa


def ws_concave_dt_wa(ws, wa, *params):
    wa = np.asarray(wa)

    return (
        concave_function(ws, params[0], params[1], params[2])
        + inverted_shifted_parabola(wa, params[3], params[4], params[5])
        + ws_times_wa(ws, wa, params[6])
        + inverted_shifted_parabola(360 - wa, params[7], params[8], params[9])
        + ws_times_wa(ws, 360 - wa, params[10])
    )


def ws_wa_s_dt(ws, wa, *params):
    wa = np.asarray(wa)

    return (
        s_shaped(ws, params[0], params[1], params[2], params[3], downturn=True)
        + s_shaped(
            wa, params[4], params[5], params[6], params[7], downturn=True
        )
        + ws_times_wa(ws, wa, params[8])
        + s_shaped(
            360 - wa,
            params[9],
            params[10],
            params[11],
            params[12],
            downturn=True,
        )
        + ws_times_wa(ws, 360 - wa, params[13])
    )


def ws_s_dt_wa_gauss(ws, wa, *params):
    wa = np.asarray(wa)

    return (
        s_shaped(ws, params[0], params[1], params[2], params[3], downturn=True)
        + gaussian_model(wa, params[4], params[5], params[6])
        + gaussian_model(360 - wa, params[7], params[8], params[9])
    )


def ws_s_s_dt_wa_gauss_comb(ws, wa, *params):
    wa = np.asarray(wa)

    return (
        s_shaped(ws, params[0], params[1], params[2], params[3], downturn=True)
        + gaussian_model(wa, params[4], params[5], params[6])
        + ws_times_wa(ws, wa, params[7])
        + gaussian_model(360 - wa, params[8], params[9], params[10])
        + ws_times_wa(ws, 360 - wa, params[11])
    )


def ws_s_wa_gauss(ws, wa, *params):
    wa = np.asarray(wa)

    return (
        s_shaped(ws, params[0], params[1], params[2], params[3])
        + gaussian_model(wa, params[4], params[5], params[6])
        + gaussian_model(360 - wa, params[7], params[8], params[9])
    )


def ws_s_wa_gauss_and_square(tws, twa, *args):
    tws = np.asarray(tws)
    twa = np.asarray(twa)
    return (
            (s_shaped(tws, args[0], args[1], args[2], args[3], downturn=True)
             + tws * (gaussian_model(twa, args[4], args[5], args[6])
                      + gaussian_model(twa, args[7], args[8], args[9]))
             )
            * twa * (360 - twa)
    )
