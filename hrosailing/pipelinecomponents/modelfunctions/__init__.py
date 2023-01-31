"""Collection of model functions that can be used with the `Regressor` class
to model various different ship behaviours.

"""


import numpy as np

from ._models2d import (
    concave_function,
    gaussian_model,
    inverted_shifted_parabola,
    s_shaped,
)


def ws_times_wa(ws, wa, scal):
    """
    Returns a scaled version of the product of wind speeds and wind angles.

    Parameters
    --------
    ws: array_like
        Wind speeds.

    wa: array_like
        Wind angles.

    scal: int or float
        The scaling factor.
    """
    ws = np.asarray(ws)
    wa = np.asarray(wa)
    return scal * ws * wa


def ws_concave_dt_wa(ws, wa, *params):
    """A concave function with downturn for wind speed and angle.

    Notes
    -----
    Models the function:

    .. math:: a_0 + a_1x - a_2x^2 + a_3y + a_5(y - a_4)^2 + a_6xy \
    + a_7(360 - y) + a_9((360 - y) - a_8)^2 + a_{10}x(360 - y)

    """
    wa = np.asarray(wa)

    return (
        concave_function(ws, params[0], params[1], params[2])
        + inverted_shifted_parabola(wa, params[3], params[4], params[5])
        + ws_times_wa(ws, wa, params[6])
        + inverted_shifted_parabola(360 - wa, params[7], params[8], params[9])
        + ws_times_wa(ws, 360 - wa, params[10])
    )


def ws_wa_s_dt(ws, wa, *params):
    """An s-shaped function with downturn for wind speed and angle.

    Notes
    -----
    Models the function:

    .. math:: \\frac{a_2}{1 + \\exp(a_0 - a_1x)} - a_3x^2 \
    + \\frac{a_6}{1 + \\exp(a_4 - a_5y)} - a_7y^2 + a_8xy \
    + \\frac{a_{11}}{1 + \\exp(a_9 - a_{10}(360 - y))} - a_{12}(360 - y)^2 \
    + a_{13}x(360 - y)

    """
    wa = np.asarray(wa)

    return (
        s_shaped(ws, params[0], params[1], params[2], params[3])
        + s_shaped(wa, params[4], params[5], params[6], params[7])
        + ws_times_wa(ws, wa, params[8])
        + s_shaped(360 - wa, params[9], params[10], params[11], params[12])
        + ws_times_wa(ws, 360 - wa, params[13])
    )


def ws_s_dt_wa_gauss(ws, wa, *params):
    """An s-shaped function with downturn for wind speed and
    a gaussian function for wind angle.

    Notes
    -----
    Models the function:

    .. math:: \\frac{a_2}{1 + \\exp(a_0 - a_1x)} - a_3x^2 \
    + a_4\\exp\\left(\\frac{-(y - a_5)^2}{2a_6}\\right) \
    + a_7\\exp\\left(\\frac{-((360 - y) - a_8)^2}{2a_9}\\right)

    """
    wa = np.asarray(wa)

    return (
        s_shaped(ws, params[0], params[1], params[2], params[3])
        + gaussian_model(wa, params[4], params[5], params[6])
        + gaussian_model(360 - wa, params[7], params[8], params[9])
    )


def ws_s_s_dt_wa_gauss_comb(ws, wa, *params):
    """

    Notes
    -----
    Models the function:

    .. math:: \\frac{a_2}{1 + \\exp(a_0 - a_1x)} - a_3x^2 \
    + a_4\\exp\\left(\\frac{-(y - a_5)^2}{2a_6}\\right) + a_7xy \
    + a_8\\exp\\left(\\frac{-((360 - y) - a_9)^2}{2a_{10}}\\right) \
    + a_11x(360 - y)

    """
    wa = np.asarray(wa)

    return (
        s_shaped(ws, params[0], params[1], params[2], params[3])
        + gaussian_model(wa, params[4], params[5], params[6])
        + ws_times_wa(ws, wa, params[7])
        + gaussian_model(360 - wa, params[8], params[9], params[10])
        + ws_times_wa(ws, 360 - wa, params[11])
    )


def ws_s_wa_gauss(ws, wa, *params):
    """An s-shaped function for wind speed and a gaussian function for wind
    angle.

    Notes
    -----
    Models the function:

    .. math:: \\frac{a_2}{1 + \\exp(a_0 - a_1x)} \
    + a_3\\exp\\left(\\frac{-(y - a_4)^2}{2a_5}\\right) \
    + a_6\\exp\\left(\\frac{-((360 - y) - a_7)^2}{2a_8}\\right)

    """
    wa = np.asarray(wa)

    return (
        s_shaped(ws, *params[0:4])
        + gaussian_model(wa, *params[4:8])
        + gaussian_model(360 - wa, *params[8:12])
    )


def ws_s_wa_gauss_and_square(ws, wa, *params):
    """

    Notes
    -----
    Models the function

    .. math:: \\left(\\frac{a_2}{1 + \\exp(a_0 - a_1x)} - a_3x^2 \
    + x\\left(a_4\\exp\\left(\\frac{-(y - a_5)^2}{2a_6}\\right) \
    + a_7\\exp\\left(\\frac{-((360 - y) - a_8)^2}{2a_9}\\right) \
    \\right)\\right)y(360 - y)

    """
    ws = np.asarray(ws)
    wa = np.asarray(wa)
    return (
        (
            s_shaped(ws, params[0], params[1], params[2], params[3])
            + ws
            * (
                gaussian_model(wa, params[4], params[5], params[6])
                + gaussian_model(360 - wa, params[7], params[8], params[9])
            )
        )
        * wa
        * (360 - wa)
    )
