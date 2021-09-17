"""

"""

# Author: Valentin F. Dannenberg


import numpy as np


def polynomial_function(x, *params, deg=1):
    val = 0
    for i in range(deg + 1):
        val += params[i] * np.power(x, i)

    return val


def inverted_parabola(x, *params):
    return params[1] + params[2] * np.square(x - params[0])


def inverted_shifted_parabola(x, *params):
    return params[0] * x + params[2] * np.square(x - params[1])


def concave_function(x, *params, downturn=True, sat_limit=False):
    if sat_limit and downturn:
        return (
            params[2]
            - params[0] * np.exp(params[1] - x)
            - params[3] * np.square(x)
        )

    if sat_limit:
        return params[2] - params[0] * np.exp(params[1] - x)

    if downturn:
        return polynomial_function(x, params[0], params[1], -params[2], deg=2)

    raise ValueError(
        "At least one of `downturn` and `sat_limit` must be `True`"
    )


def s_shaped(x, *params, downturn=False):
    val = params[2] / (1 + np.exp(params[0] - params[1] * x))
    if downturn:
        val -= params[3] * np.square(x)

    return val


def gompertz_model(x, *params, neg=False):
    if neg:
        return params[2] * np.exp(-params[0] * np.power(-params[1], x))

    return params[2] * np.exp(-params[0] * np.power(params[1], x))


def gaussian_model(x, *params, offset=False):
    val = params[0] * np.exp(-0.5 * np.square(x - params[1]) / params[2])
    if offset:
        val += params[3]

    return val


def gmm_model(x, *params):
    return (
        params[0] * np.exp(-0.5 * np.square(x - params[1]) / params[2])
        + params[3] * np.exp(-0.5 * np.square(x - params[4]) / params[5])
        + params[6]
    )
