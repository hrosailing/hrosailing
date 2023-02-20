# pylint: disable=missing-docstring

import numpy as np


def polynomial_function(x, *params, deg=1):
    """If `deg=n`, models the function
    \\[
        a_0 + a_1x + \\dots + a_nx^n
    \\]
    """
    val = 0
    for i in range(deg + 1):
        val += params[i] * np.power(x, i)

    return val


def inverted_parabola(x, *params):
    """Models the function
    \\[
        a_1 + a_2(x - a_0)^2
    \\]
    """
    return params[1] + params[2] * np.square(x - params[0])


def inverted_shifted_parabola(x, *params):
    """Models the function
    .. math:: a_0x + a_2(x - a_1)^2

    Parameters
    -----------

    *params :
        The parameters of the modeled function, i.e.
        .. math:: a_0, a_1, a_2
    """
    return params[0] * x + params[2] * np.square(x - params[1])


def concave_function(x, *params, downturn=True, sat_limit=False):
    """
    Models a concave function depending on the values of `downturn` and `sat_limits`.
    Note that at least one of the parameter `downturn` and `sat_limit` has to be set to `True`.

    - `downturn` and `sat_limit` are both `True`, models the function
        .. math:: a_2 - a_0e^{a_1 - x} - a_3x^2
    - If `downturn` is `False` and `sat_limit` is `True`, models the function
        .. math:: a_2 - a_0e^{a_1 - x}
    - If `downturn` is `True` and `sat_limits` is `False`, models the function
        .. math:: a_0 + a_1x - a_2x^2

    Parameters
    ----------
    *params :
        The parameters of the modeled function, i.e.
        .. math:: a_0, a_1, a_2, a_3
        Supported length of `params` depend on the values of `downturn` and `sat_limits`.

    downturn : bool, optional
        Defaults to `True`.

    sat_limit : bool, optional
        Defaults to `False`.
    """
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


def s_shaped(x, *params, downturn=True):
    """
    Models the function
    .. math:: \frac{a_2}{1 + e^{a_0 - a_1x}} - a_3x^2

    Parameters
    -----------

    *params :
        The parameters of the modeled function, i.e.
        .. math:: a_0, a_1, a_2, a_3
        The supported length depends on the value of `downturn`.

    downturn : bool, optional
        If `False`, we set
        .. math:: a_3 = 0
        independently from the values in `params`.
    """
    val = params[2] / (1 + np.exp(params[0] - params[1] * x))
    if downturn:
        val -= params[3] * np.square(x)

    return val


def gompertz_model(x, *params, neg=False):
    if neg:
        return params[2] * np.exp(-params[0] * np.power(-params[1], x))

    return params[2] * np.exp(-params[0] * np.power(params[1], x))


def gaussian_model(x, *params, offset=False):
    """
    Models the function
    .. math:: a_0e^{-\frac{(x - a_1)^2}{2a_2}} + a_3

    Parameters
    ----------
    *params :
        The parameters of the modeled function, i.e.
        .. math:: a_0, a_1, a_2, a_3
        Supported length of `params` depend on the value of `offset`.

    offset : bool, optional
        If `False`, we set
        .. math:: a_3 = 0
        independently from the values in `params`.
    """
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
