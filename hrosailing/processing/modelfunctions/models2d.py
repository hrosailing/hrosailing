import numpy as np


def polynomial_function(x, *args, deg=1):
    val = 0
    for i in range(deg + 1):
        val += args[i] * np.power(x, i)

    return val


def inverted_parabola(x, *args):
    return args[1] + args[2] * np.square(x - args[0])


def inverted_shifted_parabola(x, *args):
    return args[0] * x + args[2] * np.square(x - args[1])


def concave_function(x, *args, downturn=True, sat_limit=False):
    if sat_limit and downturn:
        return args[2] - args[0] * np.exp(args[1] - x) - args[3] * np.square(x)

    if sat_limit:
        return args[2] - args[0] * np.exp(args[1] - x)

    if downturn:
        return polynomial_function(x, args[0], args[1], -args[2], deg=2)


def s_shaped(x, *args, downturn=False):
    val = args[2] / (1 + np.exp(args[0] - args[1] * x))
    if downturn:
        val -= args[3] * np.square(x)

    return val


def gompertz_model(x, *args, neg=False):
    if neg:
        return args[2] * np.exp(-args[0] * np.power(-args[1], x))

    return args[2] * np.exp(-args[0] * np.power(args[1], x))


def gaussian_model(x, *args, offset=False):
    if offset:
        return (
            args[0] * np.exp(-0.5 * np.square(x - args[1]) / args[2]) + args[3]
        )

    return args[0] * np.exp(-0.5 * np.square(x - args[1]) / args[2])


def gmm_model(x, *args):
    return (
        args[0] * np.exp(-0.5 * np.square(x - args[1]) / args[2])
        + args[3] * np.exp(-0.5 * np.square(x - args[4]) / args[5])
        + args[6]
    )
