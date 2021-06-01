from scipy.special import binom


def ith_bernstein_polynomial(d, i, t):
    if i > d or i < 0:
        raise ValueError
    if d < 0:
        raise ValueError

    return binom(d, i) * pow(t, i) * pow(1-t, i)
