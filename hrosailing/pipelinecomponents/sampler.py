"""
Classes used for modular modelling of different sampling methods.

Defines the `Sampler` abstract base class that can be used to create
custom sampling methods.

Subclasses of `Sampler` can be used with the `PointcloudExtension` class
in the `hrosailing.pipeline` module.
"""


from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import ConvexHull


class SamplerInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of a `Sampler` object.
    """


class Sampler(ABC):
    """Base class for all sampler classes.


    Abstract Methods
    ----------------
    sample(self, pts)
    """

    @abstractmethod
    def sample(self, pts):
        """This method should be used, given certain points, to determine a
        constant number of sample points that are more or less representative of the trend of the given points.

        Parameters
        ----------
        pts : numpy.ndarray
            The given data points in a row-wise fashion. Most commonly this should be an (n, 2) array.

        Returns
        ----------
        sample : numpy.ndarray
            A sample representing the data points.
        """


class UniformRandomSampler(Sampler):
    """A sampler that produces a number of uniformly distributed samples,
    which all lie in the convex hull of certain given points.

    Parameters
    ----------
    n_samples : positive int
        Amount of samples that will be produced by the sampler.

    Raises
    ------
    SamplerInitializationException
        If `n_samples` is non-positive.
    """

    def __init__(self, n_samples):
        if n_samples <= 0:
            raise SamplerInitializationException("`n_samples` is non-positive")

        self._n_samples = n_samples

    def sample(self, pts):
        """Produces samples according to the procedure described above.

        Parameters
        ----------
        pts : array_like of shape (n, 2)
            Points in whose convex hull the produced samples will lie.

        Returns
        -------
        samples : numpy.ndarray of shape (n_samples, 2)
            Samples produced by the procedure described above.
        """
        rng = np.random.default_rng()
        proj_pts = pts[:, :2]
        ineqs = ConvexHull(proj_pts).equations
        samples = []
        ws_bound, wa_bound = _create_bounds(pts)
        while len(samples) < self._n_samples:
            ws = rng.uniform(
                ws_bound[0], ws_bound[1], self._n_samples - len(samples)
            )
            wa = rng.uniform(
                wa_bound[0], wa_bound[1], self._n_samples - len(samples)
            )
            wind = np.column_stack((ws, wa))
            mask = np.all((ineqs[:, :2] @ wind.T).T <= -ineqs[:, 2], axis=1)
            samples.extend(wind[mask])

        return np.asarray(samples)


def _create_bounds(pts):
    ws_bound = (np.amin(pts[:, 0]), np.amax(pts[:, 0]))
    wa_bound = (np.amin(pts[:, 1]), np.amax(pts[:, 1]))
    return ws_bound, wa_bound


class FibonacciSampler(Sampler):
    """A sampler that produces sample points on a moved and scaled version
    of the spiral (sqrt(x) * cos(x), sqrt(x) * sin(x)), such that the angles
    are distributed equidistantly by the inverse golden ratio.

    The sample points all lie in the smallest enclosing circle
    of given data points.

    Inspired by Álvaro Gonzzález - "Measurement of areas on a sphere using
    Fibonacci and latitude–longitude lattices".

    Parameters
    ----------
    n_samples : positive int
        Amount of samples that will be produced by the sampler.

    Raises
    ------
    SamplerInitializationException
        If `n_samples` is non-positive.
    """

    def __init__(self, n_samples):
        if n_samples <= 0:
            raise SamplerInitializationException("`n_samples` is non-positive")

        self._n_samples = n_samples

    def sample(self, pts):
        """Produces samples according to the procedure described above.

        Parameters
        ----------
        pts : array_like of shape (n, 2)
            Points in whose convex hull the produced samples will lie.

        Returns
        -------
        samples : numpy.ndarray of shape (n_samples, 2)
            Samples produced by the procedure described above.
        """
        # calculate smallest circle containing pts
        midpoint, r = _make_circle(pts)

        # calculate an upper bound to the number of points needed for the
        # spiral to fill the convex hull with self._n_samples points
        ch = ConvexHull(pts)
        vol = ch.volume
        ineqs = ch.equations
        ub_n_samples = int(np.pi * r**2 * self._n_samples / vol) + 10

        # create big fibonacci spiral
        golden_ratio = (1 + np.sqrt(5)) / 2
        i = np.arange(1, ub_n_samples)
        beta = 2 * np.pi * i * golden_ratio ** (-1)
        radius = np.sqrt(i / ub_n_samples)
        base_spiral = radius * np.array([np.cos(beta), np.sin(beta)])

        # move and scale fibonacci spiral to the previously calculated circle
        # use binary search to rescale until number of sample points
        # in the convex hull meets the condition

        return _binary_rescale(
            self._n_samples, _sample_generator(base_spiral, midpoint, ineqs), r
        )


class ArchimedeanSampler(Sampler):
    """A sampler that produces a number of approximately equidistant
    sample points on a moved and scaled version of the archimedean spiral
    (x * cos(x), x * sin(x)).

    The sample points all lie in the smallest enclosing circle
    of given data points.

    Inspired by
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007GC001581.

    Parameters
    ----------
    n_samples : positive int
        Amount of samples that will be produced by the sampler.

    Raises
    ------
    SamplerInitializationException
        If `n_samples` is non-positive.
    """

    def __init__(self, n_samples):
        if n_samples <= 0:
            raise SamplerInitializationException("`n_samples` is non-positive")

        self._n_samples = n_samples

    def sample(self, pts):
        """Produces samples according to the procedure described above.

        Parameters
        ----------
        pts : array_like of shape (n, 2)
            Points in whose convex hull the produced samples will lie.

        Returns
        -------
        samples : numpy.ndarray of shape (n_samples, 2)
            Samples produced by the procedure described above.
        """

        # calculate enclosing circle
        midpoint, r = _make_circle(pts)

        # compute convex hull and its volume
        ch = ConvexHull(pts)
        vol = ch.volume
        ineqs = ch.equations

        ub_n_samples = int(np.pi * r**2 * self._n_samples / vol) + 10

        # estimate upper bounds for spiral points needed
        # and simultaneously create arc values
        beta = [0]
        for _ in range(ub_n_samples):
            beta.append(beta[-1] + 8 / (beta[-1] + 2))

        beta = np.array(beta)
        base_spiral = beta * np.array([np.cos(beta), np.sin(beta)])

        return _binary_rescale(
            self._n_samples, _sample_generator(base_spiral, midpoint, ineqs), r
        )


def _binary_rescale(n_samples, generate_sample, start_value):
    """"""
    # performs binary search to create a sample
    # of size n_samples by calling generate_sample(t),
    # where t is the parameter modified
    lb = 0
    ub = None
    t = start_value
    sample = generate_sample(t)

    while len(sample) != n_samples:
        if len(sample) > n_samples:
            lb = t
        else:
            ub = t
        if ub is None:
            t *= 2
        else:
            t = (ub + lb) / 2
        sample = generate_sample(t)

    return sample


def _sample_generator(base_set, midpoint, ineqs):
    """"""

    # creates a function which generates samples as a scaled base_set
    # translated in midpoint and forfilling the
    # inequalities ineqs
    def generate_sample(t):
        spiral = (midpoint[:, None] + t * base_set).transpose()
        mask = np.all((ineqs[:, :2] @ spiral.T).T <= -ineqs[:, 2], axis=1)
        return spiral[mask]

    return generate_sample


def _make_circle(pts, eps=0.0001):
    """"""
    pts = pts.copy()
    np.random.shuffle(pts)
    k = []  # list of necessary boundary points
    circle = _small_circle(pts[k])
    i = 0  # index of currently examined point
    j = 0  # index of point to be included
    while i < len(pts):
        p = pts[i]

        if _is_in_circle(p, circle, eps):
            i += 1
            j = max(i, j)
            continue

        # p is not in circle
        if i == j:
            k = [i]
            i = 0
            circle = _small_circle(pts[k])
            continue

        # p is not in circle and len(k) < 3
        k = [t for t in k if t > i]
        k.append(i)
        circle = _small_circle(pts[k])
        i = 0

    return circle


def _is_in_circle(p, circle, eps):
    """"""
    mp, r = circle
    return np.linalg.norm(p - mp) < r + eps


def _small_circle(pts):
    """"""
    if len(pts) == 0:
        return np.zeros((2,)), 0
    if len(pts) == 1:
        return pts[0], 0
    if len(pts) == 2:
        p1, p2 = pts[0], pts[1]
        mp = 1 / 2 * (p1 + p2)
        r = 1 / 2 * np.linalg.norm(p1 - p2)
        return mp, r
    if len(pts) == 3:
        circle_m = -np.column_stack(pts).T
        circle_m = np.column_stack([np.ones(3), circle_m])
        circle_b = np.array([-np.linalg.norm(p) ** 2 for p in pts])
        # TODO: handling for degenerate case
        a, b, c = np.linalg.inv(circle_m) @ circle_b
        return np.array([b / 2, c / 2]), np.sqrt(b**2 / 4 + c**2 / 4 - a)
    raise ValueError(f"number of points should be <= 3 but is {pts}")
