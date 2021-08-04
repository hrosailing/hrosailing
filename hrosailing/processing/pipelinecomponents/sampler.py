"""
Defines a baseclass for samplers used in the
processing.processing.PolarPipeline class,
that can be used to create custom sampler for use.

Also contains various predefined and usable samplers
"""

# Author: Valentin F. Dannenberg / Ente

import math
import numpy as np
import random

from abc import ABC, abstractmethod
from scipy.spatial import ConvexHull


class SamplerException(Exception):
    pass


class Sampler(ABC):
    """Base class for all sampler classes


    Abstract Methods
    ----------------
    sample(self, pts)
    """

    @abstractmethod
    def sample(self, pts):
        pass


# TODO: Other random samplers
class UniformRandomSampler(Sampler):
    """A sampler that produces a number of uniformly distributed samples,
    which all lie in the convex hull of certain given points

    Parameters
    ----------
    n_samples : positive int
        Amount of samples that will be produced by the sampler

    Methods
    -------
    sample(self, pts):
        Produces samples according to the above described procedure
    """

    def __init__(self, n_samples):

        if not isinstance(n_samples, int) or n_samples <= 0:
            raise SamplerException(
                f"The number of samples needs to be a positive"
                f"number but {n_samples} was passed"
            )

        self._n_samples = n_samples

    def __repr__(self):
        return f"UniformRandomSampler(no_samples={self._n_samples})"

    def sample(self, pts):
        """Produces samples according to the above described procedure

        Parameters
        ----------
        pts : array_like of shape (n, 2)
            Points in whose convex hull the produced samples will lie

        Returns
        -------
        samples : numpy.ndarray of shape (no_samples, 2)
            samples produced by the above described method
        """
        rng = np.random.default_rng()
        pts = np.asarray(pts)[:, :2]
        ineqs = ConvexHull(pts).equations
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


# TODO
class FibonacciSampler(Sampler):
    """"""

    def __init__(self, n_samples):
        self._no_samples = n_samples

    def sample(self, pts):
        pass


# TODO Error Checks?
class ArchimedianSampler(Sampler):
    """A sampler that produces a number of
    approximately equidistant sample points on the archimedean spiral (x*cos(x), x*sin(x)),
    which all lie in the smallest enclosing circle of given data points
    Parameters
    ----------
    n_samples : positive int
        Amount of samples that will be produced by the sampler
    Methods
    -------
    sample(self, pts):
        Produces samples according to the above described procedure
    """

    def __init__(self, n_samples):

        if not isinstance(n_samples, int) or n_samples <= 0:
            raise SamplerException(
                f"The number of samples needs to be a positive"
                f"number but {n_samples} was passed"
            )

        self._n_samples = n_samples

    def sample(self, pts):
        """Produces samples according to the above described procedure
        Parameters
        ----------
        pts : array_like of shape (n, 2)
            Points in whose convex hull the produced samples will lie
        Returns
        -------
        samples : numpy.ndarray of shape (no_samples, 2)
            samples produced by the above described method
        """

        # find smallest enclosing circle
        x, y, r = make_circle(pts)
        midpoint = np.array([x, y])

        # create archimedean spiral with approximately equidistant steps
        beta = [0]
        for i in range(self._n_samples):
            beta.append(beta[-1] + 8 / (beta[-1] + 2))
        beta = np.array(beta)
        spiral = beta * np.array([np.cos(beta), np.sin(beta)])
        # translate and scale spiral to enclosing circle
        return (midpoint[:, None] + r / beta[-1] * spiral).transpose()


###### The following code has been copied from an extern source ########
###### and changed a bit                                        ########

# Smallest enclosing circle
#
# Copyright (c) 2014 Project Nayuki
# https://www.nayuki.io/page/smallest-enclosing-circle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program (see COPYING.txt).
# If not, see <http://www.gnu.org/licenses/>.
#


# Data conventions: A point is a pair of floats (x, y). A circle is a triple of floats (center x, center y, radius).

#
# Returns the smallest circle that encloses all the given points. Runs in expected O(n) time, randomized.
# Input: A sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].
# Output: A triple of floats representing a circle.
# Note: If 0 points are given, None is returned. If 1 point is given, a circle of radius 0 is returned.
#
def make_circle(pts):
    # Convert to float and randomize order
    shuffled = [(float(p[0]), float(p[1])) for p in pts]
    random.shuffle(shuffled)

    # Progressively add points to circle or recompute circle
    c = None
    for (i, p) in enumerate(shuffled):
        if c is None or not _is_in_circle(c, p):
            c = _make_circle_one_point(shuffled[0 : i + 1], p)
    return c


# One boundary point known
def _make_circle_one_point(pts, p):
    c = (p[0], p[1], 0.0)
    for (i, q) in enumerate(pts):
        if not _is_in_circle(c, q):
            if c[2] == 0.0:
                c = _make_diameter(p, q)
            else:
                c = _make_circle_two_points(pts[0 : i + 1], p, q)
    return c


# Two boundary points known
def _make_circle_two_points(pts, p, q):
    diameter = _make_diameter(p, q)
    if all(_is_in_circle(diameter, r) for r in pts):
        return diameter

    left = None
    right = None
    for r in pts:
        cross = _cross_product(p[0], p[1], q[0], q[1], r[0], r[1])
        c = _make_circumcircle(p, q, r)
        if c is None:
            continue
        elif cross > 0.0 and (
            left is None
            or _cross_product(p[0], p[1], q[0], q[1], c[0], c[1])
            > _cross_product(p[0], p[1], q[0], q[1], left[0], left[1])
        ):
            left = c
        elif cross < 0.0 and (
            right is None
            or _cross_product(p[0], p[1], q[0], q[1], c[0], c[1])
            < _cross_product(p[0], p[1], q[0], q[1], right[0], right[1])
        ):
            right = c

    return (
        left
        if (right is None or (left is not None and left[2] <= right[2]))
        else right
    )


def _make_circumcircle(p0, p1, p2):
    # Mathematical algorithm from Wikipedia: Circumscribed circle
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
    if d == 0.0:
        return None
    x = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / d
    y = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / d
    return x, y, math.hypot(x - ax, y - ay)


def _make_diameter(p0, p1):
    return (
        (p0[0] + p1[0]) / 2.0,
        (p0[1] + p1[1]) / 2.0,
        math.hypot(p0[0] - p1[0], p0[1] - p1[1]) / 2.0,
    )


_EPSILON = 1e-12


def _is_in_circle(c, p):
    return (
        c is not None
        and math.hypot(p[0] - c[0], p[1] - c[1]) < c[2] + _EPSILON
    )


# Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2)
def _cross_product(x0, y0, x1, y1, x2, y2):
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
