"""

"""

# Author: Valentin F. Dannenberg / Ente

import dataclasses
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull

import hrosailing.polardiagram as pol
from hrosailing.processing.pipelinecomponents import InfluenceModel


@dataclasses.dataclass
class Direction:
    angle: float
    proportion: float
    sail: Optional[str] = None


def convex_direction(
    pd: pol.PolarDiagram, ws, direction, im: Optional[InfluenceModel] = None
):
    """"""
    _, wa, bsp, *sails = pd.get_slices(ws)
    bsp = bsp.ravel()
    if im:
        bsp = im.add_influence(bsp)

    polar_pts = np.column_stack(
        (bsp * np.cos(wa).ravel(), bsp * np.sin(wa).ravel())
    )
    conv = ConvexHull(polar_pts)
    vert = sorted(conv.vertices)

    # Account for computational error?
    wa = np.rad2deg(wa)

    for left, right in zip(vert, vert[1:]):
        if wa[left] <= direction <= wa[right]:
            i1, i2 = left, right
            edge = [
                Direction(wa[i1], 1, sails[i1] if sails else None),
                Direction(wa[i2], 1, sails[i2] if sails else None),
            ]
            break
    else:
        i1, i2 = vert[0], vert[-1]
        edge = [
            Direction(wa[i1], 1, sails[i1] if sails else None),
            Direction(wa[i2], 1, sails[i2] if sails else None),
        ]

    if edge[0] == direction:
        return [edge[0]]
    elif edge[1] == direction:
        return [edge[1]]

    # direction lies on a common edge of polar diagram and convex hull
    if abs(i1 - i2) == 1 and edge[0].sail == edge[1].sail:
        return [edge[0]]

    lambda_ = (direction - wa[i2]) / (wa[i1] - wa[i2])
    if lambda_ > 1 or lambda_ < 0:
        lambda_ = (direction + 360 - wa[i2]) / (wa[i1] + 360 - wa[i2])

    edge[0].proportion = lambda_
    edge[1].proportion = 1 - lambda_
    return edge


def cruise(
    pd: pol.PolarDiagram,
    ws,
    wa,
    start,
    end,
    im: Optional[InfluenceModel] = None,
):
    """"""
    _, slice_wa, bsp, *_ = pd.get_slices(ws)

    if im:
        bsp = im.add_influence(bsp)

    direction = _right_handing_course(start, end)
    dist = _great_earth_elipsoid_distance(start, end)

    d1, *d2 = convex_direction(pd, ws, direction)

    bsp1 = bsp[np.where(slice_wa == d1.angle)[0][0]]
    if d1.proportion == 1:
        return [(d1.angle, dist / bsp1)]

    bsp2 = bsp[np.where(slice_wa == d2.angle)[0][0]]

    t = dist / (d1.proportion * bsp1 + d2.proportion * bsp2)
    t1, t2 = d1.proportion * t, d2.proportion * t

    return [(d1.angle, t1), (d2.angle, t2)]


class WeatherModel:
    pass


def cost_cruise(
    pd: pol.PolarDiagram,
    start,
    end,
    cost_func=None,
    nodes=None,
    quadrature=None,
    wm: WeatherModel = None,
    im: Optional[InfluenceModel] = None,
):
    """"""
    pass


def isocrone(
    pd: pol.PolarDiagram,
    start,
    direction,
    total_cost=None,
    min_nodes=None,
    wm: WeatherModel = None,
    im: Optional[InfluenceModel] = None,
):
    """"""
    pass


def isocost(
    pd: pol.PolarDiagram,
    start,
    direction,
    cost_func=None,
    total_cost=None,
    min_nodes=None,
    quadrature=None,
    wm: WeatherModel = None,
    im: Optional[InfluenceModel] = None,
):
    """"""
    pass


def _right_handing_course(a, b):
    numerator = np.cos(a[1]) * np.sin(b[1]) - np.cos(a[0] - b[0]) * np.cos(
        b[1]
    ) * np.sin(a[1])
    denominator = np.cos(a[0] - b[0]) * np.cos(a[1]) * np.cos(b[1]) + np.sin(
        a[1]
    ) * np.cos(b[1])

    return np.arccos(numerator / np.sqrt(1 - denominator ** 2))


EARTH_FLATTENING = 1 / 298.257223563
EQUATOR_CIRCUMFERENCE = 40075.017


def _great_earth_elipsoid_distance(a, b):
    f = (a[1] + b[1]) / 2
    g = (a[1] - b[1]) / 2
    lat = (a[0] - b[0]) / 2

    s = (np.sin(g) * np.cos(lat)) ** 2 + (np.cos(f) * np.sin(lat)) ** 2
    c = (np.cos(g) * np.cos(lat)) ** 2 + (np.sin(f) * np.sin(lat)) ** 2

    omega = np.deg2rad(np.arctan(np.sqrt(s / c)))
    d = EQUATOR_CIRCUMFERENCE * omega / np.pi

    t = np.sqrt(s * c) / omega
    h_1 = (3 * t - 1) / (2 * c)
    h_2 = (3 * t + 1) / (2 * s)

    dist = d * (
        1
        + EARTH_FLATTENING
        * (
            h_1 * (np.sin(f) * np.cos(g)) ** 2
            - h_2 * (np.cos(f) * np.sin(g)) ** 2
        )
    )
    return dist
