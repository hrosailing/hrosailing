"""

"""

# Author: Valentin F. Dannenberg / Ente
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull

import hrosailing.polardiagram as pol
from hrosailing.processing.pipelinecomponents import InfluenceModel

# TODO Maybe change some function names


def convex_direction(
    pd: pol.PolarDiagram, ws, direction, im: Optional[InfluenceModel] = None
):
    """"""
    _, wa, bsp, *members = pd.get_slices(ws)
    bsp = bsp.ravel()
    if im:
        bsp = im.add_influence(bsp)

    polar_pts = np.column_stack(
        (bsp * np.cos(wa).ravel(), bsp * np.sin(wa).ravel())
    )
    conv = ConvexHull(polar_pts)
    vert = conv.vertices

    direction = np.deg2rad(direction)
    edge = sorted(
        [
            (i, abs(wa[i] - direction), members[i] if members else 0)
            for i in vert
        ],
        key=lambda x: x[1],
    )
    edge = edge[:2]

    if not edge[0][1]:
        i = edge[0][0]
        return [(np.rad2deg(wa[i]), 1, edge[0][2])]

    i1, i2 = edge[0][0], edge[1][0]
    # if direction lies on an edge of the polar diagram, which
    # is also an edge of the convex hull, we can sail straight
    # in direction.
    if abs(i1 - i2) == 1 and edge[0][2] == edge[1][2]:
        return [(np.rad2deg(direction), 1, edge[0][2])]

    lambda_ = (direction - wa[i2]) / (wa[i1] - wa[i2])

    return [
        (np.rad2deg(wa[i1]), lambda_, edge[0][2]),
        (np.rad2deg(wa[i2]), 1 - lambda_, edge[1][2]),
    ]


def cruise(
    pd: pol.PolarDiagram,
    ws,
    wa,
    start,
    end,
    im: Optional[InfluenceModel] = None,
):
    """"""
    _, _, bsp, *members = pd.get_slices(ws)
    (wa1, lambd1), (wa2, lambd2) = convex_direction(pd, ws, wa, im)
    dist = ...
    t1, t2 = ..., ...

    return [(wa1, t1), (wa2, t2)]


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
