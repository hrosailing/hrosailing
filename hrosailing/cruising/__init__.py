"""

"""

# Author: Valentin F. Dannenberg / Ente


import numpy as np
from scipy.spatial import ConvexHull

import hrosailing.polardiagram as pol
from hrosailing.processing.pipelinecomponents.influencemodel import (
    InfluenceModel,
)

# TODO Maybe change some function names


# TODO Not yet functional for PolarDiagramMultiSails
def convex_direction(
    pd: pol.PolarDiagram, ws, direction, im: InfluenceModel = None
):
    """"""
    _, wa, bsp, *_ = pd.get_slices(ws)
    bsp = bsp.ravel()
    vert = ConvexHull(
        np.column_stack((bsp * np.cos(wa)), bsp * np.sin(wa))
    ).vertices

    direction = np.deg2rad(direction)
    edge = sorted(
        [(i, abs(wa[i] - direction)) for i in vert], key=lambda x: x[1]
    )[:2]
    if not edge[0][1]:
        i = edge[0][0]
        return [(np.rad2deg(wa[i]), 1)]

    i1, i2 = edge[0][0], edge[1][0]
    # if direction lies on an edge of the polar diagram, which
    # is also an edge of the convex hull, we can sail straight
    # in direction.
    if (
        abs(i1 - i2) == 1
        or (i1 == 0 and i2 == len(wa) - 1)
        or (i1 == len(wa) - 1 and i2 == 0)
    ):
        # Maybe doesn't work for Pointcloud?, but should
        # work for Table and Curve
        return [(np.rad2deg(direction), 1)]

    lambd = (direction - wa[i2]) / (wa[i1] - wa[i2])
    if lambd <= 0:
        raise ValueError("something went wrong")

    return [(np.rad2deg(wa[i1]), lambd), (np.rad2deg(wa[i2]), 1 - lambd)]


def cruise(
    pd: pol.PolarDiagram, ws, wa, start, end, im: InfluenceModel = None
):
    """"""
    _, _, bsp, *_ = pd.get_slices(ws)
    (wa1, lambd1), (wa2, lambd2) = convex_direction(pd, ws, wa)
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
    im: InfluenceModel = None,
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
    im: InfluenceModel = None,
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
    im: InfluenceModel = None,
):
    """"""
    pass
