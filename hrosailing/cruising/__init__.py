"""

"""


import hrosailing.polardiagram as pol
import numpy as np
from scipy.spatial import ConvexHull


def convex_direction(pd: pol.PolarDiagram, ws, direction):
    """

    """
    _, wa, bsp = pd.get_slices(ws)
    bsp = bsp.ravel()
    vert = ConvexHull(
        np.column_stack((bsp * np.cos(wa)), bsp * np.sin(wa))
    ).vertices

    # since wind angles are in rad, we
    # convert direction from deg to rad too.
    # Makes it easier to calculate distances
    direction = np.deg2rad(direction)
    edge = sorted(
        [(i, abs(wa[i] - direction)) for i in vert], key=lambda x: x[1]
    )[:2]
    if not edge[0][1]:
        i = edge[0][0]
        return [(np.rad2deg(wa[i]), 1)]
    i1, i2 = edge[0][0], edge[1][0]
    lambd = (direction - wa[i2]) / (wa[i1] - wa[i2])

    if lambd <= 0:
        raise ValueError("something went wrong")

    return [(np.rad2deg(wa[i1]), lambd), (np.rad2deg(wa[i2]), 1 - lambd)]


def cruise(pd: pol.PolarDiagram, ws, wa, start, end):
    pass


class WeatherModel:
    pass


def cost_cruise(
    pd: pol.PolarDiagram,
    wm,
    start,
    end,
    cost_func=None,
    nodes=None,
    quadrature=None,
):
    pass


def isocost(
    pd: pol.PolarDiagram,
    wm,
    start,
    direction,
    cost_func=None,
    total_cost=None,
    min_nodes=None,
    quadrature=None,
):
    pass
