"""

"""


import hrosailing.polardiagram as pol
import numpy as np
from scipy.spatial import ConvexHull


# TODO A bit cleaner, some error checks!
def convex_direction(pd: pol.PolarDiagram, ws, direction):
    _, wa, bsp = pd.get_slices(ws)
    bsp = bsp.ravel()
    kart = np.column_stack((bsp * np.cos(wa)), bsp * np.sin(wa))
    vert = ConvexHull(kart).vertices
    direction = np.deg2rad(direction)
    conv = sorted(
        [(i, abs(wa[i] - direction)) for i in vert], key=lambda x: x[1]
    )
    if not conv[0][1]:
        i = conv[0][0]
        return [(wa[i], 1)]

    i1, i2 = conv[0][0], conv[1][0]
    lambd = (direction - wa[i2]) / (wa[i1] - wa[i2])
    if lambd <= 0:
        raise ValueError("something went wrong")

    # Convert to degrees?
    return [(wa[i1], lambd), (wa[i2], 1 - lambd)]


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
