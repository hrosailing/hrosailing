"""

"""

# Author: Valentin Dannenberg & Robert Schueler

from bisect import bisect_left
import dataclasses
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull

import hrosailing.polardiagram as pol
from hrosailing.pipelinecomponents import InfluenceModel


@dataclasses.dataclass
class Direction:
    angle: float
    proportion: float
    sail: Optional[str] = None

    def __str__(self):
        out = (
            f"Sail with an angle of {self.angle} to the wind for "
            f"{self.proportion} percent of the time"
        )

        if self.sail:
            out += f", while hissing {self.sail}"

        return out


def convex_direction(
    pd: pol.PolarDiagram,
    ws,
    direction,
    im: Optional[InfluenceModel] = None,
    influence_data: Optional[dict] = None,
):
    """"""
    _, wa, bsp, *sails = pd.get_slices(ws)
    if im:
        bsp = im.add_influence(pd, influence_data)

    bsp = bsp.ravel()

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
            edge = [Direction(wa[i1], 1), Direction(wa[i2], 1)]
            break
    else:
        i1, i2 = vert[0], vert[-1]
        edge = [Direction(wa[i1], 1), Direction(wa[i2], 1)]

    if sails:
        edge[0].sail = sails[i1]
        edge[1].sail = sails[i2]

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
    wdir,
    start,
    end,
    im: Optional[InfluenceModel] = None,
    influence_data: Optional[dict] = None,
):
    """"""
    _, wa, bsp, *_ = pd.get_slices(ws)

    if im:
        bsp = im.add_influence(pd, influence_data)
    bsp = bsp.ravel()

    rhc = _right_handing_course(start, end)
    wdir = _wind_relative_to_north(wdir)

    heading = np.arccos(
        np.cos(rhc) * np.cos(wdir) + np.sin(rhc) * np.sin(wdir)
    )
    heading = 180 - np.rad2deg(heading)
    d1, *d2 = convex_direction(pd, ws, heading)

    dist = _great_earth_elipsoid_distance(start, end)

    bsp1 = bsp[np.where(wa == d1.angle)[0][0]]
    if not d2:
        return [(d1.angle, dist / bsp1)]

    bsp2 = bsp[np.where(wa == d2.angle)[0][0]]

    t = dist / (d1.proportion * bsp1 + d2.proportion * bsp2)
    t1, t2 = d1.proportion * t, d2.proportion * t

    return [(d1.angle, t1), (d2.angle, t2)]


class WeatherException(Exception):
    """"""


class WeatherModel:
    def __init__(self, data, times, lats, lons, attrs):
        self._times = times
        self._lats = lats
        self._lons = lons
        self._attrs = attrs
        self._data = data

    def get_weather(self, time, lat, lon):
        time_idx = bisect_left(self._times, time)
        lat_idx = bisect_left(self._lats, lat)
        lon_idx = bisect_left(self._lons, lon)

        if any(
            [
                time_idx == len(self._times),
                lat_idx == len(self._lats),
                lon_idx == len(self._lons),
            ]
        ):
            raise WeatherException(
                "As Prof. Oak said, there is a time and place for everything"
            )

        g = np.meshgrid(
            [time_idx, time_idx + 1],
            [lat_idx, lat_idx + 1],
            [lon_idx, lon_idx + 1],
        )
        idxs = np.vstack(tuple(map(np.ravel, g))).T

        mean = np.mean([self._data[i, j, k, :] for i, j, k in idxs], axis=0)
        return {attr: val for attr, val in zip(self._attrs, mean)}


def cost_cruise(
    pd: pol.PolarDiagram,
    start,
    end,
    cost_func=None,
    nodes=None,
    quadrature=None,
    wm: WeatherModel = None,
    im: Optional[InfluenceModel] = None,
    influence_data: Optional[dict] = None,
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
    influence_data: Optional[dict] = None,
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
    influence_data: Optional[dict] = None,
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


def _wind_relative_to_north(wdir):
    return wdir


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
