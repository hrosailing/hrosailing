"""
Functions for navigation and weather routing using PPDs
"""

# Author: Valentin Dannenberg & Robert Schueler

import itertools
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.spatial import ConvexHull

import hrosailing.polardiagram as pol
from hrosailing.pipelinecomponents import InfluenceModel


@dataclass
class Direction:
    """Dataclass to represent sections of a sailing maneuver"""

    # Angle to the wind direction in degrees
    angle: float

    # Proportion of time needed to sail into direction
    proportion: float

    # Type/Name of Sail that should be hissed, when
    # sailing in the direction (if existent)
    sail: Optional[str] = None

    def __str__(self):
        stc = (
            f"Sail with an angle of {self.angle} to the wind for "
            f"{self.proportion * 100} percent of the time"
        )

        if self.sail:
            stc += f", while hissing {self.sail}"

        return stc


def convex_direction(
    pd: pol.PolarDiagram,
    ws,
    direction,
    im: Optional[InfluenceModel] = None,
    influence_data: Optional[dict] = None,
) -> List[Direction]:
    """Given a direction, computes the "fastest" way to sail in
    that direction, assuming constant wind speed `ws`

    If sailing straight into direction is the fastest way, function
    returns that direction. Otherwise function returns two directions
    aswell as their proportions, such that sailing into one direction for
    a corresponding proportion of a time segment and then into the other
    direction for a corresponding proportion of a time segment will be
    equal to sailing into `direction` but faster.

    Parameters
    ----------
    pd : PolarDiagram
        The polar diagram of the vessel

    ws : int / float
        The current wind speed given in knots

    direction : int / float
        Angle to the wind direction

    im : InfluenceModel, optional
        The influence model used to consider additional influences
        on the boat speed

        Defaults to `None`

    influence_data: dict, optional
        Data containing information that might influence the boat speed
        of the vessel (eg. current, wave height), to be passed to
        the used influence model

        Only used, if `im` is not `None`

        Defaults to `None`

    Returns
    -------
    edge : list of Directions
        Either just one Direction instance, if sailing into `direction`
        is the optimal way, or two Direction instances, that will "equal"
        to `direction`
    """
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

    if edge[1] == direction:
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
    """Given a starting point A and and end point B,the function calculates
    the fastest time and sailing direction it takes for a sailing vessel to
    reach B from A, under constant wind.

    If needed the function will calculate two directions as well as the
    time needed to sail in each direction to get to B.

    Parameters
    ----------
    pd : PolarDiagram
        The polar diagram of the vessel

    ws : int or float
        The current wind speed given in knots

    wdir : See below
        The direction of the wind given as either

        - the wind angle relative to north
        - the true wind angle and the boat direction relative to north
        - a (ugrd, vgrd) tuple from grib data

    start : tuple of length 2
        Coordinates of the starting point of the cruising maneuver,
        given in longitude and latitude

    end : tuple of length 2
        Coordinates of the end point of the cruising maneuver,
        given in longitude and latitude

    im : InfluenceModel, optional
        The influence model used to consider additional influences
        on the boat speed

        Defaults to `None`

    influence_data: dict, optional
        Data containing information that might influence the boat speed
        of the vessel (eg. current, wave height), to be passed to
        the used influence model

        Only used, if `im` is not `None`

        Defaults to `None`

    Returns
    -------
    directions : list of tuples
        Directions as well as the time needed to sail along those,
        to get from start to end
    """
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

    d2 = d2[0]
    bsp2 = bsp[np.where(wa == d2.angle)[0][0]]

    t = dist / (d1.proportion * bsp1 + d2.proportion * bsp2)
    t1, t2 = d1.proportion * t, d2.proportion * t

    return [(d1.angle, t1), (d2.angle, t2)]


class OutsideGridException(Exception):
    """Exception raised if point accessed in weather model lies
    outside the available grid"""


class WeatherModel:
    """Models a weather model as a 3-dimensional space-time grid
    where each space-time point has certain values of a given list
    of attributes

    Parameters
    ----------
    data : array_like of shape (n, m, r, s)
        Weather data at different space-time grid points

    times : list of length n
        Sorted list of time values of the space-time grid

    lats : list of length m
        Sorted list of lattitude values of the space-time grid

    lons : list of length r
        Sorted list of longitude values of the space-time grid

    attrs : list of length s
        List of different (scalar) attributes of weather
    """

    def __init__(self, data, times, lats, lons, attrs):
        self._times = times
        self._lats = lats
        self._lons = lons
        self._attrs = attrs
        self._data = data

    def _grid(self):
        return self._times, self._lats, self._lons

    def get_weather(self, point):
        """Given a space-time point, uses the available weather model
        to calculate the weather at that point

        If the point is not a grid point, the weather data will be
        affinely interpolated, starting with the time-component, using
        the (at most) 8 grid points that span the vertices of a cube, which
        contains the given point

        Parameters
        ----------
        point: tuple of length 3
            Space-time point given as tuple of time, lattitude
            and longitude

        Returns
        -------
        weather : dict
            The weather data at the given point.

            If it is a grid point, the weather data is taken straight
            from the model, else it is interpolated as described above
        """
        # check if given point lies in the grid
        fst = (self._times[0], self._lats[0], self._lons[0])
        lst = (self._times[-1], self._lats[-1], self._lons[-1])

        outside_left = [pt < left for pt, left in zip(point, fst)]
        outside_right = [pt > right for pt, right in zip(point, lst)]

        if any(outside_left) or any(outside_right):
            raise OutsideGridException(
                "`point` is outside the grid. Weather data not available."
            )

        grid = self._grid()
        idxs = [
            bisect_left(grid_comp, comp)
            for grid_comp, comp in zip(grid, point)
        ]
        flags = [
            grid_pt[idx] == pt
            for grid_pt, idx, pt in zip(
                grid,
                idxs,
                point,
            )
        ]

        cuboid = [
            [idx - 1, idx] if not flag else [idx]
            for idx, flag in zip(idxs, flags)
        ]

        cuboid = np.meshgrid(*cuboid)
        idxs = np.vstack(tuple(map(np.ravel, cuboid))).T

        val = _interpolate_weather_data(self._data, idxs, point, flags, grid)
        return dict(zip(self._attrs, val))


def cost_cruise(
    pd: pol.PolarDiagram,
    start,
    end,
    start_time: datetime,
    wm: WeatherModel,
    cost_fun_dens=None,
    cost_fun_abs=lambda total_t, total_s: total_t,
    integration_method=trapezoid,
    im: Optional[InfluenceModel] = None,
    **ivp_kw,
):
    """Computes the total cost for traveling from a start position to an
    end position. To be precise, it calculates for a given cost density
    function cost and absolute function abs_cost

    int_0^l cost(s, t(s)) ds + abs_cost(t(l), l),

    where s is the distance travelled, l is the total distance from
    start to end and t(s) is the time travelled.
    t(s) is the solution of the initial value problem

    t(0) = 0, dt/ds = 1/bsp(s,t).

    The costs also depend on the weather forecast data, organized
    by a WeatherModel, distances are computed using the mercator projection

    Parameters
    ----------
    pd : PolarDiagram
        Polar diagram of the vessel

    start : tuple of two floats
        Coordinates of the starting point

    end : tuple of two floats
        Coordinates of the end point

    start_time : datetime.datetime
        The time at which the traveling starts

    wm : WeatherModel, optional
        The WeatherModel used

    cost_fun_dens : callable, optional
        Function giving a cost density for given time as datetime.datetime,
        lattitude as float, longitude as float and WeatherModel
        cost_fun_dens(t,lat,long,wm) corresponds to costs(s,t) above

        Defaults to `None`

    cost_fun_abs : callable, optional
        Corresponds to `abs_costs`

        Defaults to `lambda total_t, total_s: total_t`

    integration_method : callable, optional
        Function that takes two (n,) arrays y, x and computes
        an approximative integral from that.
        Is only used if `cost_fun_dens` is not None

        Defaults to `scipy.integrate.trapezoid`

    im : InfluenceModel, optional
        The influence model used to consider additional influences
        on the boat speed

        Defaults to `None`

    ivp_kw :
        Keyword arguments which will be passed to scipy.integrate.solve_ivp
        in order to solve the initial value problem described above

    Returns
    -------
    cost : float
        The total cost calculated as described above
    """
    # pylint: disable=too-many-locals

    lat_mp = (start[0] + end[0]) / 2
    proj_start = _mercator_proj(start, lat_mp)
    proj_end = _mercator_proj(end, lat_mp)
    total_s = np.linalg.norm(proj_end - proj_start)

    hdt = _right_handing_course(start, end)

    # define derivative of t by s
    def dt_ds(s, t):
        pos = proj_start + s / total_s * (proj_end - proj_start)
        return _get_inverse_bsp(pd, pos, hdt, t[0], lat_mp, start_time, wm, im)

    t_s = solve_ivp(
        fun=dt_ds,
        t_span=(0, np.linalg.norm(proj_start - proj_end)),
        y0=np.zeros(1),
        **ivp_kw,
    )

    # calculate absolute cost and return it if sufficient
    total_t = t_s.y[0][-1]  # last entry of IVP solution
    absolute_cost = cost_fun_abs(total_t, total_s)

    if not cost_fun_dens:
        return absolute_cost

    # calculate the integral described in the doc string
    pos_list = [
        proj_start + s / total_s * (proj_end - proj_start) for s in t_s.t
    ]
    lat_long_list = [_inverse_mercator_proj(pos, lat_mp) for pos in pos_list]
    t_list = [start_time + timedelta(hours=t) for t in t_s.y[0]]

    costs = [
        cost_fun_dens(t, lat, long, wm)
        for t, (lat, long) in zip(t_list, lat_long_list)
    ]

    return absolute_cost + integration_method(costs, t_s.t)


def isocrone(
    pd: pol.PolarDiagram,
    start,
    start_time,
    direction,
    wm: WeatherModel,
    total_time=1,
    min_nodes=100,
    im: Optional[InfluenceModel] = None,
):
    """
    Estimates the maximum distance that can be reached from a given start
    point in a given amount of time without tacks and jibes.
    This is done by sampling the position space and using mercator projection.
    A weather forecast, organized by a WeatherModel and an InfluenceModel
    are included in the computation.

    Parameters
    ----------
    pd : PolarDiagram
        The polar diagram of the used vessel

    start : 2-tuple of floats
        The lattitude and longitude of the starting point

    start_time : datetime.datetime
        The time at which the traveling starts

    direction : float
        The angle between North and the direction in which we aim to travel.

    wm : WeatherModel, optional
        The weather model used.

    total_time : float
        The time in hours that the vessel is supposed to travel
        in the given direction.

    min_nodes : int, optional
        The minimum amount of sample points to sample the position space.
        Defaults to 100.

    im : InfluenceModel, optional
        The influence model used.
        Defaults to ??.

    Returns
    -------
    end : 2-tuple of floats
        Lattitude and Longitude of the position that is reached when traveling
        total_time hours in the given direction

    s : float
        The length of the way traveled from start to end
    """
    # estimate first sample points as equidistant points

    lat_mp = start[0]
    proj_start = _mercator_proj(start, lat_mp)
    arc = np.pi * (1 / 2 - direction / 180)
    v_direction = np.array([np.cos(arc), np.sin(arc)])

    def dt_ds(s, t):
        pos = proj_start + s * v_direction
        return _get_inverse_bsp(
            pd, pos, direction, t, lat_mp, start_time, wm, im
        )

    # supposed boat speed for first estimation is 5 knots
    step_size = 5 * total_time / min_nodes
    s, t, steps = 0, 0, 0

    der = 0  # debug

    while t < total_time or steps < min_nodes:
        if t >= total_time:
            # start process again with smaller step size
            step_size *= steps / min_nodes
            s, t, steps = 0, 0, 0
            continue
        der = dt_ds(s, t)
        s += step_size
        t += der * step_size
        steps += 1

    # we end up with s, t such that t >= total_cost and steps > min_nodes
    # still need to correct the last step such that t == total_cost

    s = (total_time + der * s - t) / der

    proj_end = proj_start + s * v_direction
    end = _inverse_mercator_proj(proj_end, lat_mp)

    return end, s


def _inverse_mercator_proj(pt, lat_mp):
    # computes lattitude and longitude of a projected point
    # where the projection midpoint has lattitude lat_mp
    x, y = pt / 69
    return x + lat_mp, 180 / np.pi * np.arcsin(np.tanh(y))


def _mercator_proj(pt, lat_mp):
    # projects a point given as lattitude and longitude tupel using mercator
    # projection where the projection midponit has lattitude lat_mp
    lat, long = pt
    # 69 nautical miles between two lattitudes
    return 69 * np.array(
        [(lat - lat_mp), np.arcsinh(np.tan(np.pi * long / 180))]
    )


def _get_inverse_bsp(pd, pos, hdt, t, lat_mp, start_time, wm, im):
    lat, long = _inverse_mercator_proj(pos, lat_mp)
    time = start_time + timedelta(hours=t)
    try:
        data = wm.get_weather((time, lat, long))
        data["HDT"] = hdt
    except OutsideGridException:
        return 0
    if im:
        bsp = im.add_influence(pd, data)
    else:
        ugrid, vgrid = data["UGRID"], data["VGRID"]
        tws, twa = _uvgrid_to_tw(ugrid, vgrid, hdt)
        bsp = pd(tws, twa)

    if bsp != 0:
        return 1 / bsp

    return 0


def _interpolate_weather_data(data, idxs, point, flags, grid):
    # point is a grid point
    if len(idxs) == 1:
        i, j, k = idxs.T
        return data[i, j, k, :]

    # lexicograpic first and last vertex of cube
    start = idxs[0]
    end = idxs[-1]

    # interpolate along time edges first
    if flags[0] and flags[1] and not flags[2]:
        idxs[[1, 2]] = idxs[[2, 1]]

    face = [i for i, flag in enumerate(flags) if not flag]

    if len(face) == 1:
        edges = [idxs[0], idxs[1]]
    else:
        edges = [0, 1] if len(face) == 2 else [0, 1, 4, 5]
        edges = [(idxs[i], idxs[i + 2]) for i in edges]
        flatten = itertools.chain.from_iterable
        edges = list(flatten(edges))

    interim = [data[i, j, k, :] for i, j, k in edges]

    for i in face:
        mu = (point[i] - grid[i][end[i]]) / (
            grid[i][start[i]] - grid[i][end[i]]
        )
        it = iter(interim)
        interim = [mu * left + (1 - mu) * right for left, right in zip(it, it)]

    return interim[0]


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

    # gribdata:
    # wdir = 180 / pi * np.arctan2(vgrd, ugrd) + 180

    # twa + bd:
    # wdir = (rwSK + twa) % 360 ?


def _uvgrid_to_tw(ugrid, vgrid, hdt):
    # TODO: check
    tws = np.sqrt(ugrid ** 2 + vgrid ** 2)
    wa = (180 + 180 / np.pi * np.arctan2(vgrid, ugrid)) % 360
    twa = (hdt - wa) % 360
    return tws, twa


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
