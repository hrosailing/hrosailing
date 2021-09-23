"""
Functions for navigation and weather routing using PPDs
"""

# Author: Valentin Dannenberg & Robert Schueler

from bisect import bisect_left
import dataclasses
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.spatial import ConvexHull

import hrosailing.polardiagram as pol
from hrosailing.pipelinecomponents import InfluenceModel


@dataclasses.dataclass
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

    ws : int / float
        The current wind speed given in knots

    wdir :
        The direction of the wind given as either

        - the wind angle relative to north
        - the true wind angle and the boat direction relative to north
        - the apparent wind angle and the boat direction relative to north
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
    out : list of tuples
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

    bsp2 = bsp[np.where(wa == d2.angle)[0][0]]

    t = dist / (d1.proportion * bsp1 + d2.proportion * bsp2)
    t1, t2 = d1.proportion * t, d2.proportion * t

    return [(d1.angle, t1), (d2.angle, t2)]


class WeatherException(Exception):
    """"""


class WeatherModel:
    """"""

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
        return dict(zip(self._attrs, mean))


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
    """
    Computes the total cost for traveling
    from a start position to an end position

    To be precise, it calculates
    for a given cost density function
    cost and absolute function abs_cost

    int_0^l cost(s, t(s)) ds + abs_cost(t(l), l),

    where s is the distance travelled,
    l is the total distance from start to end
    and t(s) is the time travelled
    t(s) is the solution of the initial value problem
    t(0) = 0, dt/ds = 1/bsp(s,t).

    The costs also depend on the weather forecast data,
    organized by a WeatherModel
    Distances are computed using the mercator projection

    Parameter
    ----------
    pd: PolarDiagram
        Polar diagram of the vessel

    start: tuple of two floats
        Coordinates of the starting point

    end: tuple of two floats
        Coordinates of the end point

    start_time: datetime.datetime
        The time at which the traveling starts

    wm: WeatherModel, optional
        The WeatherModel used

    cost_fun_dens: callable, optional
        Function giving a cost density for given time as datetime.datetime,
        lattitude as float, longitude as float and WeatherModel
        cost_fun_dens(t,lat,long,wm) corresponds to costs(s,t) above.

        Defaults to None.

    cost_fun_abs: callable, optional
        Corresponds to abs_costs above.

        Defaults to lambda total_t, total_s: total_t

    integration_method: callable, optional
        Function that takes two (n,) arrays y, x and computes
        an approximative integral from that.
        Is only used if cost_fun_dens is not None

        Defaults to scipy.integrate.trapezoid

    im: InfluenceModel, optional
        The influence model used to consider additional influences
        on the boat speed.

        Defaults to None

    ivp_kw:
        Keyword arguments which will be redirected to scipy.integrate.solve_ivp
        in order to solve the initial value problem described above

    Returns
    -------
    out : float
        The total cost calculated as described above
    """

    # TODO: default value handling for wm and im

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

    Parameter
    ----------

    pd: PolarDiagram
        The polar diagram of the used vessel

    start: 2-tuple of floats
        The lattitude and longitude of the starting point

    start_time: datetime.datetime
        The time at which the traveling starts

    direction: float
        The angle between North and the direction in which we aim to travel.

    wm: WeatherModel, optional
        The weather model used.

    total_time: float
        The time in hours that the vessel is supposed to travel
        in the given direction.

    min_nodes: int, optional
        The minimum amount of sample points to sample the position space.
        Defaults to 100.

    im: InfluenceModel, optional
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
    # TODO: Default handling of wm and im
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
        data = wm.get_weather(time, lat, long)
        data["HDT"] = hdt
    except:
        return 0
    bsp = im.add_influence(pd, data)

    if bsp != 0:
        return 1 / bsp

    return 0


# def isocost(
#     pd: pol.PolarDiagram,
#     start,
#     direction,
#     cost_func=None,
#     total_cost=None,
#     min_nodes=None,
#     quadrature=None,
#     wm: WeatherModel = None,
#     im: Optional[InfluenceModel] = None,
# ):
#     """"""


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
