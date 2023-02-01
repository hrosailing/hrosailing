"""
Functions for navigation and weather routing using polar diagrams.
"""


import itertools
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.spatial import ConvexHull

from hrosailing.cruising.weather_model import (
    GriddedWeatherModel,
    OutsideGridException,
    WeatherModel,
)
from hrosailing.pipelinecomponents import InfluenceModel


class CruisingException(Exception):
    """Exception which will be raised if a non-Standard error in a cruising
    method occurs."""


@dataclass
class Direction:
    """Dataclass to represent recommended sections of a sailing maneuver."""

    #: Right headed angle between the boat heading and the wind direction.
    #:   Same as TWA but from the boat's perspective.
    angle: float

    #: The recommended proportion of time needed to sail into this direction.
    #: Given as number between 0 and 1.
    proportion: float

    #: Type/Name of sail that should be hissed, when
    #: sailing in the direction (if existent)
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
    pd,
    ws,
    direction,
    im: Optional[InfluenceModel] = None,
    influence_data: Optional[dict] = None,
) -> List[Direction]:
    """Given a direction, computes the "fastest" way to sail in
    that direction, assuming constant wind speed `ws`.

    If sailing straight into direction is the fastest way, function
    returns that direction. Otherwise, function returns two directions
    as well as their proportions, such that sailing into one direction for
    a corresponding proportion of a time segment and then into the other
    direction for a corresponding proportion of a time segment will be
    equal to sailing into `direction` but faster.

    Parameters
    ----------
    pd : PolarDiagram
        The polar diagram of the vessel.
    ws : int or float
        The current wind speed.
    direction : int or float
        Right handed angle between the heading of the boat and
        the negative of the wind direction.
        Numerically equals TWA, but interpreted from the perspective of the
        boat.
    im : InfluenceModel, optional
        The influence model used to consider additional influences
        on the boat speed.
        Defaults to `None`.
    influence_data : dict, optional
        Data containing information that might influence the boat speed
        of the vessel (e.g. current, wave height), to be passed to
        the used influence model.
        Will only be used if `im` is not `None`.
        Defaults to `None`.

    Returns
    -------
    edge : list of Directions
        Either just one `Direction` instance, if sailing into `direction`
        is the optimal way, or two `Direction` instances, that will be "equal"
        to `direction`.

    Raises
    -------
    CruisingException
        If the given polar diagram slice can not be evaluated in the given
        direction. For example, this could be the case, if the polar diagram
        only has data for angles between 0 and 180 degrees.
    """
    _, wa, bsp, *sails = pd.get_slices(ws)
    if im:
        bsp = im.add_influence(pd, influence_data)
    bsp = np.array(bsp).ravel()
    wa = np.array(wa).ravel()

    polar_pts = np.column_stack(
        (bsp * np.cos(wa).ravel(), bsp * np.sin(wa).ravel())
    )
    conv = ConvexHull(polar_pts)
    vert = sorted(conv.vertices)

    wa = np.rad2deg(wa)

    for left, right in zip(vert, vert[1:]):
        if wa[left] <= direction <= wa[right]:
            i1, i2 = left, right
            edge = [Direction(wa[i1], 1), Direction(wa[i2], 1)]
            break
    else:
        i1, i2 = vert[0], vert[-1]
        if abs(wa[i1] - wa[i2]) < 180:
            raise CruisingException(
                "The given direction is not supported by the given"
                " polar_diagram."
            )
        edge = [Direction(wa[i1], 1), Direction(wa[i2], 1)]

    if sails:
        edge[0].sail = sails[0][i1]
        edge[1].sail = sails[0][i2]

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
    pd,
    start,
    end,
    wind,
    wind_fmt="ws_wan",
    im: Optional[InfluenceModel] = None,
    influence_data: Optional[dict] = None,
):
    """Calculates the fastest time and sailing direction for a vessel to reach `end`
    from `start`, under constant wind.

    If needed the function will calculate two directions as well as the
    time needed to sail in each direction to get to `end`.

    Wind has to be given by one of the following combinations of parameters:

    - `ws` and `wa_north`,
    - `ws`, `wa` and `hdt`,
    - `uv_grd`.

    Parameters
    ----------
    pd : PolarDiagram
        The polar diagram of the vessel.
    start : tuple of length 2
        Coordinates of the starting point of the cruising maneuver,
        given in longitude and latitude.
    end : tuple of length 2
        Coordinates of the end point of the cruising maneuver,
        given in longitude and latitude.
    wind: tuple
        Description of the wind. The exact interpretation depends on
        `wind_fmt`. See the description of `wind_fmt` for details.
    wind_fmt: {"ws_wan", ws_wa_hdt", "uv_grd"}, default: `"ws_wan"`
        Specification how to interpret the parameter `wind`.

        - "ws_wan": `wind` is interpreted as
            (true wind speed, wind angle relative to north),
        - "ws_wa_hdt": `wind` is interpreted as
            (true wind speed, true wind angle,
            heading of the boat relative to north),
        - "uv_grd": `wind` is interpreted as (u_grd, v_grd) as can be read from
            a GRIB file.
    im : InfluenceModel, optional
        The influence model used to consider additional influences
        on the boat speed.
        Defaults to `None`.
    influence_data : dict, optional
        Further data to be passed to the used influence model.
        Only use data which does not depend on the wind and the heading.

        Will only be used if `im` is not `None`.

        Defaults to `None`.

    Returns
    -------
    directions : list of tuples
        Directions as well as the time (in hours) needed to sail along those,
        to get from start to end.

    Raises
    -------
    AttributeError
        If `wind_fmt` is not a supported string.
    """

    ws, wdir = _wind_relative_to_north(wind, wind_fmt)

    _, wa, bsp, *_ = pd.get_slices(ws)
    wa = np.rad2deg(wa)
    if im:
        for key, val in influence_data.items():
            influence_data[key] = [val] * len(wa)
        influence_data["TWS"] = [ws] * len(wa)
        influence_data["TWA"] = wa
        bsp = im.add_influence(pd, influence_data)
    bsp = np.array(bsp).ravel()

    rhc = _right_handing_course(start, end)

    heading = np.arccos(
        np.cos(rhc) * np.cos(wdir) + np.sin(rhc) * np.sin(wdir)
    )
    heading = 180 - np.rad2deg(heading)
    d1, *d2 = convex_direction(pd, ws, heading)

    dist = _great_earth_ellipsoid_distance(start, end)

    bsp1 = bsp[np.where(wa == d1.angle)[0]]
    if not d2:
        return [(d1.angle, float(dist / bsp1))]

    d2 = d2[0]
    bsp2 = bsp[np.where(wa == d2.angle)[0]]

    t = dist / (d1.proportion * bsp1 + d2.proportion * bsp2)
    t1, t2 = d1.proportion * t, d2.proportion * t

    return [(d1.angle, float(t1)), (d2.angle, float(t2))]


def cost_cruise(
    pd,
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
    end position. To be precise:
    Let 'l' be the total distance of the start position and the end position,
    'cost' be a density cost function describing the costs generated at each
    point along the way (for example the indicator function for bad
    weather) and 'abs_cost' be a cost function describing the cost independent
    of the weather along the way.
    Note that 'abs_cost' only depends on the expected travel time and the
    expected travel distance.

    The method first approximates the travelled time (t)
    as a function dependent on distance travelled (s) by numerically solving
    the initial value problem

    ..math::
        t(0) = 0, \\frac{dt}{ds} = \\frac{1}{bsp(s,t)}.

    Using this, it then uses numeric integration to predict the total costs as

    ..math::
        int_{0}^{l} cost(s, t(s)) \\,ds + abs\\_cost(t(l), l).

    Note that the costs in this mathematical description indirectly depend on
    weather forecast data, organized by a `WeatherModel`.
    Distances are computed using the mercator projection.

    Parameters
    ----------
    pd : PolarDiagram
        Polar diagram of the vessel.
    start : tuple of two floats
        Coordinates of the starting point.
    end : tuple of two floats
        Coordinates of the end point.
    start_time : datetime.datetime
        The time at which the traveling starts.
    wm : WeatherModel
        The weather model used. Needs to support the keys 'TWA' and 'TWS'.
    cost_fun_dens : callable, optional
        Function giving a cost density for given time as `datetime.datetime`,
        latitude as float, longitude as float and WeatherModel.
        `cost_fun_dens(t,lat,long,wm)` corresponds to `costs(s,t)` above.
        Defaults to `None`.
    cost_fun_abs : callable, optional
        Corresponds to `abs_costs`.
        Defaults to `lambda total_t, total_s: total_t`.
    integration_method : callable, optional
        Function that takes two (n,) arrays y, x and computes
        an approximate integral from that.
        Will only be used if `cost_fun_dens` is not `None`.
        Defaults to `scipy.integrate.trapezoid`.
    im : InfluenceModel, optional
        The influence model used to consider additional influences
        on the boat speed.
        Defaults to `None`.
    ivp_kw : Keyword arguments
        Keyword arguments which will be passed to `scipy.integrate.solve_ivp`
        in order to solve the initial value problem described above.

    Returns
    -------
    cost : float
        The total cost calculated as described above.
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
        inv_bsp = _get_inverse_bsp(
            pd, pos, hdt, t[0], lat_mp, start_time, wm, im
        )
        return inv_bsp

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


def isochrone(
    pd,
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
    Weather forecast data, organized by a `WeatherModel` and an `InfluenceModel`,
    is included in the computation.

    Parameters
    ----------
    pd : PolarDiagram
        The polar diagram of the used vessel.
    start : 2-tuple of floats
        The latitude and longitude of the starting point.
    start_time : datetime.datetime
        The time at which the traveling starts.
    direction : float
        The angle between North and the direction in which we aim to travel.
    wm : WeatherModel
        The weather model used. Needs to support the keys 'TWA' and 'TWS'.
    total_time : float, optional
        The time in hours that the vessel is supposed to travel
        in the given direction.
        Defaults to 1.
    min_nodes : int, optional
        The minimum amount of sample points to sample the position space.
        Defaults to 100.
    im : InfluenceModel, optional
        The influence model used.

        Defaults to `None`.

    Returns
    -------
    end : 2-tuple of floats
        Latitude and longitude of the position that is reached when traveling
        `total_time` hours in the given direction.

    s : float
        The length of the way traveled from start to end.
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

    # we end up with s, t such that t >= total_time and steps > min_nodes
    # still need to correct the last step such that t == total_time

    s = (total_time + der * s - t) / der

    proj_end = proj_start + s * v_direction
    end = _inverse_mercator_proj(proj_end, lat_mp)

    return end, s


def _inverse_mercator_proj(pt, lat_mp):
    """
    Computes point from its mercator projection with reference point `lat_mp`.
    """
    x, y = pt / 69
    return x + lat_mp, 180 / np.pi * np.arcsin(np.tanh(y))


def _mercator_proj(pt, lat_mp):
    """
    Computes the mercator projection with reference point `lat_mp` of a point.
    """
    lat, long = pt

    # 69 nautical miles between two latitudes
    return 69 * np.array(
        [(lat - lat_mp), np.arcsinh(np.tan(np.pi * long / 180))]
    )


def _get_inverse_bsp(pd, pos, hdt, t, lat_mp, start_time, wm, im):
    """"""
    lat, long = _inverse_mercator_proj(pos, lat_mp)
    time = start_time + timedelta(hours=t)
    try:
        data = wm.get_weather((time, lat, long))
    except OutsideGridException:
        return 1000  # maybe a better default?
    data["HDT"] = hdt
    if im:
        data = {key: [val] for key, val in data.items()}
        bsp = im.add_influence(pd, data)[0]
    else:
        ugrid, vgrid = data["UGRID"], data["VGRID"]
        tws, twa = _uvgrid_to_tw(ugrid, vgrid, hdt)
        bsp = pd(tws, twa)

    if bsp != 0:
        return 1 / bsp

    return 0


def _right_handing_course(a, b):
    """Calculates course between two points on the surface of the earth
    relative to true north.
    """
    numerator = np.cos(a[1]) * np.sin(b[1]) - np.cos(a[0] - b[0]) * np.cos(
        b[1]
    ) * np.sin(a[1])
    denominator = np.cos(a[0] - b[0]) * np.cos(a[1]) * np.cos(b[1]) + np.sin(
        a[1]
    ) * np.cos(b[1])

    return np.arccos(numerator / np.sqrt(1 - denominator**2))


def _wind_relative_to_north(wind, wind_fmt):
    """Calculates the wind speed and the wind direction relative to true north.

    Parameters
    ----------
    wind: tuple
        Description of the wind. The exact interpretation depends on
        `wind_fmt`. See the description of `wind_fmt` for details.

    wind_fmt: {"ws_wan", "ws_wa_hdt", "uv_grd"}
        Specification how to interpret the parameter `wind`.

        - "ws_wan": `wind` is interpreted as
            (true wind speed, wind angle relative to north)
        - "ws_wa_hdt": `wind` is interpreted as
            (true wind speed, true wind angle,
            heading of the boat relative to north)
        - "uv_grd": `wind` is interpreted as (u_grd, v_grd) as can be read from
            a GRIB file.

    Returns
    -------
    ws : float,
        The current wind speed.

    ndir : float between 0 and 360

    Raises
    --------
    AttributeError
        If `wind_fmt` is not a supported string.
    """
    if wind_fmt == "ws_wan":
        return wind

    if wind_fmt == "ws_wa_hdt":
        ws, wa, hdt = wind
        return ws, (hdt - wa) % 360

    if wind_fmt == "uv_grd":
        wind = np.array(wind)
        u, v = wind
        return np.linalg.norm(wind), 180 / np.pi * np.arctan2(v, u)

    raise AttributeError(f"wind_fmt '{wind_fmt}' is not supported")

    # grib data:
    # wdir = 180 / np.pi * np.arctan2(vgrd, ugrd) + 180

    # twa + bd:
    # wdir = (rwSK + twa) % 360 ?


def _uvgrid_to_tw(ugrid, vgrid, hdt):
    """Calculates the true wind speed and wind angle from given grib data."""
    tws = np.sqrt(ugrid**2 + vgrid**2)
    wa = (180 + 180 / np.pi * np.arctan2(vgrid, ugrid)) % 360
    twa = (hdt - wa) % 360
    return tws, twa


EARTH_FLATTENING = 1 / 298.257223563
EQUATOR_CIRCUMFERENCE = 40075.017


def _great_earth_ellipsoid_distance(a, b):
    """Calculates the distance on the surface for two points on the
    earth surface.
    """
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
