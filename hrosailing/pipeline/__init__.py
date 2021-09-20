"""
Pipeline to create PPDs from raw data
"""

# Author: Valentin Dannenberg


import logging.handlers
from abc import ABC, abstractmethod
from typing import Optional
import warnings

import numpy as np

import hrosailing.polardiagram as pol
from hrosailing.pipelinecomponents.modelfunctions import (
    ws_s_s_dt_wa_gauss_comb,
)
import hrosailing.pipelinecomponents as pc
from hrosailing.wind import _set_resolution


logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.handlers.TimedRotatingFileHandler(
            "hrosailing/logging/pipeline.log", when="midnight"
        )
    ],
)

logger = logging.getLogger(__name__)


class PipelineException(Exception):
    """"""


class PipelineExtension(ABC):
    """"""

    @abstractmethod
    def process(self, w_pts: pc.WeightedPoints):
        """"""


class PolarPipeline:
    """A Pipeline class to create polar diagrams from raw data

    Parameters
    ----------
    extension: PipelineExtension

    handler : DataHandler

    weigher : Weigher, optional

    filter_ : Filter, optional
    """

    def __init__(
        self,
        extension: PipelineExtension,
        handler: pc.DataHandler,
        weigher: pc.Weigher = pc.CylindricMeanWeigher(),
        filter_: pc.Filter = pc.QuantileFilter(),
        im: Optional[pc.InfluenceModel] = None,
    ):
        self.handler = handler
        self.im = im
        self.weigher = weigher
        self.filter = filter_
        self.extension = extension

    def __call__(
        self,
        data,
        check_finite: bool = True,
        tw: bool = True,
        filtering: bool = True,
    ) -> pol.PolarDiagram:
        """
        Parameters
        ----------
        data : FooBar

        check_finite : bool, optional

        tw : bool, optional

        filtering : bool, optional

        Returns
        -------
        out : PolarDiagram
        """
        data = self.handler.handle(data)

        if self.im is not None:
            data = self.im.remove_influence(data)
        else:
            ws = data.get("Wind speed")
            wa = data.get("Wind angle")
            bsp = (
                data.get("Speed over ground knots")
                or data.get("Water speed knots")
                or data.get("Boat speed")
            )

            data = np.column_stack((ws, wa, bsp))

        # NaN and infinite values can't normally be handled
        if check_finite:
            data = np.asarray_chkfinite(data, float)
        else:
            data = np.asarray(data, float)

        # Non-array_like `data` is not allowed after handling
        if data.dtype is object:
            raise PipelineException("`data` is not array_like")

        # `data` should have 3 columns corresponding to wind speed,
        # wind angle and boat speed, after handling
        if data.shape[1] != 3:
            raise PipelineException("`data` has incorrect shape")

        w_pts = pc.WeightedPoints(data, weigher=self.weigher, tw=tw)

        if filtering:
            mask = self.filter.filter(w_pts.weights)
            w_pts = w_pts[mask]

        return self.extension.process(w_pts)


class TableExtension(PipelineExtension):
    """"""

    def __init__(
        self,
        w_res=None,
        neighbourhood: pc.Neighbourhood = pc.Ball(),
        interpolator: pc.Interpolator = pc.ArithmeticMeanInterpolator(1),
    ):
        self.w_res = w_res
        self.neighbourhood = neighbourhood
        self.interpolator = interpolator

    def process(self, w_pts: pc.WeightedPoints) -> pol.PolarDiagramTable:
        """"""
        ws_res, wa_res = _set_wind_resolution(self.w_res, w_pts.points)
        ws, wa = np.meshgrid(ws_res, wa_res)

        i_points = np.column_stack((ws.ravel(), wa.ravel()))
        bsp = _interpolate_points(
            i_points, w_pts, self.neighbourhood, self.interpolator
        )

        bsp = np.asarray(bsp)[:, 2].reshape(len(wa_res), len(ws_res))

        return pol.PolarDiagramTable(ws_res=ws_res, wa_res=wa_res, bsps=bsp)


class CurveExtension(PipelineExtension):
    """"""

    def __init__(
        self,
        regressor: pc.Regressor = pc.ODRegressor(
            model_func=ws_s_s_dt_wa_gauss_comb,
            init_values=(0.25, 10, 1.7, 0, 1.9, 30, 17.6, 0, 1.9, 30, 17.6, 0),
        ),
        radians: bool = False,
    ):
        self.regressor = regressor
        self.radians = radians

    def process(self, w_pts: pc.WeightedPoints) -> pol.PolarDiagramCurve:
        """"""
        if self.radians:
            w_pts.points[:, 1] = np.deg2rad(w_pts.points[:, 1])

        self.regressor.fit(w_pts.points)

        return pol.PolarDiagramCurve(
            self.regressor.model_func,
            *self.regressor.optimal_params,
            radians=self.radians,
        )


class PointcloudExtension(PipelineExtension):
    """"""

    def __init__(
        self,
        sampler: pc.Sampler = pc.UniformRandomSampler(2000),
        neighbourhood: pc.Neighbourhood = pc.Ball(),
        interpolator: pc.Interpolator = pc.ArithmeticMeanInterpolator(50),
    ):
        self.sampler = sampler
        self.neighbourhood = neighbourhood
        self.interpolator = interpolator

    def process(self, w_pts: pc.WeightedPoints) -> pol.PolarDiagramPointcloud:
        """"""
        sample_pts = self.sampler.sample(w_pts.points)
        pts = _interpolate_points(
            sample_pts, w_pts, self.neighbourhood, self.interpolator
        )

        return pol.PolarDiagramPointcloud(pts=pts)


def _set_wind_resolution(w_res, pts):
    if w_res == "auto":
        ws_res = _extract_wind(pts[:, 0], 2, 100)
        wa_res = _extract_wind(pts[:, 1], 5, 30)
        return ws_res, wa_res

    if w_res is None:
        w_res = (None, None)

    ws_res, wa_res = w_res
    return _set_resolution(ws_res, "speed"), _set_resolution(wa_res, "angle")


def _extract_wind(pts, n, threshhold):
    w_max = round(pts.max())
    w_min = round(pts.min())
    w_start = (w_min // n + 1) * n
    w_end = (w_max // n) * n
    res = [w_max, w_min]

    for w in range(w_start, w_end + n, n):
        if w == w_start:
            mask = np.logical_and(w >= pts, pts >= w_min)
        elif w == w_end:
            mask = np.logical_and(w_max >= pts, pts >= w)
        else:
            mask = np.logical_and(w >= pts, pts >= w - n)

        if len(pts[mask]) >= threshhold:
            res.append(w)

    return res


class InterpolationWarning(Warning):
    pass


def _interpolate_points(i_points, w_pts, neighbourhood, interpolator):
    pts = []

    _warning_flag = True

    logger.info(f"Beginning to interpolate `w_res` with {interpolator}")

    for i_pt in i_points:
        mask = neighbourhood.is_contained_in(w_pts.points[:, :2] - i_pt)

        if not np.any(mask):
            if _warning_flag:
                warnings.warn(
                    "Neighbourhood possibly to 'small', or"
                    "chosen resolution not fitting for data. "
                    "Interpolation will not lead to complete results",
                    category=InterpolationWarning,
                )
                _warning_flag = False
            pts.append(0)

        interpol = interpolator.interpolate(w_pts[mask], i_pt)
        interpol_pt = np.array(list(i_pt) + [interpol])
        pts.append(interpol_pt)

    return pts
