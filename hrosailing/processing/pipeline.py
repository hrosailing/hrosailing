"""
A Pipeline class to automate getting a
polar diagram from "raw" data
"""

# Author: Valentin F. Dannenberg / Ente


import logging.handlers
from abc import ABC, abstractmethod

import numpy as np

import hrosailing.polardiagram as pol
import hrosailing.processing.modelfunctions as mf
import hrosailing.processing.pipelinecomponents as pc
from hrosailing.wind import set_resolution

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    filename="hrosailing/logging/processing.log",
)

LOG_FILE = "hrosailing/logging/processing.log"

logger = logging.getLogger(__name__)
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when="midnight"
)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


class PipelineException(Exception):
    """"""

    pass


class PipelineExtension(ABC):
    """"""

    @abstractmethod
    def process(self, w_pts: pc.WeightedPoints):
        pass


class PolarPipeline:
    """A Pipeline class to create polar diagrams from raw data

    Parameters
    ----------
    extension: PipelineExtension

    handler : DataHandler

    weigher : Weigher, optional

    filter_ : Filter, optional


    Raises a PipelineException
    """

    def __init__(
        self,
        extension: PipelineExtension,
        handler: pc.DataHandler,
        weigher: pc.Weigher = None,
        filter_: pc.Filter = None,
        im: pc.InfluenceModel = None,
    ):
        if im is not None and not isinstance(im, pc.InfluenceModel):
            raise PipelineException("`im` is not an InfluenceModel")
        if weigher is None:
            weigher = pc.CylindricMeanWeigher()
        elif not isinstance(weigher, pc.Weigher):
            raise PipelineException("`weigher` is not a Weigher")

        if filter_ is None:
            filter_ = pc.QuantileFilter()
        elif not isinstance(filter_, pc.Filter):
            raise PipelineException("`filter_` is not a Filter")

        if not isinstance(extension, PipelineExtension):
            raise PipelineException("`extension` is not a PipelineExtension")

        if not isinstance(handler, pc.DataHandler):
            raise PipelineException("`handler` is not a DataHandler")

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

        # NaN and infinite values can't normally be handled
        if check_finite:
            try:
                data = np.asarray_chkfinite(data, float)
            except ValueError as ve:
                raise PipelineException(
                    "`data` contains infinite or NaN values"
                ) from ve
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
        neighbourhood: pc.Neighbourhood = None,
        interpolator: pc.Interpolator = None,
    ):
        if neighbourhood is None:
            neighbourhood = pc.Ball()
        elif not isinstance(neighbourhood, pc.Neighbourhood):
            raise PipelineException("`neighbourhood` is not a Neighbourhood")

        if interpolator is None:
            interpolator = pc.ArithmeticMeanInterpolator(1)
        elif not isinstance(interpolator, pc.Interpolator):
            raise PipelineException("`interpolator` is not an Interpolator")

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

        bsp = np.asarray(bsp).reshape(len(wa_res), len(ws_res))

        return pol.PolarDiagramTable(ws_res=ws_res, wa_res=wa_res, bsps=bsp)


class CurveExtension(PipelineExtension):
    """"""

    def __init__(self, regressor: pc.Regressor = None, radians=False):
        if regressor is None:
            # initial values chosen by prior work of Stefan Simon
            regressor = pc.ODRegressor(
                model_func=mf.tws_s_s_dt_twa_gauss_comb,
                init_values=(0.25, 10, 1.7, 0, 1.9, 30, 17.6, 0),
            )
        elif not isinstance(regressor, pc.Regressor):
            raise PipelineException("`regressor` is not a Regressor")

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
        sampler: pc.Sampler = None,
        neighbourhood: pc.Neighbourhood = None,
        interpolator: pc.Interpolator = None,
    ):
        if sampler is None:
            sampler = pc.UniformRandomSampler(500)
        elif not isinstance(sampler, pc.Sampler):
            raise PipelineException("`sampler` is not a Sampler")

        if neighbourhood is None:
            neighbourhood = pc.Ball()
        elif not isinstance(neighbourhood, pc.Neighbourhood):
            raise PipelineException("`neighbourhood` is not a Neighbourhood")

        if interpolator is None:
            interpolator = pc.ArithmeticMeanInterpolator(1)
        elif not isinstance(interpolator, pc.Interpolator):
            raise PipelineException("`interpolator` is not an Interpolator")

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
    return set_resolution(ws_res, "speed"), set_resolution(wa_res, "angle")


# TODO Better approach?
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


def _interpolate_points(i_points, w_pts, neighbourhood, interpolator):
    pts = []

    logger.info(f"Beginning to interpolate `w_res` with {interpolator}")

    for i_pt in i_points:
        mask = neighbourhood.is_contained_in(w_pts.points[:, :2] - i_pt)
        if not np.any(mask):
            raise PipelineException(
                f"No points were contained in the neighbourhood of "
                f"{i_pt}. Interpolation not possible"
            )
        interpol = interpolator.interpolate(w_pts[mask], i_pt)
        interpol_pt = np.array(list(i_pt) + [interpol])
        pts.append(interpol_pt)

    return pts
