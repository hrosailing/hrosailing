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
from hrosailing.polardiagram import PolarDiagramException
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


# TODO Move finiteness and other in common error checks
#      from components to pipeline?

class PolarPipeline:
    """A Pipeline class to create polar diagrams from raw data
    Parameters
    ----------
    extension: PipelineExtension
    weigher : Weigher, optional
    filter_ : Filter, optional
    Methods
    -------
    weigher
        Returns a read only version of self._weigher
    filter
        Returns a read only version of self._filter
    sampler
        Returns a read only version of self._sampler
    interpolater
        Returns a read only version of self._interpolater
    regressor
        Returns a read only version of self._regressor
    __call__(p_type: PolarDiagram,
             data=None, data_file=None,
             file_format=None, file_mode='mean',
             tw=True, filtering=True, w_res=None,
             neighbourhood=None)
    """

    def __init__(
        self,
        extension: PipelineExtension,
        handler: pc.DataHandler,
        weigher: pc.Weigher = None,
        filter_: pc.Filter = None,
    ):
        if weigher is None:
            weigher = pc.CylindricMeanWeigher()
        if not isinstance(weigher, pc.Weigher):
            raise PipelineException(f"{weigher.__name__} is not a Weigher")
        self.weigher = weigher

        if filter_ is None:
            filter_ = pc.QuantileFilter()
        if not isinstance(filter_, pc.Filter):
            raise PipelineException(f"{filter_.__name__} is not a Filter")
        self.filter = filter_

        if not isinstance(extension, PipelineExtension):
            raise PipelineException(
                f"{extension.__name__} is not a PipelineExtension"
            )
        self.extension = extension

        if not isinstance(handler, pc.DataHandler):
            raise PipelineException(f"{handler.__name__} is not a DataHandler")
        self.handler = handler

    def __repr__(self):
        pass

    def __call__(self, data, tw=True, filtering=True):
        """
        Parameters
        ----------
        data :

        tw : bool, optional

        filtering : bool, optional

        Returns
        -------
        out : PolarDiagram
            An instance of the given p_type based on the input data
        """
        data = self.handler.handle(data)
        w_pts = pc.WeightedPoints(data, weigher=self.weigher, tw=tw)
        if filtering:
            mask = self.filter.filter(w_pts.weights)
            w_pts = w_pts[mask]

        self.extension.process(w_pts)


class TableExtension(PipelineExtension):
    """"""

    def __init__(
        self,
        w_res=None,
        neighbourhood: pc.Neighbourhood = None,
        interpolator: pc.Interpolator = None,
    ):
        self.w_res = w_res

        if neighbourhood is None:
            neighbourhood = pc.Ball()
        if not isinstance(neighbourhood, pc.Neighbourhood):
            raise PipelineException(
                f"{neighbourhood.__name__} is not a Neighbourhood"
            )
        self.neighbourhood = neighbourhood

        if interpolator is None:
            interpolator = pc.ArithmeticMeanInterpolator(1, 1)
        if not isinstance(interpolator, pc.Interpolator):
            raise PipelineException(
                f"{interpolator.__name__} is not an Interpolator"
            )
        self.interpolator = interpolator

    def process(self, w_pts: pc.WeightedPoints) -> pol.PolarDiagramTable:
        """"""
        w_res = _set_wind_resolution(self.w_res, w_pts.points)
        bsp = _interpolate_grid_points(
            w_res, w_pts, self.neighbourhood, self.interpolator
        )

        try:
            return pol.PolarDiagramTable(
                ws_res=w_res[0], wa_res=w_res[1], bsps=bsp
            )
        except PolarDiagramException as pe:
            raise PipelineException(
                f"During creation of the polar diagram, an error occured"
            ) from pe


# TODO Add options for radians
class CurveExtension(PipelineExtension):
    """"""

    def __init__(self, regressor: pc.Regressor = None):
        if regressor is None:
            regressor = pc.ODRegressor(
                model_func=mf.tws_s_s_dt_twa_gauss_comb,
                init_values=(0.25, 10, 1.7, 0, 1.9, 30, 17.6, 0),
            )
        if not isinstance(regressor, pc.Regressor):
            raise PipelineException(f"{regressor.__name__} is not a Regressor")
        self.regressor = regressor

    def process(self, w_pts: pc.WeightedPoints) -> pol.PolarDiagramCurve:
        """"""
        self.regressor.fit(w_pts.points)

        try:
            return pol.PolarDiagramCurve(
                self.regressor.model_func, *self.regressor.optimal_params
            )
        except PolarDiagramException as pe:
            raise PipelineException(
                f"During creation of the polar diagram, an error occured"
            ) from pe


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
        if not isinstance(sampler, pc.Sampler):
            raise PipelineException(f"{sampler.__name__} is not a Sampler")
        self.sampler = sampler

        if neighbourhood is None:
            neighbourhood = pc.Ball()
        if not isinstance(neighbourhood, pc.Neighbourhood):
            raise PipelineException(
                f"{neighbourhood.__name__} is not a Neighbourhood"
            )
        self.neighbourhood = neighbourhood

        if interpolator is None:
            interpolator = pc.ArithmeticMeanInterpolator(1, 1)
        if not isinstance(interpolator, pc.Interpolator):
            raise PipelineException(
                f"{interpolator.__name__} is not an Interpolator"
            )
        self.interpolator = interpolator

    def process(self, w_pts: pc.WeightedPoints) -> pol.PolarDiagramPointcloud:
        """"""
        sample_pts = self.sampler.sample(w_pts.points)
        pts = []
        logger.info(
            f"Beginning to interpolate sample_pts with "
            f"{self.interpolator.__name__}"
        )
        for s_pt in sample_pts:
            mask = self.neighbourhood.is_contained_in(
                w_pts.points[:, :2] - s_pt
            )
            if not np.any(mask):
                raise PipelineException(
                    f"No points where contained in the neighbourhood of "
                    f"{s_pt}. Interpolation not possible"
                )
            pts.append(self.interpolator.interpolate(w_pts[mask], s_pt))

        try:
            return pol.PolarDiagramPointcloud(pts=pts)
        except PolarDiagramException as pe:
            raise PipelineException(
                f"During creation of the polar diagram, an error occured"
            ) from pe


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
            mask = pts >= w_min & pts <= w
        elif w == w_end:
            mask = pts >= w & pts <= w_max
        else:
            mask = pts >= w - n & pts <= w

        if len(pts[mask]) >= threshhold:
            res.append(w)

    return res


def _interpolate_grid_points(w_res, w_pts, nhood, ipol):
    ws_res, wa_res = w_res
    bsp = np.zeros((len(wa_res), len(ws_res)))

    logger.info(f"Beginning to interpolate w_res with {ipol.__name__}")
    for i, ws in enumerate(ws_res):
        for j, wa in enumerate(wa_res):
            grid_point = np.array([ws, wa])
            mask = nhood.is_contained_in(w_pts.points[:, :2] - grid_point)
            if not np.any(mask):
                raise PipelineException(
                    f"No points were contained in the neighbourhood of "
                    f"{grid_point}. Interpolation not possible"
                )
            bsp[j, i] = ipol.interpolate(w_pts[mask], grid_point)

    return bsp
