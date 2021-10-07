"""
Pipeline to create PPDs from raw data
"""

# Author: Valentin Dannenberg


import logging.handlers
import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

import hrosailing._logfolder as log
import hrosailing.pipelinecomponents as pc
import hrosailing.polardiagram as pol
from hrosailing.pipelinecomponents.modelfunctions import \
    ws_s_wa_gauss_and_square
from hrosailing.wind import _set_resolution

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.handlers.TimedRotatingFileHandler(
            log.log_folder + "/pipeline.log", when="midnight"
        )
    ],
)
logger = logging.getLogger(__name__)
del log


class PipelineException(Exception):
    """Exception raised if an error occurs in the pipeline"""


class PipelineExtension(ABC):
    """Base class for all pipeline extensions

    Abstract Methods
    ----------------
    process(w_pts)
    """

    @abstractmethod
    def process(self, w_pts: pc.WeightedPoints) -> pol.PolarDiagram:
        """This method, given an instance of WeightedPoints, should
        return a polar diagram object, which represents the trends
        and data contained in the WeightedPoints instance
        """


class PolarPipeline:
    """A Pipeline class to create polar diagrams from raw data

    Parameters
    ----------
    extension: PipelineExtension
        Extension that is called in the pipeline, after all preprocessing
        is done, to generate a polar diagram from the processed data.

        Determines the subclass of PolarDiagram, that the pipeline will
        produce

        Note: Any class with a method `process(w_pts: WeightedPoints)` should
        work here, but we recommend using a subclass of PipelineExtension

    handler : DataHandler
        Handler that is responsible to extract actual data from the input

        Determines the type and format of input the pipeline should accept

        Note: Any class with a method `handle(data) -> dict` should work here,
        but we recommend using a subclass of DataHandler

    weigher : Weigher, optional
        Determines the method with which the points will be weight.

        Defaults to CylindricMeanWeigher()

        Note: Any class with a method `weigh(pts) -> (n,) array_like` should
        work here, but we recommend using a subclass of Weigher

    filter_ : Filter, optional
        Determines the methods with which the points will be filtered,
        if `filtering` is `True` in __call__ method

        Defaults to QuantileFilter()

        Note: Any class with a method `filter(wts) -> (n,) boolean array`
        should work here, but we recommend using a subclass of Filter
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
        tw: bool = True,
        filtering: bool = True,
        n_zeros: int = 500,
    ) -> pol.PolarDiagram:
        """
        Parameters
        ----------
        data : compatible with `self.handler`
            Data from which to create the polar diagram.
            The input should be compatible with the DataHandler instance
            given in initialization of the pipeline instance

        tw : bool, optional
            Specifies, if the wind components of the points are true
            or appearent wind

            If false, wind will be converted to true wind

            Defautls to `True`

        filtering : bool, optional
            Specfies, if points should be filtered after weighing

            Defaults to `True`

        n_zeros: int, optional
            If not None, describes the number of additional data points
            at (tws, 0) and (tws, 360) respectively, which are appended
            to the filtered data

            Needed to better model the actual behaviour of sailing boats
            when sailing against the wind

            Defaults to 500

        Returns
        -------
        pd : PolarDiagram
            PolarDiagram subclass instance, which represents the
            trends in `data`

            Type depends on the chosen PipelineExtension subclass
        """
        data = self.handler.handle(data)

        if self.im is not None:
            data = self.im.remove_influence(data)
        else:
            ws, wa, bsp = _get_relevant_data(data)
            data = np.column_stack((ws, wa, bsp))

        # NaN and infinite values aren't allowed
        data = np.asarray_chkfinite(data, float)

        # Non-array_like `data` is not allowed after handling
        if data.dtype is object:
            raise PipelineException("`data` is not array_like")

        # `data` should have 3 columns corresponding to wind speed,
        # wind angle and boat speed, after the influence model
        if data.shape[1] != 3:
            raise PipelineException("`data` has incorrect shape")

        w_pts = pc.WeightedPoints(data, weigher=self.weigher, tw=tw)

        if filtering:
            mask = self.filter.filter(w_pts.weights)
            w_pts = w_pts[mask]

        if n_zeros:
            w_pts = _add_zeros(w_pts, n_zeros)

        return self.extension.process(w_pts)


class TableExtension(PipelineExtension):
    """Pipeline extension to produce PolarDiagramTable instances
    from preprocessed data

    Parameters
    ----------
    w_res : , optional

    neighbourhood : Neighbourhood, optional
        Determines the neighbourhood around a point from which to draw
        the data points used in the interpolation of that point

        Defaults to Ball(radius=1)

        Note: Any class with a method `is_contained_in(pts) -> (n,) boolean array`
        should work here, but we recommend using a subclass of Neighbourhood

    interpolator : Interpolator, optional
        Determines which interpolation method is used

        Defaults to ArithmeticMeanInterpolator(50)

        Note: Any class with a method `interpolate(w_pts, grid_pt) -> val`
        should work here, but we recommend using a subclass of Interpolator
    """

    def __init__(
        self,
        w_res=None,
        neighbourhood: pc.Neighbourhood = pc.Ball(radius=1),
        interpolator: pc.Interpolator = pc.ArithmeticMeanInterpolator(50),
    ):
        self.w_res = w_res
        self.neighbourhood = neighbourhood
        self.interpolator = interpolator

    def process(self, w_pts: pc.WeightedPoints) -> pol.PolarDiagramTable:
        """Creates a PolarDiagramTable instance from preprocessed data,
        by first determining a wind speed / wind angle grid, using
        `self.w_res`, and then interpolating the boat speed values at the
        grid points according to the interpolation method of
        `self.interpolator`, which only takes in consideration the data points
        which lie in a neighbourhood, determined by `self.neighbourhood`,
        around each grid point

        Parameters
        ----------
        w_pts : WeightedPoints
            Preprocessed data from which to create the polar diagram

        Returns
        -------
        pd : PolarDiagramTable
            A polar diagram that should represent the trends captured
            in the raw data
        """
        ws_res, wa_res = _set_wind_resolution(self.w_res, w_pts.points)
        ws, wa = np.meshgrid(ws_res, wa_res)

        i_points = np.column_stack((ws.ravel(), wa.ravel()))
        bsp = _interpolate_points(
            i_points, w_pts, self.neighbourhood, self.interpolator
        )

        bsp = np.asarray(bsp)[:, 2].reshape(len(wa_res), len(ws_res))

        return pol.PolarDiagramTable(ws_res=ws_res, wa_res=wa_res, bsps=bsp)


class CurveExtension(PipelineExtension):
    """Pipeline extension to produce PolarDiagramCurve instances
    from preprocessed data

    Parameters
    ----------
    regressor : Regressor, optional
        Determines which regression method and model function is to be used,
        to represent the data.

        The model function will also be passed to PolarDiagramCurve

        Defaults to `ODRegressor(
            model_func=ws_s_s_dt_wa_gauss_comb,
            init_values=(0.25, 10, 1.7, 0, 1.9, 30, 17.6, 0, 1.9, 30, 17.6, 0)
        )`

    radians : bool, optional
        Determines if the model function used to represent the data takes
        the wind angles to be in radians or degrees

        If `True`, will convert the wind angles of the data points to
        radians

        Defaults to `False`
    """

    def __init__(
        self,
        regressor: pc.Regressor = pc.ODRegressor(
            model_func=ws_s_wa_gauss_and_square,
            init_values=(0.2, 0.2, 10, 0.001, 0.3, 110, 2000, 0.3, 250, 2000),
        ),
        radians: bool = False,
    ):
        self.regressor = regressor
        self.radians = radians

    def process(self, w_pts: pc.WeightedPoints) -> pol.PolarDiagramCurve:
        """Creates a PolarDiagramCurve instance from preprocessed data,
        by fitting a given function to said data, using a regression
        method determined by `self.regressor`

        Parameters
        ----------
        w_pts : WeightedPoints
            Preprocessed data from which to create the polar diagram

        Returns
        -------
        pd : PolarDiagramCurve
            A polar diagram that should represent the trends captured
            in the raw data
        """
        if self.radians:
            w_pts.points[:, 1] = np.deg2rad(w_pts.points[:, 1])

        self.regressor.fit(w_pts.points)

        return pol.PolarDiagramCurve(
            self.regressor.model_func,
            *self.regressor.optimal_params,
            radians=self.radians,
        )


class PointcloudExtension(PipelineExtension):
    """Pipeline extension to produce PolarDiagramPointcloud instances
    from preprocessed data

    Parameters
    ----------
    sampler : Sampler, optional
        Determines the number of points in the resulting point cloud
        and the method used to sample the preprocessed data and represent
        the trends captured in them

        Defaults to UniformRandomSampler(2000)

        Note: Any class with a method `sample(pts) -> (n,) array` should work
        here, but we recommend using a subclass of Sampler

    neighbourhood : Neighbourhood, optional
        Determines the neighbourhood around a point from which to draw
        the data points used in the interpolation of that point

        Defaults to Ball(radius=1)

        Note: Any class with a method `is_contained_in(pts) -> (n,) boolean array`
        should work here, but we recommend using a subclass of Neighbourhood

    interpolator : Interpolator, optional
        Determines which interpolation method is used

        Defaults to ArithmeticMeanInterpolator(50)

        Note: Any class with a method `interpolate(w_pts, grid_pt) -> val`
        should work here, but we recommend using a subclass of Interpolator
    """

    def __init__(
        self,
        sampler: pc.Sampler = pc.UniformRandomSampler(2000),
        neighbourhood: pc.Neighbourhood = pc.Ball(radius=1),
        interpolator: pc.Interpolator = pc.ArithmeticMeanInterpolator(50),
    ):
        self.sampler = sampler
        self.neighbourhood = neighbourhood
        self.interpolator = interpolator

    def process(self, w_pts: pc.WeightedPoints) -> pol.PolarDiagramPointcloud:
        """Creates a PolarDiagramPointcloud instance from preprocessed data,
        first creating a set number of points by sampling the wind speed,
        wind angle space of the  data points and capturing the underlying
        trends using `self.sampler` and then interpolating the boat speed
        values at the sampled points according to the interpolation method of
        `self.interpolator`, which only takes in consideration the data points
        which lie in a neighbourhood, determined by `self.neighbourhood`,
        around each sampled point

        Parameters
        ----------
        w_pts : WeightedPoints
            Preprocessed data from which to create the polar diagram

        Returns
        -------
        pd : PolarDiagramPointcloud
            A polar diagram that should represent the trends captured
            in the raw data
        """
        sample_pts = self.sampler.sample(w_pts.points)
        pts = _interpolate_points(
            sample_pts, w_pts, self.neighbourhood, self.interpolator
        )

        return pol.PolarDiagramPointcloud(pts=pts)


WIND_KEYS = {"Wind speed", "Wind Speed", "wind speed"}
ANGLE_KEYS = {"Wind angle", "Wind Angle", "wind angle"}
SPEED_KEYS = {
    "Boat Speed",
    "Boat speed",
    "Speed Over Ground",
    "Speed over ground",
    "Speed over Ground",
    "speed over ground",
    "Speed over ground knots",
    "Water Speed",
    "Water speed",
    "water speed",
    "Water Speed knots",
    "Water speed knots",
}


def _get_relevant_data(data):
    ws = [data.get(ws_key, None) for ws_key in WIND_KEYS]
    wa = [data.get(wa_key, None) for wa_key in ANGLE_KEYS]
    bsp = [data.get(bsp_key, None) for bsp_key in SPEED_KEYS]

    ws = [w for w in ws if w is not None]
    wa = [a for a in wa if a is not None]
    bsp = [b for b in bsp if b is not None]

    # can't use pipeline if some of the data is not present
    if not ws or not wa or not bsp:
        raise PipelineException(
            "Not enough relevant data present for usage of pipeline"
        )

    return ws[0], wa[0], bsp[0]


def _add_zeros(w_pts, n_zeros):
    ws = w_pts.points[:, 0]
    ws = np.linspace(min(ws), max(ws), n_zeros)

    zero = np.zeros(n_zeros)
    full = 360 * np.ones(n_zeros)
    zeros = np.column_stack((ws, zero, zero))
    fulls = np.column_stack((ws, full, zero))

    return pc.WeightedPoints(
        pts=np.concatenate([w_pts.points, zeros, fulls]),
        wts=np.concatenate([w_pts.weights, np.ones(2 * n_zeros)]),
    )


def _set_wind_resolution(w_res, pts):
    if w_res == "auto":
        ws_res = _extract_wind(pts[:, 0], 2, 100)
        wa_res = _extract_wind(pts[:, 1], 5, 30)
        return ws_res, wa_res

    if w_res is None:
        w_res = (None, None)

    ws_res, wa_res = w_res
    return _set_resolution(ws_res, "s"), _set_resolution(wa_res, "a")


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
    """Warning raised if neighbourhood is too small for
    successful interpolation
    """


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
