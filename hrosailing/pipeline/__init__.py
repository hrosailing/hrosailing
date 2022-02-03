"""
Pipeline to create PPDs from raw data
"""


import warnings
from abc import ABC, abstractmethod

import numpy as np

import hrosailing.pipelinecomponents as pc
import hrosailing.polardiagram as pol
from hrosailing.pipelinecomponents.modelfunctions import \
    ws_s_wa_gauss_and_square
from hrosailing.polardiagram._polardiagramtable import _set_resolution


class PolarPipeline:
    """A Pipeline class to create polar diagrams from raw data

    Parameters
    ----------
    extension: PipelineExtension
        Extension that is called in the pipeline, after all preprocessing
        is done, to generate a polar diagram from the processed data.

        Determines the subclass of PolarDiagram, that the pipeline will
        produce

    handler : DataHandler
        Handler that is responsible to extract actual data from the input

        Determines the type and format of input the pipeline should accept

    weigher : Weigher, optional
        Determines the method with which the points will be weight.

        Defaults to CylindricMeanWeigher()

    filter_ : Filter, optional
        Determines the methods with which the points will be filtered,
        if `filtering` is `True` in __call__ method

        Defaults to QuantileFilter()
    """

    def __init__(
        self,
        extension,
        handler,
        weigher=pc.CylindricMeanWeigher(),
        filter_=pc.QuantileFilter(),
        influence_model=None,
    ):
        self.handler = handler
        self.influence_model = influence_model
        self.weigher = weigher
        self.filter = filter_
        self.extension = extension

    def __call__(
        self,
        data,
        apparent_wind=False,
        filtering=True,
        n_zeros=500,
        _enable_logging=False,
    ) -> pol.PolarDiagram:
        """
        Parameters
        ----------
        data : compatible with `self.handler`
            Data from which to create the polar diagram

            The input should be compatible with the DataHandler instance
            given in initialization of the pipeline instance

        apparent_wind : bool, optional
            Specifies if wind data is given in apparent wind

            If `True`, wind will be converted to true wind

            Defaults to `False`

        filtering : bool, optional
            Specifies, if points should be filtered after weighing

            Defaults to `True`

        n_zeros: positive int, optional
            Specifies the number of additional data points at `(tws, 0)` and
            `(tws, 360)` respectively, which are appended to the filtered data

            This is done to better simulate the behaviour of sailing vessels at
            wind angle 0 / 360.

            Defaults to `500`

        Returns
        -------
        pd : PolarDiagram
            `PolarDiagram` subclass instance, which represents the
            trends in `data`

            Type depends on the chosen `PipelineExtension` subclass
        """
        data = self.handler.handle(data)

        if self._has_influence_model():
            data = self.influence_model.remove_influence(data)

        weighted_points = pc.WeightedPoints(
            data,
            weigher=self.weigher,
            apparent_wind=apparent_wind,
            _enable_logging=_enable_logging,
        )

        if filtering:
            self._filter_data(weighted_points)

        weighted_points = _add_zeros(weighted_points, n_zeros)

        return self.extension.process(weighted_points)

    def _has_influence_model(self):
        return self.influence_model is not None

    def _filter_data(self, weighted_points):
        points_to_filter = self.filter.filter(weighted_points.weights)
        weighted_points = weighted_points[points_to_filter]


def _add_zeros(weighted_points, n_zeros):
    ws = weighted_points.points[:, 0]
    ws = np.linspace(min(ws), max(ws), n_zeros)

    zero = np.zeros(n_zeros)
    full = 360 * np.ones(n_zeros)
    zeros = np.column_stack((ws, zero, zero))
    fulls = np.column_stack((ws, full, zero))

    original_points = weighted_points.points
    original_weights = weighted_points.weights

    return pc.WeightedPoints(
        data=np.concatenate([original_points, zeros, fulls]),
        weights=np.concatenate([original_weights, np.ones(2 * n_zeros)]),
    )


class PipelineExtension(ABC):
    """Base class for all pipeline extensions

    Abstract Methods
    ----------------
    process(weighted_points)
    """

    @abstractmethod
    def process(self, weighted_points) -> pol.PolarDiagram:
        """This method, given an instance of WeightedPoints, should
        return a polar diagram object, which represents the trends
        and data contained in the WeightedPoints instance
        """


class TableExtension(PipelineExtension):
    """Pipeline extension to produce PolarDiagramTable instances
    from preprocessed data

    Parameters
    ----------
    wind_resolution : tuple of two array_likes or scalars, or str, optional
        Wind speed and angle resolution to be used in the final table
        Can be given as

        - a tuple of two `array_likes` with scalar entries, that
        will be used as the resolution
        - a tuple of two `scalars`, which will be used as
        stepsizes for the resolutions
        - the str `"auto"`, which will result in a resolution, that is
        somewhat fitted to the data

    neighbourhood : Neighbourhood, optional
        Determines the neighbourhood around a point from which to draw
        the data points used in the interpolation of that point

        Defaults to `Ball(radius=1)`

    interpolator : Interpolator, optional
        Determines which interpolation method is used

        Defaults to `ArithmeticMeanInterpolator(50)`
    """

    def __init__(
        self,
        wind_resolution=None,
        neighbourhood=pc.Ball(radius=1),
        interpolator=pc.ArithmeticMeanInterpolator(50),
    ):
        self.wind_resolution = wind_resolution
        self.neighbourhood = neighbourhood
        self.interpolator = interpolator

    def process(self, weighted_points) -> pol.PolarDiagramTable:
        """Creates a PolarDiagramTable instance from preprocessed data,
        by first determining a wind speed / wind angle grid, using
        `self.w_res`, and then interpolating the boat speed values at the
        grid points according to the interpolation method of
        `self.interpolator`, which only takes in consideration the data points
        which lie in a neighbourhood, determined by `self.neighbourhood`,
        around each grid point

        Parameters
        ----------
        weighted_points : WeightedPoints
            Preprocessed data from which to create the polar diagram

        Returns
        -------
        polar_diagram : PolarDiagramTable
            A polar diagram that should represent the trends captured
            in the raw data
        """
        ws_resolution, wa_resolution = self._determine_table_size(
            weighted_points.points
        )
        ws, wa = np.meshgrid(ws_resolution, wa_resolution)
        grid_points = np.column_stack((ws.ravel(), wa.ravel()))

        interpolated_points = _interpolate_points(
            grid_points, weighted_points, self.neighbourhood, self.interpolator
        )
        bsps = _extract_boat_speed(
            interpolated_points, len(wa_resolution), len(ws_resolution)
        )

        return pol.PolarDiagramTable(ws_resolution, wa_resolution, bsps)

    def _determine_table_size(self, points):
        if self.wind_resolution == "auto":
            return _automatically_determined_resolution(points)

        if self.wind_resolution is None:
            self.wind_resolution = (None, None)

        ws_resolution, wa_resolution = self.wind_resolution
        return _set_resolution(ws_resolution, "s"), _set_resolution(
            wa_resolution, "a"
        )


def _automatically_determined_resolution(points):
    ws_resolution = _extract_wind(points[:, 0], 2, 100)
    wa_resolution = _extract_wind(points[:, 1], 5, 30)

    return ws_resolution, wa_resolution


def _extract_wind(points, n, threshhold):
    w_max = round(points.max())
    w_min = round(points.min())
    w_start = (w_min // n + 1) * n
    w_end = (w_max // n) * n
    res = [w_max, w_min]

    for w in range(w_start, w_end + n, n):
        if w == w_start:
            mask = np.logical_and(w >= points, points >= w_min)
        elif w == w_end:
            mask = np.logical_and(w_max >= points, points >= w)
        else:
            mask = np.logical_and(w >= points, points >= w - n)

        if len(points[mask]) >= threshhold:
            res.append(w)

    return res


def _extract_boat_speed(interpolated_points, rows, cols):
    bsps = interpolated_points[:, 2]
    return bsps.reshape(rows, cols)


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
            model_func=ws_s_wa_gauss_and_square,
            init_values=(0.2, 0.2, 10, 0.001, 0.3, 110, 2000, 0.3, 250, 2000)
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
        regressor=pc.ODRegressor(
            model_func=ws_s_wa_gauss_and_square,
            init_values=(0.2, 0.2, 10, 0.001, 0.3, 110, 2000, 0.3, 250, 2000),
        ),
        radians=False,
    ):
        self.regressor = regressor
        self.radians = radians

    def process(self, weighted_points) -> pol.PolarDiagramCurve:
        """Creates a PolarDiagramCurve instance from preprocessed data,
        by fitting a given function to said data, using a regression
        method determined by `self.regressor`

        Parameters
        ----------
        weighted_points : WeightedPoints
            Preprocessed data from which to create the polar diagram

        Returns
        -------
        pd : PolarDiagramCurve
            A polar diagram that should represent the trends captured
            in the raw data
        """
        if self._use_radians():
            _convert_angles_to_radians(weighted_points)

        self.regressor.fit(weighted_points.points)

        return pol.PolarDiagramCurve(
            self.regressor.model_func,
            *self.regressor.optimal_params,
            radians=self.radians,
        )

    def _use_radians(self):
        return self.radians


def _convert_angles_to_radians(weighted_points):
    weighted_points.points[:, 1] = np.deg2rad(weighted_points.points[:, 1])


class PointcloudExtension(PipelineExtension):
    """Pipeline extension to produce PolarDiagramPointcloud instances
    from preprocessed data

    Parameters
    ----------
    sampler : Sampler, optional
        Determines the number of points in the resulting point cloud
        and the method used to sample the preprocessed data and represent
        the trends captured in them

        Defaults to `UniformRandomSampler(2000)`

    neighbourhood : Neighbourhood, optional
        Determines the neighbourhood around a point from which to draw
        the data points used in the interpolation of that point

        Defaults to `Ball(radius=1)`

    interpolator : Interpolator, optional
        Determines which interpolation method is used

        Defaults to `ArithmeticMeanInterpolator(50)`
    """

    def __init__(
        self,
        sampler=pc.UniformRandomSampler(2000),
        neighbourhood=pc.Ball(radius=1),
        interpolator=pc.ArithmeticMeanInterpolator(50),
    ):
        self.sampler = sampler
        self.neighbourhood = neighbourhood
        self.interpolator = interpolator

    def process(self, weighted_points) -> pol.PolarDiagramPointcloud:
        """Creates a PolarDiagramPointcloud instance from preprocessed data,
        first creating a set number of points by sampling the wind speed,
        wind angle space of the data points and capturing the underlying
        trends using `self.sampler` and then interpolating the boat speed
        values at the sampled points according to the interpolation method of
        `self.interpolator`, which only takes in consideration the data points
        which lie in a neighbourhood, determined by `self.neighbourhood`,
        around each sampled point

        Parameters
        ----------
        weighted_points : WeightedPoints
            Preprocessed data from which to create the polar diagram

        Returns
        -------
        pd : PolarDiagramPointcloud
            A polar diagram that should represent the trends captured
            in the raw data
        """
        sample_points = self.sampler.sample(weighted_points.points)
        interpolated_points = _interpolate_points(
            sample_points,
            weighted_points,
            self.neighbourhood,
            self.interpolator,
        )

        return pol.PolarDiagramPointcloud(interpolated_points)


class InterpolationWarning(Warning):
    """Warning raised if neighbourhood is too small for
    successful interpolation
    """


def _interpolate_points(
    interpolating_points, weighted_points, neighbourhood, interpolator
):
    interpolated_points = [
        _interpolate_point(point, weighted_points, neighbourhood, interpolator)
        for point in interpolating_points
    ]

    return np.array(interpolated_points)


def _interpolate_point(point, weighted_points, neighbourhood, interpolator):
    considered = neighbourhood.is_contained_in(
        weighted_points.points[:, :2] - point
    )

    if _neighbourhood_too_small(considered):
        warnings.warn(
            "Neighbourhood possibly to `small`, or"
            "chosen resolution not fitting for data. "
            "Interpolation will not lead to complete results",
            category=InterpolationWarning,
        )
        return np.concatenate([point, 0])

    interpolated_value = interpolator.interpolate(
        weighted_points[considered], point
    )

    return np.concatenate([point, interpolated_value])


def _neighbourhood_too_small(considered_points):
    return not np.any(considered_points)
