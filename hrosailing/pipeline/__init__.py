"""
Pipeline to create polar diagrams from raw data.
"""


import warnings
from abc import ABC, abstractmethod
from datetime import datetime as dt
from typing import NamedTuple

import numpy as np

import hrosailing.pipelinecomponents as pc
import hrosailing.polardiagram as pol
from hrosailing.pipelinecomponents.modelfunctions import (
    ws_s_wa_gauss_and_square,
)

from .extensions import (
    CurveExtension,
    PipelineExtension,
    PointcloudExtension,
    TableExtension,
)


class Statistics(NamedTuple):
    """
    Organizes the statistics returned by different `Pipelinecomponents`.
    The attributes correspond to the parameters of `PolarPipeline.__init__`
    and each contains dictionaries with relevant statistics.

    See also
    ----------
    `PolarPipeline`
    """

    data_handler: dict
    imputator: dict
    smoother: dict
    pre_expanding_weigher: dict
    pre_expanding_filter: dict
    expander: dict
    pre_influence_weigher: dict
    pre_influence_filter: dict
    influence_model: dict
    post_weigher: dict
    post_filter: dict
    injector: dict
    extension: dict
    quality_assurance: dict


_EMPTY_STATISTIC = Statistics(
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
)


class PipelineOutput(NamedTuple):
    """
    Organizes the output of a `PolarPipeline` call.

    Attributes
    ----------
    polardiagram: PolarDiagram
        The resulting polar diagram.

    training_statistics: Statistics
        Relevant statistics obtained in the preprocessing and processing of the training data.

    test_statistics: Statistics
        Relevant statistics obtained in the preprocessing of the test data.
        The attributes `extension` and `quality_assurance` only contain an empty dictionary.
    """

    polardiagram: pol.PolarDiagram
    training_statistics: Statistics
    test_statistics: Statistics


class PolarPipeline:
    """A Pipeline class to create polar diagrams from raw data.

    Other Parameters
    ---------------------------
    data_handler : DataHandler or list of DataHandler, optional
        Handlers that are responsible to extract actual data from the input.
        If only one handler is given, this handler will be used for all given inputs,
        otherwise the handlers will be used one after another for each data input including the training data.

        Determines the type and format of input the pipeline should accept.

    imputator : Imputator, optional
        Determines the method which will be used to produce data without
        `None` entries.

        Defaults to `RemoveOnlyImputator()`.

    smoother: Smoother, optional
        Determines the method which will be used to smoothen out the rounding
        following from low measurement precision.

        Defaults to `LazySmoother()`.

    pre_expander_weigher : Weigher, optional
        Determines the method with which the points will be weighted before application of the expander.

        Defaults to `CylindricMeanWeigher()`

    pre_expander_filter : Filter, optional
        Determines the method with which the points will be filtered with before application of the expander.

        Defaults to `QuantileFilter()`

    expander: Expander, optional
        Determines the method which will be used to expand the data by several more data fields.
        For example weather data from a weather model.

    pre_influence_weigher, pre_weigher : Weigher, optional
        Determines the method with which the points will be weighted before
        application of the influence model.

        Defaults to `CylindricMeanWeigher()`.

    pre_influence_filter, pre_filter : Filter, optional
        Determines the methods which the points will be filtered with before application of the influence model.

        Defaults to `QuantileFilter()`.

    influence_model : InfluenceModel, optional
        Determines the influence model which is applied and fitted to the data.

        Defaults to 'IdentityInfluenceModel()'.

    post_weigher, weigher : Weigher, optional
        Determines the method with which the points will be weighted after
        application of the influence model.

        Defaults to `CylindricMeanWeigher()`.

    post_filter, filter : Filter, optional
        Determines the methods with which the points will be filtered
        after the application of the influence model,
        if `post_filtering` in `__call__` method.

        Defaults to `QuantileFilter()`.

    injector : Injector, optional
        Determines the method used to add additional artificial data points to the
        data.

        Defaults to `ZeroInjector(500)`.

    extension: PipelineExtension
        Extension that is called in the pipeline, after all preprocessing
        is done, to generate a polar diagram from the processed data.

        Determines the subclass of `PolarDiagram`, that the pipeline will
        produce.

        Defaults to `TableExtension()`.

    quality_assurance : QualityAssurance, optional
        Determines the method which is used to measure the quality of the
        resulting polar diagram using preprocessed test_data.

        Defaults to `MinimalQualityAssurance()`.
    """

    def __init__(self, **custom_components):
        defaults = [
            pc.NMEAFileHandler(),
            pc.RemoveOnlyImputator(),
            pc.LazySmoother(),
            pc.CylindricMeanWeigher(),
            pc.QuantileFilter(),
            pc.LazyExpander(),
            pc.CylindricMeanWeigher(),
            pc.QuantileFilter(),
            pc.IdentityInfluenceModel(),
            pc.CylindricMeanWeigher(),
            pc.QuantileFilter(),
            pc.ZeroInjector(500),
            TableExtension(),
            pc.MinimalQualityAssurance(),
        ]

        keys = [
            "data_handler",
            "imputator",
            "smoother",
            "pre_expander_weigher",
            "pre_expander_filter",
            "expander",
            "pre_influence_weigher",
            "pre_influence_filter",
            "influence_model",
            "post_weigher",
            "post_filter",
            "injector",
            "extension",
            "quality_assurance",
        ]

        aliases = {
            "filter": "post_filter",
            "weigher": "post_weigher",
            "pre_filter": "pre_influence_filter",
            "pre_weigher": "pre_influence_weigher",
        }

        self._set_with_default(custom_components, keys, defaults, aliases)

    def __call__(
        self, training_data, test_data=None, apparent_wind=False, **enabling
    ):
        """
        Parameters
        ----------
        training_data : list of data compatible with `self.data_handler`
            Data from which to create the polar diagram.

            The input should be compatible with the DataHandler instances
            given in initialization of the pipeline instance.
            Also, the input should be suitable to be interpreted as chronologically
            ordered time series by the before-mentioned data handler.

        test_data: list of data compatible with `self.data_handler` or `None`
            Data which is preprocessed and then used to check the quality of
            the resulting polar diagram.

            The input should be compatible with the DataHandler instances
            given in initialization of the pipeline instance.
            If `None` no quality check is performed.

            Default to `None`.

        apparent_wind : bool, optional
            Specifies if wind data is given in apparent wind.

            If `True`, wind will be converted to true wind.

            Defaults to `False`.

        Other Parameters
        ---------------------------

        pre_expander_weighing : bool, optional
            Specifies, whether the pre_influence_weigher should be applied before application of the
            expander.
            Otherwise, each point will be assigned the weight 1.

            Defaults to `True`.

        pre_expander_filtering : bool, optional
            Specifies, whether the pre_influence_filter should be applied before application of the expander.

            Defaults to `True`.


        pre_influence_weighing, pre_weighing : bool, optional
            Specifies, whether the pre_influence_weigher should be applied before application of the
            influence model.
            Otherwise, each point will be assigned the weight 1.

            Defaults to `True`.

        pre_influence_filtering, pre_filtering : bool, optional
            Specifies, whether the pre_influence_filter should be applied before application of the influence model.

            Defaults to `True`.

        smoothing : bool, optional
            Specifies, if measurement errors of the time series should be
            smoothened after pre_filtering.

        post_weighing, weighing : bool, optional
            Specifies, if points should be weighed after application of the
            influence model.
            Otherwise, each point will be assigned the weight 1.

            Defaults to `True`.

        post_filtering, filtering : bool, optional
            Specifies, if points should be filtered after post_weighing.

            Defaults to `True`.

        injecting : bool, optional
            Specifies, if artificial points should be added to the data.

            Defaults to `True`.

        testing : bool, optional
            Specifies, if the resulting polar diagram should be tested against
            test data.

        Returns
        -------
        out : PipelineOutput
        """
        start_time = dt.now()
        keys = [
            "smoothing",
            "pre_expander_weighing",
            "pre_expander_filtering",
            "pre_influence_weighing",
            "pre_influence_filtering",
            "post_weighing",
            "post_filtering",
            "injecting",
            "testing",
        ]
        aliases = {
            "pre_weighing": "pre_influence_weighing",
            "pre_filtering": "pre_influence_filtering",
            "weighing": "post_weighing",
            "filtering": "post_filtering",
        }
        defaults = [True] * len(keys)

        self._set_with_default(enabling, keys, defaults, aliases)

        if test_data is None:
            self.testing = False

        preproc_training_data, pp_training_statistics = self._preprocess(
            training_data,
            True,
        )

        if is_empty_data(preproc_training_data):
            raise RuntimeError(
                "Empty data after preprocessing. Try to use weaker filters."
            )

        if self.injecting:
            pts_to_inject, injector_statistics = _collect(
                self.injector, self.injector.inject, preproc_training_data
            )
        else:
            pts_to_inject, injector_statistics = (
                pc.WeightedPoints(np.empty((0, 3)), np.empty(0)),
                {},
            )

        preproc_training_data.extend(pts_to_inject)

        polar_diagram, extension_statistics = _collect(
            self.extension, self.extension.process, preproc_training_data
        )

        if self.testing:
            preproc_test_data, test_statistics = self._preprocess(
                test_data,
                False,
            )
            quality_assurance_statistics = self.quality_assurance.check(
                polar_diagram, preproc_test_data.data
            )
        else:
            test_statistics = _EMPTY_STATISTIC
            quality_assurance_statistics = {}

        quality_assurance_statistics["execution time"] = (
            dt.now() - start_time
        ).total_seconds()

        training_statistics = Statistics(
            pp_training_statistics.data_handler,
            pp_training_statistics.imputator,
            pp_training_statistics.smoother,
            pp_training_statistics.pre_expanding_weigher,
            pp_training_statistics.pre_influence_filter,
            pp_training_statistics.expander,
            pp_training_statistics.pre_influence_weigher,
            pp_training_statistics.pre_influence_filter,
            pp_training_statistics.influence_model,
            pp_training_statistics.post_weigher,
            pp_training_statistics.post_filter,
            injector_statistics,
            extension_statistics,
            quality_assurance_statistics,
        )

        return PipelineOutput(
            polar_diagram, training_statistics, test_statistics
        )

    def _preprocess(self, data, influence_fitting):
        data, handler_statistics = self._handle_data(data)

        data, imputator_statistics = self._map(
            _collector_fun(self.imputator, self.imputator.impute), data
        )

        if self.smoothing:
            data, smooth_statistics = self._map(
                _collector_fun(self.smoother, self.smoother.smooth),
                data,
            )
        else:
            smooth_statistics = {}

        (
            data,
            pre_exp_weigher_statistics,
            pre_exp_filter_statistics,
        ) = self._map(
            lambda x: _weigh_and_filter(
                x,
                self.pre_expander_weigher,
                self.pre_expander_filter,
                self.pre_expander_weighing,
                self.pre_expander_filtering,
            ),
            data,
        )

        data = [weighted_point.data for weighted_point in data]

        data, expanded_statistics = self._map(
            _collector_fun(self.expander, self.expander.expand),
            data,
        )

        (
            data,
            pre_weigher_statistics,
            pre_filter_statistics,
        ) = self._map(
            lambda data: _weigh_and_filter(
                data,
                self.pre_influence_weigher,
                self.pre_influence_filter,
                self.pre_influence_weighing,
                self.pre_influence_filtering,
            ),
            data,
        )

        data = pc.data.Data.concatenate([wp.data for wp in data])

        if influence_fitting:
            self.influence_model.fit(data)
            influence_fit_statistics = (
                self.influence_model.get_latest_statistics()
            )
        else:
            influence_fit_statistics = {}

        data, influence_statistics = _collect(
            self.influence_model,
            self.influence_model.remove_influence,
            data,
        )

        influence_statistics.update(influence_fit_statistics)

        (
            data,
            post_weigher_statistics,
            post_filter_statistics,
        ) = _weigh_and_filter(
            data,
            self.post_weigher,
            self.post_filter,
            self.post_weighing,
            self.post_filtering,
        )

        statistics = Statistics(
            handler_statistics,
            imputator_statistics,
            smooth_statistics,
            pre_exp_weigher_statistics,
            pre_exp_filter_statistics,
            expanded_statistics,
            pre_weigher_statistics,
            pre_filter_statistics,
            influence_statistics,
            post_weigher_statistics,
            post_filter_statistics,
            {},
            {},
            {},
        )

        return data, statistics

    def _set_with_default(self, dict_, keys, defaults, aliases):
        for key in list(dict_.keys()):
            if key in aliases:
                dict_[aliases[key]] = dict_[key]
        for key, default in zip(keys, defaults):
            setattr(self, key, self._switch_default(dict_, key, default))

    def _switch_default(self, dict_, key, default):
        return dict_[key] if key in dict_ else default

    def _handle_data(self, data):
        if isinstance(self.data_handler, pc.DataHandler):
            handler_output = [
                _collect(self.data_handler, self.data_handler.handle, field)
                for field in data
            ]
        else:
            handler_output = []
            for field in data:
                handler = next(self.data_handler)
                handler_output.append(_collect(handler, handler.handle, field))

        return self._list_statistics(handler_output)

    def _map(self, method, data):
        output = [method(field) for field in data]
        return self._list_statistics(output)

    @staticmethod
    def _list_statistics(pipe_output):
        data, *statistics = tuple(zip(*pipe_output))
        statistics = tuple(
            dict(enumerate(statistic)) for statistic in statistics
        )
        return (list(data),) + statistics


def _collect(comp, method, data):
    out = method(data)
    statistics = comp.get_latest_statistics()
    return out, statistics


def _collector_fun(comp, method):
    return lambda data: _collect(comp, method, data)


def is_empty_data(data):
    if isinstance(data, pc.data.Data) and data.n_rows == 0:
        return True
    if isinstance(data, np.ndarray) and len(data) == 0:
        return True
    if isinstance(data, pc.WeightedPoints):
        return is_empty_data(data.data)
    return False


def _weigh_and_filter(data, weigher, filter_, weighing, filtering):
    if is_empty_data(data):
        return (pc.WeightedPoints(data, []), {}, {})
    if weighing:
        weights, weigher_statistics = _collect(weigher, weigher.weigh, data)
    else:
        def_weigher = pc.AllOneWeigher()
        weights, weigher_statistics = _collect(
            def_weigher, def_weigher.weigh, data
        )

    weighed_data = pc.WeightedPoints(data, weights)

    filtered_data, filter_statistics = (
        _filter_data(filter_, weighed_data)
        if filtering
        else (weighed_data, {})
    )

    return filtered_data, weigher_statistics, filter_statistics


def _filter_data(filter_, weighted_points):
    points_to_filter, filter_statistics = _collect(
        filter_, filter_.filter, weighted_points.weights
    )
    return weighted_points[points_to_filter], filter_statistics


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
