"""
Pipeline to create polar diagrams from raw data.
"""


import warnings
from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np

import hrosailing.pipelinecomponents as pc
import hrosailing.polardiagram as pol
from .extensions import (
    PipelineExtension, TableExtension, PointcloudExtension, CurveExtension
)
from hrosailing.pipelinecomponents.modelfunctions import (
    ws_s_wa_gauss_and_square,
)

class Statistics(NamedTuple):
    """"""
    data_handler: dict
    imputator: dict
    pre_weigher: dict
    pre_filter: dict
    smoother: dict
    expander: dict
    influence_model: dict
    post_weigher: dict
    post_filter: dict
    injector: dict
    extension: dict
    quality_assurance: dict


_EMPTY_STATISTIC = Statistics({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})


class PipelineOutput(NamedTuple):
    """"""
    polardiagram: pol.PolarDiagram
    training_statistics: Statistics
    test_statistics: Statistics


class PolarPipeline:
    """A Pipeline class to create polar diagrams from raw data

    Parameters
    ----------

    data_handler : DataHandler or list of DataHandler
        Handlers that are responsible to extract actual data from the input.
        If only one handler is given, this handler will be used for all given inputs,
        otherwise the handlers will be used one after another for each data input including the training data.

        Determines the type and format of input the pipeline should accept

    imputator : Imputator, optional
        Determines the method which will be used to produce data without
        None entries

        Defaults to `FillLocalImputator()`

    smoother: Smoother, optional
        Determines the method which will be used to smoothen out the rounding
        following from low measurement precision.

        Defaults to `LazySmoother()`

    expander: Expander, optional
        Determines the method which will be used to expand the data by several more data fields.
        For example weather data from a weather model.

    pre_weigher : Weigher, optional
        Determines the method with which the points will be weight before
        application of the influence model.

        Defaults to `CylindricMeanWeigher()`

    pre_filter : Filter, optional
        Determines the methods with which the points will be filtered,
        if `pre_filtering` in __call__ method

        Defaults to `QuantileFilter()`

    influence_model : InfluenceModel, optional
        Determines the influence model which is applied and fitted to the data

        Defaults to 'IdentityInfluenceModel()'

    post_weigher : Weigher, optional
        Determines the method with which the points will be weight after
        application of the influence model.

        Defaults to `CylindricMeanWeigher()`

    post_filter : Filter, optional
        Determines the methods with which the points will be filtered
        after the apllication of the influence model,
        if `post_filtering` in __call__ method

        Defaults to `QuantileFilter()`

    injector : Injector, optional
        Determines the method to add additional artificial data points to the
        data

        Defaults to 'None'

    extension: PipelineExtension
        Extension that is called in the pipeline, after all preprocessing
        is done, to generate a polar diagram from the processed data.

        Determines the subclass of `PolarDiagram`, that the pipeline will
        produce

        Defaults to 'None'

    quality_assurance : QualityAssurance, optional
        Determines the method which is used to measure the quality of the
        resulting polar diagram using preprocessed test_data
    """

    def __init__(
        self,
        data_handler=pc.NMEAFileHandler(),
        imputator=pc.FillLocalImputator(),
        expander=pc.LazyExpander(),
        pre_weigher=pc.CylindricMeanWeigher(),
        pre_filter=pc.QuantileFilter(),
        smoother=pc.LazySmoother(),
        influence_model=pc.IdentityInfluenceModel(),
        post_weigher=pc.CylindricMeanWeigher(),
        post_filter=pc.QuantileFilter(),
        injector=pc.ZeroInjector(500),
        extension=TableExtension(),
        quality_assurance=pc.MinimalQualityAssurance()
    ):
        self.data_handler = data_handler
        self.imputator = imputator
        self.pre_weigher = pre_weigher
        self.pre_filter = pre_filter
        self.smoother = smoother
        self.expander = expander
        self.influence_model = influence_model
        self.post_weigher = post_weigher
        self.post_filter = post_filter
        self.injector = injector
        self.extension = extension
        self.quality_assurance = quality_assurance

    def __call__(
        self,
        training_data,
        test_data=None,
        apparent_wind=False,
        pre_weighing=True,
        pre_filtering=True,
        smoothing=True,
        post_weighing=True,
        post_filtering=True,
        injecting=True,
        testing=True
    ):
        """
        Parameters
        ----------
        training_data : list of data compatible with `self.data_handler`
            Data from which to create the polar diagram

            The input should be compatible with the DataHandler instances
            given in initialization of the pipeline instance

        test_data: list of data compatible with `self.data_handler` or `None`
            Data which is preprocessed and then used to check the quality of
            the resulting polar diagram

            The input should be compatible with the DataHandler instances
            given in initialization of the pipeline instance.
            If `None` no quality check is performed.

            Default to `None`

        apparent_wind : bool, optional
            Specifies if wind data is given in apparent wind

            If `True`, wind will be converted to true wind

            Defaults to `False`

        pre_weighing : bool, optional
            Specifies, if points should be weighed before application of the
            influence model.
            Otherwise each point will be assigned the weight 1.

            Defaults to `True`

        pre_filtering : bool, optional
            Specifies, if points should be filtered after pre_weighing

            Defaults to `True`

        smoothing : bool, optional
            Specifies, if measurement errors of the time series should be
            smoothened after pre_filtering.

        post_weighing : bool, optional
            Specifies, if points should be weighed after application of the
            influence model.
            Otherwise each point will be assigned the weight 1.

            Defaults to `True`

        post_filtering : bool, optional
            Specifies, if points should be filtered after post_weighing

            Defaults to `True`

        injecting : bool, optional
            Specifies, if artificial points should be added to the data

            Defaults to `True`

        testing : bool, optional
            Specifies, if the resulting polar diagram should be tested against
            test data

        Returns
        -------
        out : PipelineOutput
        """

        if test_data is None:
            testing = False

        preproc_training_data, pp_training_statistics = self._preprocess(
            training_data,
            pre_weighing,
            pre_filtering,
            smoothing,
            post_weighing,
            post_filtering,
            True
        )

        if injecting:
            pts_to_inject, injector_statistics \
                = self.injector.inject(preproc_training_data)
        else:
            pts_to_inject, injector_statistics \
                = pc.WeightedPoints(np.array((0, 3)), np.array(0)), {}

        preproc_training_data.extend(pts_to_inject)

        polar_diagram, extension_statistics = self.extension.process(
            preproc_training_data
        )

        if testing:
            preproc_test_data, test_statistics = self._preprocess(
                test_data,
                pre_weighing,
                pre_filtering,
                post_weighing,
                post_filtering,
                False
            )
            quality_assurance_statistics = self.quality_assurance.check(
                polar_diagram,
                preproc_test_data.data
            )
        else:
            test_statistics = _EMPTY_STATISTIC
            quality_assurance_statistics = {}

        training_statistics = Statistics(
            pp_training_statistics.data_handler,
            pp_training_statistics.imputator,
            pp_training_statistics.pre_weigher,
            pp_training_statistics.pre_filter,
            pp_training_statistics.smoother,
            pp_training_statistics.expander,
            pp_training_statistics.influence_model,
            pp_training_statistics.post_weigher,
            pp_training_statistics.post_filter,
            injector_statistics,
            extension_statistics,
            quality_assurance_statistics
        )

        return PipelineOutput(
            polar_diagram,
            training_statistics,
            test_statistics
        )

    def _preprocess(
        self,
        data,
        pre_weighing,
        pre_filtering,
        smoothing,
        post_weighing,
        post_filtering,
        influence_fitting
    ):

        handled_data, handler_statistics = self._handle_data(data)

        imputated_data, imputator_statistics = self._imputate_data(
            handled_data
        )

        imputated_data = self._concatenate_data(imputated_data)

        pre_filtered_data, pre_weigher_statistics, pre_filter_statistics = \
            _weigh_and_filter(
                imputated_data,
                self.pre_weigher,
                self.pre_filter,
                pre_weighing,
                pre_filtering
            )

        if smoothing:
            smooth_data, smooth_statistics = self.smoother.smooth(
                pre_filtered_data.data
            )
        else:
            smooth_data = pre_filtered_data.data
            smooth_statistics = {}

        expanded_data, expanded_statistics = self.expander.expand(smooth_data)

        if influence_fitting:
            influence_fit_statistics = self.influence_model.fit(expanded_data)
        else:
            influence_fit_statistics = {}

        influence_free_data, influence_statistics = \
            self.influence_model.remove_influence(expanded_data)

        influence_statistics.update(influence_fit_statistics)

        post_filtered_data, post_weigher_statistics, post_filter_statistics = \
            _weigh_and_filter(
                influence_free_data,
                self.post_weigher,
                self.post_filter,
                post_weighing,
                post_filtering
            )

        statistics = Statistics(
            handler_statistics,
            imputator_statistics,
            pre_weigher_statistics,
            pre_filter_statistics,
            smooth_statistics,
            expanded_statistics,
            influence_statistics,
            post_weigher_statistics,
            post_filter_statistics,
            {},
            {},
            {}
        )

        return post_filtered_data, statistics

    def _handle_data(self, data):
        if isinstance(self.data_handler, pc.DataHandler):
            handler_output = [self.data_handler.handle(field) for field in data]
        else:
            handler_output = []
            for field in data:
                handler = next(self.data_handler)
                handler_output.append(handler.handle(field))

        return self._list_statistics(handler_output)

    def _imputate_data(self, data):
        imputator_output = [self.imputator.imputate(field) for field in data]
        return self._list_statistics(imputator_output)

    @staticmethod
    def _list_statistics(pipe_output):
        data, statistics = tuple(zip(*pipe_output))
        statistics = {
            i: stat for i, stat in enumerate(statistics)
        }
        return list(data), statistics

    @staticmethod
    def _concatenate_data(list_of_data):
        concat = {}
        max_len = 0
        for data in list_of_data:
            for key, val in data.items():
                if key in concat:
                    concat[key].append(val)
                    max_len = max(len(concat[key]), max_len)
                else:
                    concat[key] = [None]*max_len
                    concat[key].append(val)
        return concat



def _weigh_and_filter(
    data,
    weigher,
    filter_,
    weighing,
    filtering
):
    weights, weigher_statistics = \
        weigher.weigh(data) if weighing \
        else pc.AllOneWeigher().weigh(data)

    weighed_data = pc.WeightedPoints(data, weights)

    filtered_data, filter_statistics = \
        _filter_data(filter_, weighed_data) if filtering \
        else (weighed_data, {})

    return filtered_data, weigher_statistics, filter_statistics


def _filter_data(filter, weighted_points):
    points_to_filter, filter_statistics = \
        filter.filter(weighted_points.weights)
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
