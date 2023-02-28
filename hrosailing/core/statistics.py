"""
Functions for the handling of statistics.
"""


class ComponentWithStatistics:
    """
    Interface class for pipelinecomponents enabling to save and handle run statistics.
    """

    def __init__(self):
        self._statistics = {}

    def set_statistics(self, **kwargs):
        """
        Sets the statistics dict corresponding to the keyword arguments.
        Supposed to be overwritten by inheriting classes with stronger requirements.
        """
        self._statistics = kwargs

    def get_latest_statistics(self):
        """
        Return
        ------
        statistics: dict
            The statistics of the latest run of the handler.
        """
        return self._statistics
