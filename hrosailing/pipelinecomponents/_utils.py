# pylint: disable=missing-module-docstring

import numpy as np


def scaled_norm(norm, scal_factors):
    scaled = np.asarray(scal_factors)

    def s_norm(vec):
        return norm(scaled * vec)

    return s_norm


def euclidean_norm(vec):
    return np.linalg.norm(vec, axis=1)


def scaled_euclidean_norm(vec):
    if vec.shape[1] == 2:
        norm_val = scaled_norm(euclidean_norm, [1 / 40, 1 / 360])(vec)
    elif vec.shape[1] == 3:
        norm_val = scaled_norm(euclidean_norm, [1 / 40, 1 / 360, 1 / 20])(vec)
    else:
        raise NotImplementedError(
            "scaled_euclidean_norm only supports 2 and 3 dimensional inputs"
        )
    return norm_val


def data_dict_to_numpy(data_dict, keys):
    """
    Method to transform a dictionary of lists of floats into a numpy array.

    Parameter
    ---------
    data_dict : dict,
        Dictionary to transform.

    keys : [str],
        Keys that indicate which lists of the data dictionary should be used
        to create the columns of the resulting array.

    Returns
    --------
    (n, d) array where 'n' is the length of a list in the data dictionary and
    'd' is 'len(keys)'.
    """
    return np.column_stack([data_dict[key] for key in keys])


def _safe_operation(operand, value):
    try:
        return operand(value)
    except (ValueError, TypeError, IndexError, KeyError, ZeroDivisionError):
        return None


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
