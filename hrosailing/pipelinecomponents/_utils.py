# pylint: disable=missing-module-docstring

import numpy as np


def scaled_norm(norm, scal_factors):
    scaled = np.asarray(scal_factors)

    def s_norm(vec):
        return norm(scaled * vec)

    return s_norm


def euclidean_norm(vec):
    return np.linalg.norm(vec, axis=1)


scaled_euclidean_norm = scaled_norm(euclidean_norm, [1 / 40, 1 / 360])


def data_dict_to_numpy(data_dict, keys):
    """
    Method to transform a dictionary of lists of floats into a numpy array

    Parameter
    ---------
    data_dict : dict,
        dictionary to transform

    keys : [str],
        keys that indicate which lists of the data dictionary should be used
        to create the columns of the resulting array

    Returns
    --------
    (n, d) array where 'n' is the length of a list in the data dictionary and
    'd' is 'len(keys)'
    """
    return np.column_stack([data_dict[key] for key in keys])