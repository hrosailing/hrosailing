"""
Classes used to

Defines the DataHandler Abstract Base Class that can be used to
create custom

Subclasses of DataHandler can be used with the PolarPipeline class
in the hrosailing.pipeline module
"""


# pylint: disable=import-outside-toplevel
# pylint: disable=import-error


import csv
from abc import ABC, abstractmethod
from ast import literal_eval
from datetime import date, time
from decimal import Decimal

import numpy as np


class HandlerInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of a DataHandler
    """


class HandleException(Exception):
    """Exception raised if an error occurs during
    calling of the .handle() method
    """


class DataHandler(ABC):
    """Base class for all datahandler classes


    Abstract Methods
    ----------------
    handle(self, data)
    """

    @abstractmethod
    def handle(self, data) -> dict:
        """This method should be used, given some data in a format
        that is dependent on the handler, to output a dictionary
        containing the given data, where the values should be
        lists.

        The dictionary should atleast contain the following keys:
        'Wind speed', 'Wind angle' and one of 'Speed over ground knots',
        'Water speed knots' or 'Boat speed'

        The names of the keys in the dictionary should also be compatible
        with the keys that a possible InfluenceModel instance might use
        """


class ArrayHandler(DataHandler):
    """A data handler to convert data given as an array-type
    to a dictionary
    """

    # ArrayHandler usable even if pandas is not installed.
    try:
        __import__("pandas")
        pand = True
    except ImportError:
        pand = False

    if pand:
        import pandas as pd

    def handle(self, data) -> dict:
        """Extracts data from array-types of data

        Parameters
        ----------
        data: pandas.DataFrame or tuple of array_like and ordered iterable
            Data contained in a pandas.DataFrame or in an array_like.

        Returns
        -------
        data_dict: dict
            If data is a pandas.DataFrame, data_dict is the output
            of the DataFrame.to_dict()-method, otherwise the keys of
            the dict will be the entries of the ordered iterable with the
            value being the corresponding column of the array_like

        Raises
        ------
        HandleException
        """
        if self.pand and isinstance(data, self.pd.DataFrame):
            return data.to_dict()

        arr, keys = data
        arr = np.asarray(arr)

        if len(keys) != arr.shape[1]:
            raise HandleException("Too few keys for data")

        return {key: arr[:, i] for i, key in enumerate(keys)}


class CsvFileHandler(DataHandler):
    """A data handler to extract data from a .csv file and convert it
    to a dictionary

    .csv file should be ordered in a column-wise fashion, with the
    first row, describing what each column represents
    """

    # Check if pandas is available to use from_csv()-method
    try:
        __import__("pandas")
        pand = True
    except ImportError:
        pand = False

    if pand:
        import pandas as pd

    def handle(self, data) -> dict:
        """Reads a .csv file and extracts the contained data points
        The delimiter used in the .csv file

        Parameters
        ----------
        data : path_like
            Path to a .csv file

        Returns
        -------
        data_dict : dict
            Dictionary having the first row entries as keys and
            as values the corresponding columns given as lists

        Raises
        ------
        OSError
            If no read permission is given for file
        """
        if self.pand:
            df = self.pd.read_csv(data)
            return df.to_dict()

        with open(data, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            keys = next(csv_reader)
            data_dict = {key: [] for key in keys}
            for row in csv_reader:
                for i, entry in enumerate(row):
                    data_dict[keys[i]].append(literal_eval(entry))

        return data_dict


class NMEAFileHandler(DataHandler):
    """A data handler to extract data from a text file containing
    certain nmea sentences and convert it to a dictionary

    Parameters
    ---------
    sentences : Iterable of str,

    attributes : Iterable of str,

    """

    def __init__(self, sentences, attributes):
        self._wanted_sentences = sentences
        self._wanted_attributes = attributes

    def handle(self, data) -> dict:
        """Reads a text file containing nmea-sentences and extracts
        data points

        Parameters
        ----------
        data : path-like
            Path to a text file, containing nmea-0183 sentences

        Returns
        -------
        data_dict : dict
            Dictionary where the keys are the given attributes

        Raises
        ------
        OSError
            If no read permission is given for file
        """
        from pynmea2 import parse

        data_dict = {attribute: [] for attribute in self._wanted_attributes}
        ndata = 0

        with open(data, "r", encoding="utf-8") as file:
            nmea_sentences = filter(
                lambda line: any(
                    sentence in line for sentence in self._wanted_sentences
                ),
                file,
            )

            for sentence in nmea_sentences:
                parsed_sentence = parse(sentence)
                wanted_fields = filter(
                    lambda field: any(
                        field[0] == attribute
                        for attribute in self._wanted_attributes
                    ),
                    parsed_sentence.fields,
                )

                wanted_fields = map(lambda x: x[:2], wanted_fields)

                for name, attribute in wanted_fields:
                    length = len(data_dict[name])
                    if length == ndata:
                        ndata += 1
                    else:
                        data_dict[name].extend([None] * (ndata - length - 1))

                    value = getattr(parsed_sentence, attribute)
                    if isinstance(value, Decimal):
                        value = float(value)

                    data_dict[name].append(value)

            # fill last entries
            for attribute in self._wanted_attributes:
                length = len(data_dict[attribute])
                data_dict[attribute].extend([None] * (ndata - length))

        # componentwise completion of data entries
        _handle_surplus_data(data_dict)
        return data_dict


def _handle_surplus_data(data_dict):
    idx_dict = {
        key: [i for i, data in enumerate(data_dict[key]) if data is not None]
        for key in data_dict
    }

    for key, idx in idx_dict.items():
        # every entry before the first non-None entry gets the value of
        # the first non-None entry
        first = data_dict[key][idx[0]]
        data_dict[key][0 : idx[0]] = [first] * idx[0]

        # convex interpolation of entries between non-None entries
        for idx1, idx2 in zip(idx, idx[1:]):
            lambda_ = idx2 - idx1
            left = data_dict[key][idx1]
            right = data_dict[key][idx2]

            if isinstance(left, (str, time, date)):
                data_dict[key][idx1 + 1 : idx2] = [left] * (idx2 - (idx1 + 1))
                continue

            k = 1
            for i in range(idx1 + 1, idx2):
                mu = k / lambda_
                data_dict[key][i] = mu * left + (1 - mu) * right
                k += 1

        # every entry after the last non-None entry gets the value of
        # the last non-None entry
        last = data_dict[key][idx[-1]]
        data_dict[key][idx[-1] :] = [last] * (len(data_dict[key]) - idx[-1])
