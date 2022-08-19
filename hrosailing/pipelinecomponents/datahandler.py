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
from datetime import date, time, datetime
from decimal import Decimal

import numpy as np

from hrosailing.pipelinecomponents.constants import KEYSYNONYMS
from hrosailing.cruising.weather_model import OutsideGridException
from hrosailing.pipelinecomponents.data import Data


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
    def handle(self, data) -> (dict, dict):
        """This method should be used to read given data in a format
        that is dependent on the handler.
        The output should be a tuple of two dictionaries, the first should be
        a dictionary with str keys and list values containing the read data,
        the second should be a dictionary of relevant statistics.

        We recommend using 'hrosailing_standard_format' at the end of your
        custom handler.
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
        data_dict, statistics: (dict, dict)
            If data is a pandas.DataFrame, data_dict is the output
            of the DataFrame.to_dict()-method, otherwise the keys of
            the dict will be the entries of the ordered iterable with the
            value being the corresponding column of the array_like

            statistics contains the number of created rows and colums
            as 'n_rows' and 'n_cols' respectively

        Raises
        ------
        HandleException
        """
        if self.pand and isinstance(data, self.pd.DataFrame):
            data_dict = data.to_dict()
            data_dict = Data().update(data_dict)
        else:
            arr, keys = data
            arr = np.asarray(arr)

            if len(keys) != arr.shape[1]:
                raise HandleException("Too few keys for data")

            data_dict = {key: arr[:, i] for i, key in enumerate(keys)}
            data_dict = Data().update(data_dict)

        data_dict.hrosailing_standard_format()

        statistics = get_datahandler_statistics(data_dict)
        return data_dict, statistics


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

    def handle(self, data) -> (dict, dict):
        """Reads a .csv file and extracts the contained data points
        The delimiter used in the .csv file

        Parameters
        ----------
        data : path_like
            Path to a .csv file

        Returns
        -------
        data_dict, statistics : dict, dict
            data_dict is a dictionary having the hrosailing standard version of
            the first row entries as keys and
            as values the corresponding columns given as lists

            statistics contains the number of created rows and colums
            as 'n_rows' and 'n_cols' respectively

        Raises
        ------
        OSError
            If no read permission is given for file
        """
        if self.pand:
            df = self.pd.read_csv(data)
            data_dict = Data().update(df.to_dict())
        else:
            with open(data, "r", encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                keys = next(csv_reader)
                data_dict = {key: [] for key in keys}
                for row in csv_reader:
                    for i, entry in enumerate(row):
                        data_dict[keys[i]].append(literal_eval(entry))
                data_dict = Data().update()

        data_dict.hrosailing_standard_format()

        statistics = get_datahandler_statistics(data_dict)
        return data_dict, statistics


class MultiDataHandler(DataHandler):
    """A data handler that uses multiple data handlers and weather models
    in conjunction

    Parameters
    -----------
    ?????
    """
    def __init__(self, handlers, weather_models):
        self._handlers = handlers
        self._weather_models = weather_models

    def handle(self, data) -> (dict, dict):
        comp_dict = {}
        comp_statistics = []
        for dh, data_entry in zip(self._handlers, data):
            data_dict, statistics = dh.handle(data_entry)
            self._update(comp_dict, data_dict)
            comp_statistics.append(statistics)

        try:
            coords = zip(
                comp_dict["lat"],
                comp_dict["lon"],
                comp_dict["datetime"]
            )
        except KeyError:
            statistics = {
                "data_handler_statistics": comp_statistics,
                "weather_model_statistics":
                    "coordinates not given in 'lat', 'lon', 'datetime'"
            }
            return comp_dict, statistics

        weather_dict = {}

        for lat, lon, time in coords:
            for wm in self._weather_models:
                try:
                    weather = wm.get_weather(lat, lon, time)
                    weather = {key: [value] for key, value in weather.items()}
                    self._update(weather_dict, weather)
                    break
                except OutsideGridException:
                    continue
            self._add_col(weather_dict)

        comp_dict.update(weather_dict)

        statistics = {
            "data_handler_statistics": comp_statistics,
            "weather_model_statistics": {}
        }

        return comp_dict, statistics

    @staticmethod
    def _update(dict_, data):
        max_len = max([len(value) for _, value in data.items()])
        for key, value in data.items():
            if key not in dict_:
                dict_[key] = []
            dict_[key].extend([None]*(max_len - len(dict_[key])))
            dict_[key].extend(value)

    @staticmethod
    def _add_col(dict_):
        for key in dict_.keys():
            dict_[key].append([None])


class NMEAFileHandler(DataHandler):
    """A data handler to extract data from a text file containing
    certain nmea sentences and convert it to a dictionary

    Parameters
    ---------
    wanted_sentences : Iterable of str, optional
        NMEA sentences that will be read

        Default to 'None'

    unwanted_sentences : Iterable of str, optional
        NMEA sentences that will be ignored.
        If 'wanted_sentences' is set this keyword is ignored
        If both 'wanted_sentences' and 'unwanted_sentences' are None,
        all NMEA sentences will be read

        Defaults to 'None'

        wanted_sentences : Iterable of str,
        NMEA sentences that will be read

        Default to 'None'


    wanted_attributes : Iterable of str, optional
        NMEA attributes that will be added to the output dictionary.
        If set to "numerical" all fields with float values will be read.

        Default to 'None'

    unwanted_attributes : Iterable of str, optional
        NMEA attributes that will be ignored
        If 'wanted_attributes' is set this option is ignored
        If both 'wanted_sentences' and 'unwanted_sentences' are None,
        all NMEA sentences will be read

        Defaults to 'None'

    post_filter_types: tuple of types, optional
        The resulting dictionary only contains data which is `None` or of a
        type given in `post_filter_types`.
        If set to `False` all attributes are taken into account

        Defaults to `(float, datetime.date, datetime.time, datetime.datetime)`
    """

    def __init__(
            self,
            wanted_sentences=None,
            wanted_attributes=None,
            unwanted_sentences=None,
            unwanted_attributes=None,
            post_filter_types=(float, date, time, datetime)
    ):
        if wanted_sentences is not None:
            self._sentence_filter = lambda line: any(
                    sentence in line for sentence in wanted_sentences
                )
        elif unwanted_sentences is not None:
            self._sentence_filter = lambda line: all(
                    sentence not in line for sentence in unwanted_sentences
                )
        else:
            self._sentence_filter = lambda line: True

        if wanted_attributes is not None:
            self._attribute_filter = lambda field: any(
                field[0] == attribute
                for attribute in wanted_attributes
                )
        elif unwanted_attributes is not None:
            self._attribute_filter = lambda field: all(
                field[0] != attribute
                for attribute in unwanted_attributes
                )
        else:
            self._attribute_filter = lambda field: True

        self._post_filter_types = post_filter_types

    def handle(self, data) -> (dict, dict):
        """Reads a text file containing nmea-sentences and extracts
        data points

        Parameters
        ----------
        data : path-like
            Path to a text file, containing nmea-0183 sentences


        Returns
        -------
        comp_data: Data
            The data read from the file where the columns are the filtered
            attributes

        statistics : dict
            Contains the number of created rows and colums
            as 'n_rows' and 'n_cols' respectively
            data_dict is a dictionary having the hrosailing standard version of the
            first row entries as keys and
            as values the corresponding columns given as lists

        Raises
        ------
        OSError
            If no read permission is given for file
        """
        from pynmea2 import parse

        comp_data = Data()
        ndata = 0

        with open(data, "r", encoding="utf-8") as file:
            nmea_sentences = filter(
                self._sentence_filter,
                file,
            )

            for sentence in nmea_sentences:
                parsed_sentence = parse(sentence)
                wanted_fields = filter(
                    self._attribute_filter,
                    parsed_sentence.fields,
                )

                wanted_fields = [x[:2] for x in wanted_fields]

                comp_data.update({
                    name: [getattr(parsed_sentence, attribute)]
                    for name, attribute in wanted_fields
                })

        comp_data.hrosailing_standard_format()

        comp_data.filter_types(self._post_filter_types)

        statistics = get_datahandler_statistics(comp_data)

        return comp_data, statistics


def get_datahandler_statistics(data):
    """
    Computes standard statistics for the output of a data handler.
    """
    return {
        "n_rows": data.n_rows,
        "n_cols": data.n_cols
    }
