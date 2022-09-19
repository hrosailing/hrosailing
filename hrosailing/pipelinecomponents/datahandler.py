"""
Classes used to transform given data to the hrosailing data format for further processing.

Defines the `DataHandler` abstract base class that can be used to
create custom data handlers to handle formats which are not supported yet.

Subclasses of `DataHandler` can be used with the `PolarPipeline` class
in the `hrosailing.pipeline` module.
"""


# pylint: disable=import-outside-toplevel
# pylint: disable=import-error


import csv
from abc import ABC, abstractmethod
from ast import literal_eval
from datetime import date, time, datetime

import numpy as np

import hrosailing.pipelinecomponents.data
from hrosailing.pipelinecomponents.data import Data
from hrosailing.pipelinecomponents.constants import HROSAILING_TO_NMEA


class HandlerInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of a `DataHandler`.
    """


class HandleException(Exception):
    """Exception raised if an error occurs during
    calling of the `.handle()`-method.
    """


class DataHandler(ABC):
    """Base class for all data handler classes.


    Abstract Methods
    ----------------
    handle(self, data)
    """

    @abstractmethod
    def handle(self, data):
        """This method should be used to interpret given data in a format
        that is dependent on the handler.

        We recommend using the `hrosailing.pipelinecomponents.data.Data.hrosailing_standard_format`-method
        at the end of your custom handler for compatibility with other built in components.

        Parameters
        ----------
        data :
            Data in a format compatible to the inheriting handler.

        Returns
        -------
        data : Data
            The interpreted data in hrosailing format.

        statistics : dict
            Relevant statistics. If not stated otherwise contains the number of created rows and columns
            as `n_rows` and `n_cols` respectively.
        """


class ArrayHandler(DataHandler):
    """A data handler to interpret data given as an array-type.
    """

    # ArrayHandler usable even if pandas is not installed.
    try:
        __import__("pandas")
        pand = True
    except ImportError:
        pand = False

    if pand:
        import pandas as pd

    def handle(self, data):
        """Extracts data from array-types of data.

        Parameters
        ----------
        data : pandas.DataFrame or tuple containing an array_like and an ordered iterable
            If given as a tuple, the `array_like` should contain the values organized in such a way, that the columns
            correspond to different attributes and the rows to different data points.
            In this case, the ordered iterable should contain the names of the attributes corresponding to the columns.

        Raises
        ------
        HandleException
            If the given `array_like` has a different number of columns than the number of given attributes.

        See also
        --------
        `Datahandler.handle`
        """
        if self.pand and isinstance(data, self.pd.DataFrame):
            data_dict = data.to_dict()
            data = Data().update(data_dict)
        else:
            arr, keys = data
            arr = np.asarray(arr)

            if len(keys) != arr.shape[1]:
                raise HandleException("Number of keys does not match data")

            data_dict = {key: arr[:, i] for i, key in enumerate(keys)}
            data = Data()
            data.update(data_dict)

        data.hrosailing_standard_format()

        statistics = get_datahandler_statistics(data)
        return data, statistics


class CsvFileHandler(DataHandler):
    """A data handler to extract data from a .csv file.

    The .csv file should be ordered in a column-wise fashion, with the
    first entry of each column describing the corresponding attribute.
    """

    # Check if pandas is available to use from_csv()-method
    try:
        __import__("pandas")
        pand = True
    except ImportError:
        pand = False

    if pand:
        import pandas as pd

    def handle(self, data):
        """
        Parameters
        ----------
        data : path_like
            Path to a .csv file.

        See also
        --------
        `Datahandler.handle`
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
                data = Data()
                data.update(data_dict)

        data.hrosailing_standard_format()

        statistics = get_datahandler_statistics(data_dict)
        return data_dict, statistics


class NMEAFileHandler(DataHandler):
    """A data handler to extract data from a text file containing
    certain NMEA sentences and convert it to a dictionary.

    Parameters
    ----------
    wanted_sentences : iterable of str, optional
        NMEA sentences that will be read.

        Defaults to `None`.

    unwanted_sentences : iterable of str, optional
        NMEA sentences that will be ignored.
        If `wanted_sentences` is set, this keyword is ignored.
        If both `wanted_sentences` and `unwanted_sentences` are `None`,
        all NMEA sentences will be read.

        Defaults to `None`.

    wanted_attributes : iterable of str, optional
        NMEA attributes or hrosailing standard keys that will be appended to the output.


        Defaults to `None`.

    unwanted_attributes : iterable of str, optional
        NMEA attributes that will be ignored.
        If `wanted_attributes` is set, this option is ignored.
        If both `wanted_attributes` and `unwanted_attributes` are `None`,
        all NMEA attributes will be read.

        Defaults to `None`.

    post_filter_types : tuple of types, optional
        The resulting dictionary only contains data which is `None` or of a
        type given in `post_filter_types`.
        If set to `False` all attributes are taken into account.

        Defaults to `(float, datetime.date, datetime.time, datetime.datetime)`.
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
            wanted_nmea_attributes = []
            for key in wanted_attributes:
                if key in HROSAILING_TO_NMEA:
                    wanted_nmea_attributes.extend(HROSAILING_TO_NMEA[key])
                else:
                    wanted_nmea_attributes.append(key)

            self._attribute_filter = lambda field: any(
                field[0] == attribute
                for attribute in wanted_nmea_attributes
                )
        elif unwanted_attributes is not None:
            self._attribute_filter = lambda field: all(
                field[0] != attribute
                for attribute in unwanted_attributes
                )
        else:
            self._attribute_filter = lambda field: True

        self._post_filter_types = post_filter_types

    def handle(self, data):
        """Reads a text file containing NMEA sentences and extracts
        data points.

        Parameters
        ----------
        data : path-like
            Path to a text file that contains NMEA 0183 sentences.

        See also
        --------
        `DataHandler.handle`
        """
        from pynmea2 import parse

        comp_data = Data()

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

                parsed_dict = {
                    name: getattr(parsed_sentence, attribute)
                    for name, attribute in wanted_fields
                }

                processed_dict = self._postprocess(parsed_dict)

                comp_data.update(processed_dict)

        comp_data.hrosailing_standard_format()

        if self._post_filter_types:
            comp_data.filter_types(self._post_filter_types)

        statistics = get_datahandler_statistics(comp_data)

        return comp_data, statistics

    def _postprocess(self, parsed_sentence):
        if "Wind angle" in parsed_sentence and "Reference" in parsed_sentence and "Wind speed" in parsed_sentence:
            wind_angle = parsed_sentence["Wind angle"]
            wind_speed = parsed_sentence["Wind speed"]
            reference = parsed_sentence["Reference"]
            if reference == "R":
                parsed_sentence["AWA"] = wind_angle
                parsed_sentence["AWS"] = wind_speed
            elif reference == "T":
                parsed_sentence["TWA"] = wind_angle
                parsed_sentence["TWS"] = wind_speed
            del parsed_sentence["Reference"]
            del parsed_sentence["Wind angle"]
            del parsed_sentence["Wind speed"]
        if "Latitude" in parsed_sentence:
            parsed_sentence["lat"] = self._from_nmea_format(
                parsed_sentence["Latitude"]
            )
            del parsed_sentence["Latitude"]
        if "Longitude" in parsed_sentence:
            parsed_sentence["lon"] = self._from_nmea_format(
                parsed_sentence["Longitude"]
            )
            del parsed_sentence["Longitude"]
        return parsed_sentence

    def _from_nmea_format(self, value):
        value = float(value)
        degrees = int(value/100)
        minutes = value - 100*degrees
        return degrees + minutes/60


def get_datahandler_statistics(data):
    """
    Computes standard statistics for the output of a data handler.

    Parameters
    ----------
    data : Data
    """
    return {
        "n_rows": data.n_rows,
        "n_cols": data.n_cols
    }
