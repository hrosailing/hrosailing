"""

"""

# Author: Valentin Dannenberg

# pylint: disable=import-outside-toplevel

from abc import ABC, abstractmethod
from ast import literal_eval
import csv
from decimal import Decimal
from typing import Union

import numpy as np
import pynmea2 as pynmea

from hrosailing.wind import _convert_wind


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

    def handle(self, data: Union[(np.ndarray, tuple), pd.DataFrame]) -> dict:
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
        """
        if self.pand and isinstance(data, self.pd.DataFrame):
            return data.to_dict()

        arr, keys = data
        arr = np.asarray(arr)

        return {key : arr[:, i] for i, key in enumerate(keys)}


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
        data : path-like
            Path to a .csv file

        Returns
        -------
        data_dict : dict
            Dictionary having the first row entries as keys and
            as values the corresponding columns given as lists
        """
        if self.pand:
            df = self.pd.read_csv(data)
            return df.to_dict()

        with open(data, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            keys = next(csv_reader)
            data_dict = {key : [] for key in keys}
            for row in csv_reader:
                for i, entry in enumerate(row):
                    data_dict[keys[i]].append(literal_eval(entry))

        return data_dict


class NMEAFileHandler(DataHandler):
    """A data handler to extract data from a text file containing
    certain nmea sentences and convert it to a dictionary

    Parameters
    ---------
    sentences : Iterable of Strings,

    attributes : Iterable of Strings,

    mode : string, optional
        In the case where there is more recorded wind data than speed data,
        specifies how to handle the surplus

        - "interpolate": handles the surplus by taking convex combinations
        of two recorded speed datas together with the recorded wind data
        "between" those two points to create multiple data points
        - "mean": handles the surplus by taking the mean of the wind data
        "belonging" to a given speed data to create a singe data point

        Defaults to "interpolate"

    Raises a HandlerInitializationException if mode is not one of
    the above choices
    """

    def __init__(self, sentences, attributes, mode="interpolate"):
        if mode not in {"mean", "interpolate"}:
            raise HandlerInitializationException("`mode` not implemented")

        sentences = list(sentences)
        attributes = list(attributes)

        if "MWV" not in sentences:
            sentences.append("MWV")

        if "Wind speed" not in attributes:
            attributes.append("Wind speed")
        elif "Wind angle" not in attributes:
            attributes.append("Wind angle")
        elif "Reference" not in attributes:
            attributes.append("Reference")

        self._nmea_filter = sentences
        self._attr_filter = attributes
        self.mode = mode

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
        """
        data_dict = {attr: [] for attr in self._attr_filter}
        ndata = 0

        with open(data, "r", encoding="utf-8") as file:
            nmea_stcs = filter(
                lambda line: any(abbr in line for abbr in self._nmea_filter),
                file,
            )

            for stc in nmea_stcs:
                parsed = pynmea.parse(stc)
                nmea_attr = filter(
                    lambda pair: any(
                        attr == pair[0][0] for attr in self._attr_filter
                    ),
                    zip(parsed.fields, parsed.data),
                )

                for field, val in nmea_attr:
                    name = field[0]
                    len_ = len(data_dict[name])
                    if len_ == ndata:
                        data_dict[name].append(
                            literal_eval(val)
                            if len(field) == 3
                            and field[2] in {int, float, Decimal}
                            else val
                        )
                        ndata += 1
                    else:
                        data_dict[name].extend(
                            [None] * (ndata - len_ - 1)
                            + [
                                literal_eval(val)
                                if len(field) == 3
                                and field[2] in {int, float, Decimal}
                                else val
                            ]
                        )

                for attr in self._attr_filter:
                    len_ = len(data_dict[attr])
                    data_dict[attr].extend([None] * (ndata - len_ - 1))

        # wa = data_dict.get("Wind Angle")
        # ws = data_dict.get("Wind Speed")
        # bsp = data_dict.get("Speed over ground knots") or data_dict.get("Speed over water knots")
        # ref = data_dict.get("Reference")
        #
        # aw = [[s, a, b] for s, a, b, r in zip(ws, wa, bsp, ref) if all([s, a, b, r]) and r == "R"]
        # if aw:
        #     cw = _convert_wind(aw, -1, False, _check_finite=True)
        #
        # ws, wa = zip(*[w[:2] for w in cw if r else [None, None]])
        #
        # data_dict["Wind Speed"] = list(x)
        # data_dict["Wind Angle"] = list(y)

        return data_dict


def _handle_surplus_data(data_dict, mode):
    pass


def _get_wind_data(wind_data, stc):
    wind = pynmea.parse(stc)
    wind_data.append(
        [float(wind.wind_speed), float(wind.wind_angle), wind.reference]
    )


def _process_data(nmea_data, wind_data, stc, bsp, mode):
    if mode == "mean":
        wind = np.array([w[:2] for w in wind_data])
        wind = np.mean(wind, axis=0)
        nmea_data.append([wind[0], wind[1], bsp, wind_data[0][2]])
    elif mode == "interpolate":
        bsp2 = float(pynmea.parse(stc).data[4])

        inter = len(wind_data)
        for i in range(inter):
            inter_bsp = ((inter - i) / inter) * bsp + (i / inter) * bsp2
            nmea_data.append(
                [wind_data[i][0], wind_data[i][1], inter_bsp, wind_data[i][2]]
            )
