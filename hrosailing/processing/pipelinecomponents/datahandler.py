"""

"""

# Author: Valentin F. Dannenberg / Ente


import csv
from abc import ABC, abstractmethod

import numpy as np
import pynmea2

from hrosailing.wind import apparent_wind_to_true, WindException


class HandlerException(Exception):
    """Custom exception for errors that may appear whilst
    working with the DataHandler class and subclasses
    """

    pass


class DataHandler(ABC):
    """Base class for all datahandler classes


    Abstract Methods
    ----------------
    handle(self, data)
    """

    @abstractmethod
    def handle(self, data):
        pass


class ArrayHandler(DataHandler):
    """A data handler to handle data, given
    as an array_like sequence.

    Doesn't really do anything since error handling
    and array conversion is handeled by the pipeline itself

    Only needed for general layout of the pipeline
    """

    @staticmethod
    def handle(data):
        return data


# TODO Error handling
class CsvFileHandler(DataHandler):
    """A data handler to extract data from a .csv file
    with the first three columns representing wind speed, wind angle,
    and boat speed respectively
    """

    @staticmethod
    def handle(data, **pandas_kw):
        """Reads a .csv file and extracts the contained data points
        The delimiter used in the .csv file

        Parameters
        ----------
        data : path-like
            Path to a .csv file

        pandas_kw :

        Returns
        -------
        out : pandas.Dataframe


        Raises a HandlerException
        """
        from pandas import read_csv
        return read_csv(data, **pandas_kw)


class NMEAFileHandler(DataHandler):
    """A data handler to extract data from a text file containing
    certain nmea sentences

    Parameters
    ---------
    mode : string, optional
        In the case where there is more recorded wind data than speed data,
        specifies how to handle the surplus

        - "interpolate": handles the surplus by taking convex combinations
        of two recorded speed datas together with the recorded wind data
        "between" those two points to create multiple data points
        - "mean": handles the surplus by taking the mean of the wind data
        "belonging" to a given speed data to create a singe data point

        Defaults to "interpolate"

    Raises a HandlerException if
    """

    def __init__(self, mode="interpolate"):
        if mode not in {"mean", "interpolate"}:
            raise HandlerException("`mode` not implemented")

        self.mode = mode

    def handle(self, data):
        """Reads a text file containing nmea-sentences and extracts
        data points based on recorded wind speed, wind angle, and speed
        over water

        Function looks for sentences of type:

        - MWV for wind data
        - VHW for speed trough water

        Parameters
        ----------
        data : path-like
            Path to a text file, containing nmea-0183 sentences

        Returns
        -------
        out : list of lists of length 3


        Raises a HandlerException

        - if `data` doesn't contain relevant nmea senteces
        - if nmea senteces are not sorted
        - if an error occurs whilst reading
        - if an error occurs whilst parsing of the nmea senteces
        - if an error occurs during conversion of apperant wind
        """
        try:
            with open(data, "r") as file:
                nmea_stcs = filter(
                    lambda line: "VHW" in line or "MWV" in line, file
                )

                stc = next(nmea_stcs, None)
                # check if file is "empty"
                if stc is None:
                    raise HandlerException(
                        "`data` doesn't contain any (relevant) nmea sentences"
                    )

                nmea_data = []
                while True:
                    bsp = float(pynmea2.parse(stc).data[4])
                    stc = next(nmea_stcs, None)

                    # eof
                    if stc is None:
                        break

                    # check if nmea sentences is "sorted"
                    if "VHW" in stc:
                        raise HandlerException(
                            "No wind records in between speed records. "
                            "Parsing not possible"
                        )

                    wind_data = []
                    # record wind data until next VHW sentence
                    while stc is not None and "VHW" not in stc:
                        _get_wind_data(wind_data, stc)
                        stc = next(nmea_stcs, None)

                    _process_data(nmea_data, wind_data, stc, bsp, self.mode)

                aw = [data[:3] for data in nmea_data if data[3] == "R"]
                tw = [data[:3] for data in nmea_data if data[3] != "R"]

                # if no apparent wind occurs, return true wind as is
                if not aw:
                    return tw

                aw = apparent_wind_to_true(aw)
                return tw.extend(aw)
        except OSError as oe:
            raise HandlerException(
                "While reading `data` an error occured"
            ) from oe
        except pynmea2.ParseError as pe:
            raise HandlerException(
                f"During parsing of {stc}, an error occured"
            ) from pe
        except WindException as we:
            raise HandlerException(
                "While converting wind, an error occured"
            ) from we


def _get_wind_data(wind_data, stc):
    wind = pynmea2.parse(stc)
    wind_data.append(
        [float(wind.wind_speed), float(wind.wind_angle), wind.reference]
    )


def _process_data(nmea_data, wind_data, stc, bsp, mode):
    if mode == "mean":
        wind = np.array([w[:2] for w in wind_data])
        wind = np.mean(wind, axis=0)
        nmea_data.append([wind[0], wind[1], bsp, wind_data[0][2]])
    elif mode == "interpolate":
        bsp2 = float(pynmea2.parse(stc).data[4])

        inter = len(wind_data)
        for i in range(inter):
            inter_bsp = ((inter - i) / inter) * bsp + (i / inter) * bsp2
            nmea_data.append(
                [wind_data[i][0], wind_data[i][1], inter_bsp, wind_data[i][2]]
            )
