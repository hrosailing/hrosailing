"""

"""

# Author: Valentin F. Dannenberg / Ente


import csv
from abc import ABC, abstractmethod

import numpy as np
import pynmea2

from hrosailing.wind import apparent_wind_to_true


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
    """"""

    def handle(self, data):
        return data


class CsvFileHandler(DataHandler):
    """A data handler to extract data from
    a .csv file with the following format

    Parameters
    ----------
    delimiter : str, optional
        Delimiter used in the .csv file
        If nothing is passed, the python parsing engine will try to
        autodetect the used delimiter, when reading a given .csv file


    Methods
    -------
    handle(self, data)

    """

    def __init__(self, delimiter: str = None):
        self.delimiter = delimiter

    def handle(self, data):
        """Reads a .csv file of data points and returns a numpy.ndarray
        of those data points
        Parameters
        ----------
        data : string
            Path to a .csv file which will be read
        Returns
        -------
        out : numpy.ndarray
            Array of the data points contained in the .csv file

        """
        try:
            with open(data, "r", newline="") as file:
                csv_reader = csv.reader(file, delimiter=self.delimiter)
                return np.array(
                    [[eval(pt) for pt in row] for row in csv_reader]
                )
        except OSError:
            raise HandlerException(f"Can't find/open/read {data}")


class NMEAFileHandler(DataHandler):
    """A data handler to extract data from a text file containing
    certain nmea sentences

    Parameters
    ---------
    mode : string, optional
        In the case where there is more recorded wind data than speed data,
        specifies how to handle the surplus
            - "interpolate": handles the surplus by taking
            convex combinations of two recorded speed datas
            together with the recorded wind data "between" those
            two points to create multiple data points
            - "mean": handles the surplus by taking the mean
            of the wind data "belonging" to a given speed data
            to create a singe data point
        Defaults to "interpolate"
    tw : bool, optional
        Specifies if occuring apparent wind should be automatically
        converted to true wind.
        If False, each point will have an extra component,
        specifying if it is true or apparent wind
        Defaults to True


    """

    def __init__(self, mode="interpolate", tw: bool = True):
        if mode not in {"mean", "interpolate"}:
            raise HandlerException(f"Mode {mode} not implemented")

        self.mode = mode

        if not isinstance(tw, bool):
            raise HandlerException("")

        self.tw = tw

    def handle(self, data: str):
        """Reads a text file containing nmea-sentences and extracts
        data points based on recorded wind speed, wind angle, and speed
        over water
        Function looks for sentences of type:
            - MWV for wind data
            - VHW for speed over water
        Parameters
        ----------
        data : string
            Path to a text file, containing nmea-0183 sentences, which will
            be read

        Returns
        -------
        nmea_data : list
            If tw:
                A list of points consisting of wind speed, wind angle,
                and boat speed, with the wind being true wind
            else:
                A list of points consisting of wind speed, wind angle,
                boat speed, and a reference, specifying wether the wind
                is true or appearent wind

        Raises a FileReadingException
            - if file can't be found, opened, or read
            - if file isn't "sorted", meaning there has to be at least
            one recorded wind data "between" two recorded speed datas
            - if file is empty or doesn't contain any relevant sentences
            - if file contains invalid relevant nmea sentences
        """
        try:
            with open(data, "r") as nmea_file:
                nmea_data = []
                nmea_stcs = filter(
                    lambda line: "VHW" in line or "MWV" in line, nmea_file
                )

                stc = next(nmea_stcs, None)
                if stc is None:
                    raise HandlerException(
                        "File didn't contain any relevant nmea sentences"
                    )

                while True:
                    try:
                        bsp = pynmea2.parse(stc).data[4]
                    except pynmea2.ParseError as pe:
                        raise HandlerException(
                            f"During parsing of {stc}, the error {pe} occured"
                        )

                    stc = next(nmea_stcs, None)

                    if stc is None:
                        # eof
                        break

                    # check if nmea-file is in a
                    # way "sorted"
                    if "VHW" in stc:
                        raise HandlerException(
                            "No recorded wind data in between recorded speed "
                            "data. Parsing not possible"
                        )

                    wind_data = []
                    while "VHW" not in stc and stc is not None:
                        _get_wind_data(wind_data, stc)
                        stc = next(nmea_stcs, None)

                    _process_data(nmea_data, wind_data, stc, bsp, self.mode)

                if self.tw:
                    aw = [data[:3] for data in nmea_data if data[3] == "R"]
                    tw = [data[:3] for data in nmea_data if data[3] != "R"]
                    if not aw:
                        return tw

                    aw = apparent_wind_to_true(aw)
                    return tw.extend(aw)

                return nmea_data

        except OSError:
            raise HandlerException(f"Can't find/open/read {data}")


def _get_wind_data(wind_data, stc):
    try:
        wind = pynmea2.parse(stc)
    except pynmea2.ParseError:
        raise HandlerException(
            f"Invalid nmea-sentences encountered: {stc}"
        )

    wind_data.append(
        [float(wind.wind_speed), float(wind.wind_angle), wind.reference]
    )


def _process_data(nmea_data, wind_data, stc, bsp, mode):
    if mode == "mean":
        wind_arr = np.array([w[:2] for w in wind_data])
        wind_arr = np.mean(wind_arr, axis=0)
        nmea_data.append([wind_arr[0], wind_arr[1], bsp, wind_data[0][2]])

    if mode == "interpolate":
        try:
            bsp2 = pynmea2.parse(stc).data[4]
        except pynmea2.ParseError as pe:
            raise HandlerException(
                f"During parsing of {stc}, the error {pe} occured"
            )

        inter = len(wind_data)
        for i in range(inter):
            inter_bsp = ((inter - i) / inter) * bsp + (i / inter) * bsp2
            nmea_data.append(
                [wind_data[i][0], wind_data[i][1], inter_bsp, wind_data[i][2]]
            )
