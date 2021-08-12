"""

"""

# Author: Valentin F. Dannenberg / Ente

import csv
import numpy as np
import pynmea2

from abc import ABC, abstractmethod

from hrosailing.polardiagram import FileReadingException
from hrosailing.wind import apparent_wind_to_true


class HandlerException(Exception):
    pass


class DataHandler(ABC):
    @abstractmethod
    def handle(self, data):
        pass


class ArrayHandler(DataHandler):
    def handle(self, data):
        return data


class CsvFileHandler(DataHandler):
    def __init__(self, delimiter: str = None):
        self.delimiter = delimiter

    def handle(self, data):
        try:
            with open(data, "r", newline="") as file:
                csv_reader = csv.reader(file, delimiter=self.delimiter)
                return np.array(
                    [[eval(pt) for pt in row] for row in csv_reader]
                )
        except OSError:
            raise FileReadingException(f"Can't find/open/read {data}")


class NMEAFileHandler(DataHandler):
    def __init__(self, mode="interpolate", tw: bool = True):
        if mode not in {"mean", "interpolate"}:
            raise HandlerException(f"Mode {mode} not implemented")

        self.mode = mode

        if not isinstance(tw, bool):
            raise HandlerException("")

        self.tw = tw

    def handle(self, data: str):
        try:
            with open(data, "r") as nmea_file:
                nmea_data = []
                nmea_stcs = filter(
                    lambda line: "VHW" in line or "MWV" in line, nmea_file
                )

                stc = next(nmea_stcs, None)
                if stc is None:
                    raise FileReadingException(
                        "File didn't contain any relevant nmea sentences"
                    )

                while True:
                    try:
                        bsp = pynmea2.parse(stc).data[4]
                    except pynmea2.ParseError as pe:
                        raise FileReadingException(
                            f"During parsing of {stc}, the error {pe} occured"
                        )

                    stc = next(nmea_stcs, None)

                    if stc is None:
                        # eof
                        break

                    # check if nmea-file is in a
                    # way "sorted"
                    if "VHW" in stc:
                        raise FileReadingException(
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
            raise FileReadingException(f"Can't find/open/read {data}")


def _get_wind_data(wind_data, stc):
    try:
        wind = pynmea2.parse(stc)
    except pynmea2.ParseError:
        raise FileReadingException(
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
            raise FileReadingException(
                f"During parsing of {stc}, the error {pe} occured"
            )

        inter = len(wind_data)
        for i in range(inter):
            inter_bsp = ((inter - i) / inter) * bsp + (i / inter) * bsp2
            nmea_data.append(
                [wind_data[i][0], wind_data[i][1], inter_bsp, wind_data[i][2]]
            )
