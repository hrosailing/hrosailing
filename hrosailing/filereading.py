"""
Defines various functions to read .csv files with
different formats, aswell as files containing nmea sentences
"""

# Author: Valentin F. Dannenberg / Ente


import csv
import logging.handlers
import numpy as np
import pynmea2

from hrosailing.exceptions import FileReadingException
from hrosailing.windconversion import apparent_wind_to_true


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='hrosailing/logging/filereading.log')
LOG_FILE = "hrosailing/logging/filereading.log"

logger = logging.getLogger(__name__)
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when='midnight')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


def read_csv_file(csv_path, delimiter=None):
    """Reads a .csv file
    of data points and
    returns a numpy.ndarray
    of those data points

    Parameters
    ----------
    csv_path : string
        Path to a .csv file
        which will be read
    delimiter : string, optional
        Delimiter used in
        the .csv file

        If nothing is passed,
        the python parsing
        engine autodetects the
        used delimiter

    Returns
    -------
    csv_data : numpy.ndarray
        Array of data points
        contained in the
        .csv file
    """

    try:
        with open(csv_path, 'r', newline='') as file:
            csv_reader = csv.reader(
                file, delimiter=delimiter)
            return read_pointcloud(csv_reader)
    except OSError:
        raise FileReadingException(
            f"Can't find/open/read {csv_path}")


def read_nmea_file(nmea_path, mode='interpolate',
                   convert_wind=True):
    """Reads a text file
    containing nmea-sentences
    and extracts data points
    based on recorded wind speed,
    wind angle, and either speed
    over water or speed over
    ground and returns
    a list of said points

    Function looks for
    sentences of type:
        MWV for wind data
        either VHW for
        speed over water or
        if not present
        RMC for speed over
        ground

    Parameters
    ----------
    nmea_path : string
        Path to a text file,
        containing nmea-0183
        sentences, which will
        be read
    mode : string, optional
        In the case where there is
        more recorded wind data
        than speed data,
        specifies how to handle
        the surplus:
            interpolate: handles the
            surplus by taking
            convex combinations of
            two recorded speed datas
            together with the recorded
            wind data "between" those
            two points to create
            multiple data points

            mean: handles the
            surplus by taking the
            mean of the wind data
            "belonging" to a given
            speed data to create a
            singe data point

        Defaults to 'interpolate'
    convert_wind : bool, optional
        Specifies if occuring
        apparent wind should
        be automatically converted
        to true wind.

        If False, each point will
        have an extra component,
        specifying if it is true
        or apparent wind

        Defaults to True

    Returns
    -------
    nmea_data : list
        If convert_wind is True:
            A list of points
            consisting of wind
            speed, wind angle,
            and boat speed,
            with the wind being
            true wind
        If convert_wind is False:
            A list of points
            consisting of wind
            speed, wind angle,
            boat speed, and
            a reference, specifying
            wether the wind is
            true or appearent wind


    Function raises an exception:
        If file can't be found,
        opened, or read

        If file isn't "sorted",
        meaning there has to be
        at least one recorded
        wind data between two
        recorded speed datas

        If file is empty or
        doesn't contain any
        relevant sentences

        If file contains invalid
        speed or wind sentences
    """

    # TODO: Also look for rmc sentences, if
    #       there are no vhws
    if mode not in ('mean', 'interpolate'):
        raise FileReadingException(
            f"Mode {mode} not implemented")

    with open(nmea_path, 'r') as nmea_file:
        nmea_data = []
        nmea_stcs = filter(
            lambda line: "VHW" in line
                         or "MWV" in line,
            nmea_file)

        stc = next(nmea_stcs, None)
        if stc is None:
            raise FileReadingException(
                "File didn't contain any "
                "relevant nmea sentences")

        while True:
            try:
                bsp = pynmea2.parse(stc).spd_over_water
            except pynmea2.ParseError:
                raise FileReadingException(
                    f"Invalid nmea-sentences "
                    f"encountered: {stc}")

            stc = next(nmea_stcs, None)

            if stc is None:
                # eof
                break
            # check if nmea-file is in a
            # way "sorted"
            if "VHW" in stc:
                raise FileReadingException("")

            wind_data = []
            while "VHW" not in stc and stc is not None:
                _get_wind_data(wind_data, stc)
                stc = next(nmea_stcs, None)

            _process_data(
                nmea_data, wind_data, stc,
                bsp, mode)

        if convert_wind:
            aw = [data[:3] for data in nmea_data
                  if data[3] == 'R']
            tw = [data[:3] for data in nmea_data
                  if data[3] != 'R']
            if not aw:
                return tw

            aw = apparent_wind_to_true(aw)
            return tw.extend(aw)

        return nmea_data


def _get_wind_data(wind_data, stc):
    try:
        wind = pynmea2.parse(stc)
    except pynmea2.ParseError:
        raise FileReadingException(
            f"Invalid nmea-sentences "
            f"encountered: {stc}")

    wind_data.append(
        [float(wind.wind_speed),
         float(wind.wind_angle),
         wind.reference])


def _process_data(nmea_data, wind_data,
                  stc, bsp, mode):
    if mode == 'mean':
        wind_arr = np.array(
            [w[:2] for w in wind_data])
        wind_arr = np.mean(wind_arr, axis=0)
        nmea_data.append(
            [wind_arr[0],
             wind_arr[1],
             bsp,
             wind_data[0][2]])

    if mode == 'interpolate':
        try:
            bsp2 = pynmea2.parse(stc).spd_over_grnd
        except pynmea2.ParseError:
            raise FileReadingException(
                f"Invalid nmea-sentences "
                f"encountered: {stc}")

        inter = len(wind_data)
        for i in range(inter):
            inter_bsp = ((inter - i) / inter * bsp
                         + i / inter * bsp2)
            nmea_data.append(
                [wind_data[i][0],
                 wind_data[i][1],
                 inter_bsp,
                 wind_data[i][2]])


def read_table(csv_reader):
    next(csv_reader)
    ws_res = [eval(ws) for ws in next(csv_reader)]
    next(csv_reader)
    wa_res = [eval(wa) for wa in next(csv_reader)]
    next(csv_reader)
    bsps = []
    for row in csv_reader:
        bsps.append([eval(bsp) for bsp in row])

    return ws_res, wa_res, bsps


def read_pointcloud(csv_reader):
    points = []
    next(csv_reader)
    for row in csv_reader:
        points.append([eval(entry) for entry in row])

    return np.array(points)


def read_extern_format(csv_path, fmt):
    if fmt == 'array':
        return read_array_csv(csv_path)

    if fmt == 'orc':
        delimiter = ';'
    else:
        delimiter = ','

    return read_sail_csv(csv_path, delimiter)


def read_sail_csv(csv_path, delimiter):
    try:
        with open(csv_path, 'r', newline='') as file:
            csv_reader = csv.reader(
                file, delimiter=delimiter)
            ws_res = [eval(ws) for ws
                      in next(csv_reader)[1:]]
            wa_res, bsps = [], []
            next(csv_reader)
            for row in csv_reader:
                wa_res.append(eval(row[0]))
                bsps.append([eval(bsp) if bsp != ''
                             else 0 for bsp in row[1:]])

            return ws_res, wa_res, bsps
    except OSError:
        raise FileReadingException(
            f"Can't find/open/read {csv_path}")


def read_array_csv(csv_path):
    try:
        file_data = np.genfromtxt(
            csv_path, delimiter="\t")
        return (file_data[0, 1:],
                file_data[1:, 0],
                file_data[1:, 1:])
    except OSError:
        raise FileReadingException(
            f"Can't find/open/read {csv_path}")
