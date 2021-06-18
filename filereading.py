"""
Defines various functions to read .csv files with
different formats, aswell as files containing nmea sentences
"""

# Author: Valentin F. Dannenberg / Ente


import csv
import logging.handlers
import numpy as np
import pynmea2

from exceptions import FileReadingException


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='logging/filereading.log')
LOG_FILE = "logging/filereading.log"

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


# TODO: Remove refence from nmea_data and
#       instead return the list of points
#       and a bool for wether its true wind
#       or not

# TODO: Make it cleaner!
def read_nmea_file(nmea_path, mode='interpolate'):
    """Reads a text file
    containing nmea-sentences
    and returns the data
    points containe in it

    Parameters
    ----------
    nmea_path : string
        Path to a text file,
        containing nmea-0183
        sentences, which will
        be read
    mode : string, optional
        In the case there are more
        MWV sentences than
        RMC sentences, specifies
        how to handle the
        surplus of data
            interpolate: handles the
            surplus by taking
            convex combinations of
            the speed over ground of
            two RMC sentences together
            with the wind speed and
            wind angle of the MWV
            sentences "between" them
            to create multiple
            data points

            mean: handles the
            surplus by taking mean
            of the wind angle and
            wind speed of the MWV
            sentences "belonging"
            to an RMC sentence to
            create a single data
            point

    Returns
    -------
    nmea_data : list
        List of lists of length 4
        containing data point given
        by wind speed, wind angle,
        boat speed, and reference,
        where reference specifies
        if the wind data is true
        wind or not


    Function raises an exception:
        If file can't be read or
        doesnt exist

        If file isn't "sorted",
        meaning there has to be
        at least one MWV sentence
        between two RMC sentences

        If file is empty

        If file contains invalid
        RMC or WMV sentences
    """

    if mode not in ('mean', 'interpolate'):
        raise FileReadingException(
            f"Mode {mode} not implemented")

    with open(nmea_path, 'r') as nmea_file:
        nmea_data = []
        nmea_sentences = filter(
            lambda line: "RMC" in line
                         or "MWV" in line,
            nmea_file)

        stc = next(nmea_sentences, None)
        if stc is None:
            raise FileReadingException(
                "File didn't contain any "
                "relevant nmea sentences")

        while True:
            try:
                bsp = pynmea2.parse(stc).spd_over_grnd
            except pynmea2.ParseError:
                raise FileReadingException(
                    f"Invalid nmea-sentences "
                    f"encountered: {stc}")

            stc = next(nmea_sentences, None)

            if stc is None:
                # eof
                break
            # check if nmea-file is in a
            # way "sorted"
            if "RMC" in stc:
                raise FileReadingException("")

            wind_data = []
            while "RMC" not in stc and stc is not None:
                _get_wind_data(wind_data, stc)
                stc = next(nmea_sentences, None)

            _process_data(
                nmea_data, wind_data, stc,
                bsp, mode)

    return nmea_data


def _get_wind_data(wind_data, nmea_sentence):
    try:
        wind = pynmea2.parse(nmea_sentence)
    except pynmea2.ParseError:
        raise FileReadingException(
            f"Invalid nmea-sentences "
            f"encountered: {nmea_sentence}")

    wind_data.append(
        [float(wind.wind_speed),
         float(wind.wind_angle),
         wind.reference])


def _process_data(nmea_data, wind_data,
                  nmea_sentence, bsp, mode):
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
            bsp2 = pynmea2.parse(nmea_sentence).spd_over_grnd
        except pynmea2.ParseError:
            raise FileReadingException(
                f"Invalid nmea-sentences "
                f"encountered: {nmea_sentence}")

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
