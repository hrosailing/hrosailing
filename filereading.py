"""
Functions to read csv files with various formats and files containing
nmea sentences
"""

# Author: Valentin F. Dannenberg / Ente

import csv
import logging
import logging.handlers
import numpy as np
import pynmea2

from exceptions import FileReadingException


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='filereading.log')
LOG_FILE = "filereading.log"

logger = logging.getLogger(__name__)
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when='midnight')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


def read_table(csv_reader):
    logger.info(f"""Function 'read_table(
                 csv_reader={csv_reader.__name__})' called""")

    next(csv_reader)
    ws_res = [eval(ws) for ws in next(csv_reader)]
    next(csv_reader)
    wa_res = [eval(wa) for wa in next(csv_reader)]
    next(csv_reader)
    data = []
    for row in csv_reader:
        data.append([eval(bsp) for bsp in row])

    return ws_res, wa_res, data


def read_pointcloud(csv_reader):
    logger.info(f"""Function 'read_pointcloud(
                 csv_reader={csv_reader.__name__})' called""")
    data = []
    next(csv_reader)
    for row in csv_reader:
        data.append([eval(entry) for entry in row])

    return np.array(data)


def read_extern_format(csv_path, fmt):
    logger.info(f"""Function 'read_extern_format(csv_path={csv_path}, 
                 fmt={fmt})' called""")

    if fmt == 'array':
        return read_array_csv(csv_path)

    if fmt == 'orc':
        delimiter = ';'
    else:
        delimiter = ','

    return read_sail_csv(csv_path, delimiter)


def read_sail_csv(csv_path, delimiter):
    logger.info(f"""Function 'read_sail_csv(csv_path={csv_path},
                 delimiter={delimiter})' called""")

    try:
        with open(csv_path, 'r', newline='') as file:
            csv_reader = csv.reader(file, delimiter=delimiter,
                                    quotechar='"')
            ws_res = [eval(ws) for ws in next(csv_reader)[1:]]
            wa_res, data = [], []
            next(csv_reader)
            for row in csv_reader:
                wa_res.append(eval(row[0]))
                data.append([eval(bsp) if bsp != '' else 0 for bsp in row[1:]])

            return ws_res, wa_res, data
    except OSError:
        logger.error(f"Error occured when accessing file {csv_path}")
        raise FileReadingException(f"can't find or open {csv_path}")


def read_array_csv(csv_path):
    logger.info(f"Function 'read_array_csv(csv_path={csv_path}' called")

    try:
        file_data = np.genfromtxt(csv_path, delimiter="\t")
        return file_data[0, 1:], file_data[1:, 0], file_data[1:, 1:]
    except OSError:
        logger.error(f"Error occured when accessing file {csv_path}")
        raise FileReadingException(f"can't find or open {csv_path}")


def read_nmea_file(nmea_path, mode='interpolate'):
    logger.info(f"""Function 'read_nmea_file(nmea_path={nmea_path},
                 mode={mode})' called""")

    if mode not in ('mean', 'interpolate'):
        logger.error(f"Error occured when requesting mode")
        raise FileReadingException(
            f"mode {mode} not implemented")

    with open(nmea_path, 'r') as nmea_file:
        nmea_data = []
        nmea_sentences = filter(
            lambda line: "RMC" in line
                         or "MWV" in line,
            nmea_file)

        stc = next(nmea_sentences, None)
        if stc is None:
            logger.error(f"Error occured when trying to read {stc}")
            raise FileReadingException(
                """nmea-file didn't contain
                any necessary data""")

        while True:
            try:
                bsp = pynmea2.parse(stc).spd_over_grnd
            except pynmea2.ParseError:
                logger.error(f"Error occured when parsing {stc}")
                raise FileReadingException

            stc = next(nmea_sentences, None)

            if stc is None:
                # eof
                break
            # check if nmea-file is in a
            # way "sorted"
            if "RMC" in stc:
                logger.error(f"Error occured when trying to read {stc}")
                raise FileReadingException(
                    """nmea-file has two GPRMC
                    sentences with no wind data
                    in between them.""")

            wind_data = []
            while "RMC" not in stc and stc is not None:
                _get_wind_data(wind_data, stc)
                stc = next(nmea_sentences, None)

            _process_data(
                nmea_data, wind_data, stc,
                bsp, mode)

    return nmea_data


def _get_wind_data(wind_data, nmea_sentence):
    # logger.info(f"""Function _get_wind_data(wind_data,
    #              nmea_sentence)' called""")

    try:
        wind = pynmea2.parse(nmea_sentence)
    except pynmea2.ParseError:
        logger.error("Error occured when parsing nmea_sentence")
        raise FileReadingException()

    wind_data.append(
        [float(wind.wind_speed),
         float(wind.wind_angle),
         wind.reference])


def _process_data(nmea_data, wind_data, nmea_sentence, bsp, mode):
    # logger.info(f"""Function '_process_data(nmea_data,
    #              wind_data, nmea_sentence, bsp,
    #              mode={mode})' called""")

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
            logger.error("Error occured when parsing nmea_sentence")
            raise FileReadingException()

        inter = len(wind_data)
        for i in range(inter):
            inter_bsp = ((inter - i) / inter * bsp
                         + i / inter * bsp2)
            nmea_data.append(
                [wind_data[i][0],
                 wind_data[i][1],
                 inter_bsp,
                 wind_data[i][2]])
