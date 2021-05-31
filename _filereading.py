import numpy as np
import csv
import pynmea2

from _exceptions import FileReadingException


def read_table(csv_reader):
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
    data = []
    next(csv_reader)
    for row in csv_reader:
        data.append([eval(entry) for entry in row])

    return np.array(data)


def read_extern_format(csv_path, fmt):
    if fmt == 'array':
        return read_array_csv(csv_path)

    if fmt == 'orc':
        delimiter = ';'
    else:
        delimiter = ','

    return read_sail_csv(csv_path, delimiter)


def read_sail_csv(csv_path, delimiter):
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


def read_array_csv(csv_path):
    file_data = np.genfromtxt(csv_path, delimiter="\t")
    return file_data[0, 1:], file_data[1:, 0], file_data[1:, 1:]


def read_nmea_file(nmea_path, mode='interpolate'):
    nmea_data = []

    if mode not in ('mean', 'interpolate'):
        raise FileReadingException(
            f"mode {mode} not implemented"
        )

    with open(nmea_path, 'r') as nmea_file:
        nmea_sentences = filter(
            lambda line: "RMC" in line
                         or "MWV" in line,
            nmea_file
        )

        stc = next(nmea_sentences, None)
        if stc is None:
            raise FileReadingException(
                """nmea-file didn't contain 
                any necessary data"""
            )

        while True:
            bsp = pynmea2.parse(stc).spd_over_grnd
            stc = next(nmea_sentences, None)

            if stc is None:
                # eof
                break
            # check if nmea-file is in some
            # form sorted
            if "GPRMC" in stc:
                raise FileReadingException(
                    """nmea-file has two GPRMC 
                    sentences with no wind data 
                    in between them."""
                )

            wind_data = []
            while "$GPRMC" not in stc and stc is not None:
                _get_wind_data(wind_data, stc)
                stc = next(nmea_sentences, None)

            _process_data(
                nmea_data, wind_data, stc,
                bsp, mode)

    return nmea_data


def _get_wind_data(wind_data, nmea_sentence):
    wind = pynmea2.parse(nmea_sentence)
    wind_data.append(
        [float(wind.wind_speed),
         float(wind.wind_angle),
         wind.reference])


def _process_data(nmea_data, wind_data, nmea_sentence, bsp, mode):
    if mode == 'mean':
        wind_arr = np.array(
            [w[:2] for w in wind_data])
        wind_arr = np.mean(wind_arr, axis=0)
        nmea_data.append(
            [wind_arr[0],
             wind_arr[1],
             bsp,
             wind_data[2]])

    if mode == 'interpolate':
        bsp2 = pynmea2.parse(
            nmea_sentence).spd_over_grnd
        inter = len(wind_data)
        for i in range(inter):
            inter_bsp = ((inter - i) / inter * bsp
                         + i / inter * bsp2)
            nmea_data.append(
                [wind_data[i][0],
                 wind_data[i][1],
                 inter_bsp,
                 wind_data[i][2]])
