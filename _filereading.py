import numpy as np
import csv


# V: In Arbeit
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


# V: In Arbeit
def read_pointcloud(csv_reader):
    data = []
    next(csv_reader)
    for row in csv_reader:
        data.append([eval(entry) for entry in row])

    return np.array(data)


# V: In Arbeit
def read_extern_format(csv_path, fmt):
    if fmt == 'array':
        return read_array_csv(csv_path)

    if fmt == 'orc':
        delimiter = ';'
    if fmt == 'opencpn':
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


# V: In Arbeit
def read_array_csv(csv_path):
    file_data = np.genfromtxt(csv_path, delimiter="\t")
    return file_data[0, 1:], file_data[1:, 0], file_data[1:, 1:]
