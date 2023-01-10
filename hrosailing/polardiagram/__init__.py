"""
`PolarDiagram` classes to work with and represent polar diagrams in various
forms.
"""


import csv
from ast import literal_eval

import numpy as np

from ._basepolardiagram import (
    PolarDiagram,
    PolarDiagramException,
    PolarDiagramInitializationException,
)
from ._polardiagramcurve import PolarDiagramCurve
from ._polardiagrammultisails import PolarDiagramMultiSails
from ._polardiagrampointcloud import PolarDiagramPointcloud
from ._polardiagramtable import PolarDiagramTable

__all__ = [
    "PolarDiagram",
    "PolarDiagramCurve",
    "PolarDiagramMultiSails",
    "PolarDiagramPointcloud",
    "PolarDiagramTable",
    "from_csv",
    "FileReadingException",
    "PolarDiagramException",
    "PolarDiagramInitializationException",
]


class FileReadingException(Exception):
    """Exception raised if non-OS-error occurs,
    when reading a file.
    """


def from_csv(csv_path, fmt="hro"):
    """Reads a .csv file and returns the `PolarDiagram`
    instance contained in it.

    Parameters
    ----------
    csv_path : path-like
        Path to a .csv file.

    fmt : str
        The format of the .csv file.

        - `hro` : format created by the `to_csv`-method of the `PolarDiagram` class.
        - `orc` : format found at [ORC](https://\
            jieter.github.io/orc-data/site/).
        - `opencpn` : format created by the [OpenCPN Polar Plugin](https://\
            opencpn.org/OpenCPN/plugins/polar.html).
        - `array` : tab-separated polar diagram in form of a table, also
            see the example files for a better look at the format.

    Returns
    -------
    out : PolarDiagram
        `PolarDiagram` instance contained in the .csv file.

    Raises
    ------
    FileReadingException
        If an unknown format was specified.
    FileReadingException
        If, in the format `hro`, the first row does not match any
        `PolarDiagram` subclass.

    OSError
        If file does not exist or no read permission for that file is given.

    Examples
    --------
    (For the following and more files also see
    [examples](https://github.com/hrosailing/hrosailing/tree/main/examples))

        >>> from hrosailing.polardiagram import from_csv
        >>> pd = from_csv("table_hro_format_example.csv", fmt="hro")
        >>> print(pd)
          TWA / TWS    6.0    8.0    10.0    12.0    14.0    16.0    20.0
        -----------  -----  -----  ------  ------  ------  ------  ------
        52.0          3.74   4.48    4.96    5.27    5.47    5.66    5.81
        60.0          3.98   4.73    5.18    5.44    5.67    5.94    6.17
        75.0          4.16   4.93    5.35    5.66    5.95    6.27    6.86
        90.0          4.35   5.19    5.64    6.09    6.49    6.70    7.35
        110.0         4.39   5.22    5.68    6.19    6.79    7.48    8.76
        120.0         4.23   5.11    5.58    6.06    6.62    7.32    9.74
        135.0         3.72   4.64    5.33    5.74    6.22    6.77    8.34
        150.0         3.21   4.10    4.87    5.40    5.78    6.22    7.32
    """
    if fmt not in {"array", "hro", "opencpn", "orc"}:
        raise FileReadingException("`fmt` unknown")

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        if fmt == "hro":
            return _read_intern_format(file)

        return _read_extern_format(file, fmt)


def _read_intern_format(file):
    subclasses = {cls.__name__: cls for cls in PolarDiagram.__subclasses__()}

    first_row = file.readline().rstrip()
    if first_row not in subclasses:
        raise FileReadingException(
            f"no polar diagram format with the name {first_row} exists"
        )

    pd = subclasses[first_row]
    return pd.__from_csv__(file)


def _read_extern_format(file, fmt):
    if fmt == "array":
        ws_res, wa_res, bsps = _read_from_array(file)
    elif fmt == "orc":
        ws_res, wa_res, bsps = _read_orc_format(file)
    else:
        ws_res, wa_res, bsps = _read_opencpn_format(file)

    return PolarDiagramTable(ws_res, wa_res, bsps)


def _read_from_array(file):
    file_data = np.genfromtxt(file)
    return file_data[0, 1:], file_data[1:, 0], file_data[1:, 1:]


def _read_orc_format(file):
    csv_reader = csv.reader(file, delimiter=";")

    ws_res = _read_wind_speeds(csv_reader)

    # skip line of zeros
    next(csv_reader)

    wa_res, bsps = _read_wind_angles_and_boat_speeds(csv_reader)

    return ws_res, wa_res, bsps


def _read_wind_speeds(csv_reader):
    return [literal_eval(ws) for ws in next(csv_reader)[1:]]


def _read_wind_angles_and_boat_speeds(csv_reader):
    wa_res = []
    bsps = []

    for row in csv_reader:
        wa_res.append(literal_eval(row[0].replace("°", "")))
        bsps.append([literal_eval(bsp) if bsp != "" else 0 for bsp in row[1:]])

    return wa_res, bsps


def _read_opencpn_format(file):
    csv_reader = csv.reader(file, delimiter=",")

    ws_res = _read_wind_speeds(csv_reader)
    wa_res, bsps = _read_wind_angles_and_boat_speeds(csv_reader)

    return ws_res, wa_res, bsps
