import csv
# import logging
# import logging.handlers
import pickle
# import sys
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from collections import Iterable
from scipy.spatial import ConvexHull
from tabulate import tabulate
from utils import polar_to_kartesian


# logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
# LOG_FILE = "PolarDiagram.log"

# polardiagram_logger = logging.getLogger(__name__)
# console_handler = logging.StreamHandler(sys.stdout)
# file_handler = logging.handlers.TimedRotatingFileHandler(LOG_FILE, when='midnight')
# polardiagram_logger.addHandler(console_handler)
# polardiagram_logger.setLevel(logging.DEBUG)

# V: Noch in Arbeit
def convert(obj, convert_type):
    """Converts a PolarDiagram object to another type of PolarDiagram object

    :param obj:
        PolarDiagram object to be converted
    :type obj: ``PolarDiagram``
    :param convert_type:

    :type convert_type: ``PolarDiagram``
    """
    if type(object) is convert_type:
        return obj

    if convert_type is PolarDiagramTable:
        # wind_speed_resolution = obj.wind_speeds
        # wind_angle_resolution = obj.wind_angles
        # data = obj.boat_speeds
        # return PolarDiagramTable(wind_angle_resolution=wind_angle_resolution,
        #                         wind_speed_resolution=wind_speed_resolution,
        #                        data=data)
        pass

    if convert_type is PolarDiagramPointcloud:
        # wind_speeds = obj.wind_speeds
        # wind_angles = obj.wind_angles
        # boat_speeds = ...
        pass

    #    return PolarDiagramPointcloud()


def convex_hull_polar(points_radians, points_angles):
    # V: Bestimmt die konvexe Hülle einer endlichen Punktmenge welche in
    #    Polarkoordinaten gegeben ist.

    #    Die Berechnung findet über die scipy.spatial.ConvexHull-Funktion
    #    statt.
    converted_points = polar_to_kartesian(points_radians, points_angles)
    return ConvexHull(converted_points)


def to_csv(csv_path, obj):
    """Writes a PolarDiagram object to a .csv-file

    :param csv_path:
        Path to a .csv-file
    :type csv_path: ``str``
    :param obj:
        PolarDiagram object
    :type obj: ``PolarDiagram``
    """
    obj.to_csv(csv_path)


def from_csv(csv_path):
    """Creates a PolarDiagram object from a .csv-file

    :param csv_path:
        Path to a .csv-file
    :type csv_path: ``str``
    """
    with open(csv_path, 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=',', quotechar='"')
        row1 = next(csv_reader)[0]
        if row1 == "PolarDiagramTable":
            next(csv_reader)
            wind_speed_resolution = [eval(s) for s in next(csv_reader)]
            if len(wind_speed_resolution) == 1:
                wind_speed_resolution = wind_speed_resolution[0]
            next(csv_reader)
            wind_angle_resolution = [eval(a) for a in next(csv_reader)]
            if len(wind_angle_resolution) == 1:
                wind_angle_resolution = wind_angle_resolution[0]
            data = []
            next(csv_reader)
            for row in csv_reader:
                data.append([eval(entry) for entry in row])
            return PolarDiagramTable(data=np.array(data),
                                     wind_angle_resolution=wind_angle_resolution,
                                     wind_speed_resolution=wind_speed_resolution)
        elif row1 == "PolarDiagramPointcloud":
            next(csv_reader)
            data = []
            for row in csv_reader:
                data.append([eval(entry) for entry in row])
            return PolarDiagramPointcloud(data=np.array(data))


def pickling(pkl_path, obj):
    """Writes a PolarDiagram object to a .pkl-file

    :param pkl_path:
        Path to a .pkl-file
    :type pkl_path: ``str``
    :param obj:
        PolarDiagram object
    :type obj: ``PolarDiagram``
    """
    obj.pickling(pkl_path)


def depickling(pkl_path):
    """Creates a PolarDiagram object from a .pkl-file

    :param pkl_path:
        Path to a .pkl-file
    :type pkl_path: ``str``
    """
    with open(pkl_path, 'rb') as file:
        return pickle.load(file)


class PolarDiagramException(Exception):
    def __init__(self, exception_type, *args):
        message_dict = {
            "Wrong dimension": "Expecting 2 dimensional array to be viewed as Polar Diagram Tableau," +
                               f"\n got {args[0]} dimensional array instead.",
            "Wrong resolution": "Expecting resolution of type 'Iterable' or 'int/float'," +
                                f"\n got resolution of type {args[0]} instead",
            "Wrong shape": f"Expecting array with shape {args[0]},\n got array with shape {args[1]} instead",
            "Wind speed not in resolution": f"Expecting wind speed to be in {args[0]},\n got {args[1]} instead",
            "Wind angle not in resolution": f"Expecting wind angle to be in {args[0]},\n got {args[1]} instead",
            "No points found": f"The given true wind speed {args[0]} yielded no points in the current point cloud",
        }
        if exception_type in message_dict:
            super().__init__(message_dict[exception_type])
        else:
            super().__init__(exception_type)


class PolarDiagram(ABC):
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def to_csv(self, csv_path):
        pass

    def pickling(self, pkl_path):
        with open(pkl_path, 'wb') as file:
            pickle.dump(self, file)

    @abstractmethod
    def polar_plot_slice(self, true_wind_speed, **kwargs):
        pass

    @abstractmethod
    def flat_plot_slice(self, true_wind_speed, **kwargs):
        pass


class PolarDiagramTable(PolarDiagram):
    """
    A class to represent, visualize and work with a polar performance diagram in form of a table.

    ...

    Attributes
    ----------
    _wind_speed_resolution : ``list``

    _wind_angle_resolution : ``list``

    _data : ``numpy.ndarray``


    Methods
    -------
    __str__():

    __repr__():

    to_csv(csv_path):
        Writes object to a .csv-file
    change_entry(data, **kwargs):
        Changes entries in table
    get_slice_data(wind_speed):
        Returns a column of table
    polar_plot_slice(wind_speed, **kwargs):
        Polar plot of a slice(column) of the diagram
    flat_plot_slice(wind_speed, **kwargs):
        Kartesian plot of a slice(column) of the diagram
    plot_3d(**kwargs):
        3D-plot of the diagram
    plot_convex_hull_slice(wind_speed, **kwargs):
        Polar plot of the convex hull of a slice(column) of the diagram
    plot_convex_hull_3d(**kwargs):
        3D-plot of the convex hull of the diagram
    """
    def __init__(self, **kwargs):
        # V: Es werden 3 Keyword-Arguments unterstützt: "wind_angle_resolution"
        #                                               "wind_speed_resolution"
        #                                               "data"
        #    Falls PolarDiagramTable keine Keyword-Arguments bekommt
        #    wird ein PolarDiagramTable-Objekt mit den Attributen
        #        self._resolution_wind_speed = [2,4,6,...., 36,38,40]
        #        self._resolution_wind_angle = [0,5,10,.....,350,355]
        #        self._data = np.zeros((72,20))
        #    erstellt
        if "wind_angle_resolution" in kwargs:
            wind_angle_resolution = kwargs["wind_angle_resolution"]
            if isinstance(wind_angle_resolution, Iterable):
                self._resolution_wind_angle = list(wind_angle_resolution)
            elif isinstance(wind_angle_resolution, int) or isinstance(wind_angle_resolution, float):
                self._resolution_wind_angle = list(np.arange(wind_angle_resolution, 360,
                                                             wind_angle_resolution))
            else:
                raise PolarDiagramException("Wrong resolution", type(wind_angle_resolution))
        else:
            self._resolution_wind_angle = list(np.arange(0, 360, 5))

        if "wind_speed_resolution" in kwargs:
            wind_speed_resolution = kwargs["wind_speed_resolution"]
            if isinstance(wind_speed_resolution, Iterable):
                self._resolution_wind_speed = list(wind_speed_resolution)
            elif isinstance(wind_speed_resolution, int) or isinstance(wind_speed_resolution, float):
                self._resolution_wind_speed = list(np.arange(wind_speed_resolution, 40,
                                                             wind_speed_resolution))
            else:
                raise PolarDiagramException("Wrong resolution", type(wind_speed_resolution))
        else:
            self._resolution_wind_speed = list(np.arange(2, 42, 2))

        if "data" in kwargs:
            data = np.array(kwargs["data"])
            if data.ndim != 2:
                raise PolarDiagramException("Wrong dimension", data.ndim)
            if data.shape != (len(self._resolution_wind_angle), len(self._resolution_wind_speed)):
                raise PolarDiagramException("Wrong shape",
                                            (len(self._resolution_wind_angle), len(self._resolution_wind_speed)),
                                            data.shape)

            self._data = data
        else:
            self._data = np.zeros((len(self._resolution_wind_angle), len(self._resolution_wind_speed)))

    def __str__(self):
        # V: Gibt eine Tabelle mit den Zeilen und Spaltenbezeichnungen
        #    der Wind angle und Wind speed resolution.
        #    Falls es mehr als 15 Spalten gibt, werden nur die ersten 15 ausgegeben
        if len(self._resolution_wind_speed) > 15:
            table = np.c_[self._resolution_wind_angle, self._data[:, :15]]
            # print / raise PolarDiagramException() / logging.INFO?
            return tabulate(table, headers=["TWA \\ TWS"] + self._resolution_wind_speed[:15])
        else:
            table = np.c_[self._resolution_wind_angle, self._data]
            return tabulate(table, headers=["TWA \\ TWS"] + self._resolution_wind_speed)

    # V: Weiß nicht ob das hier tatsächlich passt, dass als "Property" zu deklarieren
    @property
    def wind_angles(self):
        return self._resolution_wind_angle

    @property
    def wind_speeds(self):
        return self._resolution_wind_speed

    @property
    def boat_speeds(self):
        return self._data

    def to_csv(self, csv_path):
        """Creates a .csv-file with the following entries
            PolarDiagramTable
            Wind speed resolution:
            self.wind_speeds
            Wind angle resolution:
            self.wind_angles
            Boat speeds:
            self.boat_speeds

        :param csv_path:
            Path to .csv-file
        :type csv_path: ``str``
        """
        with open(csv_path, 'w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"')
            csv_writer.writerow(["PolarDiagramTable"])
            csv_writer.writerow(["Wind speed resolution:"])
            csv_writer.writerow(self.wind_speeds)
            csv_writer.writerow(["Wind angle resolution:"])
            csv_writer.writerow(self.wind_angles)
            csv_writer.writerow(["Boat speeds:"])
            csv_writer.writerows(self.boat_speeds)

    # V: Noch in Arbeit.
    #    Dieses Monster muss noch doll überarbeitet werden
    def change_entry(self, data, **kwargs):
        r"""Changes specified entries in self._data

        :param data:
            New data that is to be added
        :type data: ``array_like``
        :param kwargs:
            See below

        :Keyword Arguments:
            * *true_wind_speed* (``Iterable``, ``int`` or ``float``) --

            * *apparent_wind_speed* (``Iterable``, ``int`` or ``float``) --

            * *true_wind_angle* (``Iterable``, ``int`` or ``float``) --

            * *apparent_wind_angle* (``Iterable``, ``int`` or ``float``) --

        """

        if "true_wind_speed" in kwargs:
            wind_speeds = list(kwargs["true_wind_speed"])
            speed_ind = [i for i in range(len(self._resolution_wind_speed))
                         if self._resolution_wind_speed[i] in wind_speeds]
            if len(speed_ind) < len(wind_speeds):
                pass

            if "true_wind_angle" in kwargs:
                wind_angles = list(kwargs["true_wind_angle"])
                angle_ind = [i for i in range(len(self._resolution_wind_angle))
                             if self._resolution_wind_angle[i] in wind_angles]
                if len(angle_ind) < len(wind_angles):
                    pass
                data = np.array(data)
                mask = np.zeros(self._data.shape, dtype=bool)
                for i in angle_ind:
                    for j in speed_ind:
                        mask[i, j] = True
                try:
                    self._data[mask] = data.flat
                except ValueError:
                    raise PolarDiagramException("Wrong Shape",
                                                (len(angle_ind), len(speed_ind)),
                                                data.shape)
            else:
                data = np.array(data)
                try:
                    self._data[:, speed_ind] = data
                except ValueError:
                    raise PolarDiagramException("Wrong Shape")

        elif "true_wind_angle" in kwargs:
            wind_angles = list(kwargs["true_wind_speed"])
            angle_ind = [i for i in range(len(self._resolution_wind_angle))
                         if self._resolution_wind_angle[i] in wind_angles]
            if len(angle_ind) < len(wind_angles):
                pass

            data = np.array(data)
            try:
                self._data[angle_ind, :] = data
            except ValueError:
                raise PolarDiagramException("Wrong Shape")
        else:
            data = np.array(data)
            if data.shape != (len(self._resolution_wind_angle), len(self._resolution_wind_speed)):
                raise PolarDiagramException("Wrong shape",
                                            (len(self._resolution_wind_angle), len(self._resolution_wind_speed)),
                                            data.shape)
            self._data = data

    def get_slice_data(self, wind_speed):
        # V: Für einen gegebenen wind speed wird die Spalte der Tabelle,
        #    welche zu diesem korrespondiert zurückgegeben
        #    Ist der wind speed nicht in der resolution vorhanden wird eine Exception geraised
        try:
            column = self._resolution_wind_speed.index(wind_speed)
            return self._data[:, column]
        except ValueError:
            raise PolarDiagramException("Wind speed not in resolution", self._resolution_wind_speed,
                                        wind_speed)

    def polar_plot_slice(self, wind_speed, **kwargs):
        # V: Für einen gegebenen wind speed werden (falls vorhanden) die korrespondierende
        #    Spalte y der Tabelle sowie die wind angles x in self._resolution_wind_angle als
        #    Punkte (x[i], y[i]) in einem Polarplot dargestellt

        # V: Funktion akzeptiert die selben Keywords wie matplotlib.pyplot.plot

        boat_speeds = self.get_slice_data(wind_speed)
        wind_angles = np.deg2rad(self._resolution_wind_angle)
        polar_plot = plt.subplot(1, 1, 1, projection='polar')
        polar_plot.set_theta_zero_location('N')
        polar_plot.set_theta_direction('clockwise')
        return polar_plot.plot(wind_angles, boat_speeds, **kwargs)

    def flat_plot_slice(self, wind_speed, **kwargs):
        # V: Für einen gegebenen wind speed werden (falls vorhanden)
        #    die korrespondierende Spalte y der Tabelle sowie die wind angles x in
        #    self._resolution_wind_angle als Punkte (x[i], y[i]) in einem kartesischen
        #    Plot dargestellt.

        # V: Funktion akzeptiert die selben Keywords wie matplotlib.pyplot.plot

        boat_speeds = self.get_slice_data(wind_speed)
        wind_angles = self._resolution_wind_angle
        flat_plot = plt.subplot(1, 1, 1)
        plt.xlabel("True Wind Angle")
        plt.ylabel("Boat Speed")
        return flat_plot.plot(wind_angles, boat_speeds, **kwargs)

    def plot_3d(self):
        pass

    def plot_convex_hull_slice(self, wind_speed, **kwargs):
        # V: Für einen gegebenen wind speed wird (falls vorhanden) die konvexe Hülle
        #    der Punktmenge bestehend aus Punkten (x[i], y[i]), wobei x die wind angles
        #    in self._resolution_wind_angle und y die zum wind speed korrespondierende Spalte
        #    der Tabelle ist, berechnet und in einem Polarplot dargestellt

        # V: Funktion akzeptiert die selben Keywords wie matplotlib.pyplot.plot

        speeds = self.get_slice_data(wind_speed)
        angles = np.deg2rad(self._resolution_wind_angle)
        vert = sorted(convex_hull_polar(speeds.copy(), angles.copy()).vertices)
        wind_angles = []
        boat_speeds = []
        for i in vert:
            wind_angles.append(angles[i])
            boat_speeds.append(speeds[i])
        convex_hull_plot = plt.subplot(1, 1, 1, projection='polar')
        convex_hull_plot.set_theta_zero_location('N')
        convex_hull_plot.set_theta_direction('clockwise')
        return convex_hull_plot.plot(wind_angles, boat_speeds, **kwargs)

    def plot_convex_hull_3d(self, **kwargs):
        pass


class PolarDiagramCurve(PolarDiagram):
    # V: Noch in Arbeit
    def __init__(self, f, *params):
        # V: Hier noch mögliche Errorchecks?
        self._f = f
        self._params = params

    def __str__(self):
        return f"Function: {self._f}\n Optimal parameters: {self._params}"

    @property
    def curve(self):
        return self._f

    @property
    def parameters(self):
        return self._params

    # V: Noch in Arbeit
    def to_csv(self, csv_path):
        """Creates a .csv-file with the following entries
            PolarDiagramCurve
            Function:
            self.curve
            Parameters:
            self.parameters

        :param csv_path:
            Path to .csv-file
        :type csv_path: ``str``
        """
        with open(csv_path, 'w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"')
            csv_writer.writerow(["PolarDiagramCurve"])
            csv_writer.writerow(["Function:"])
            csv_writer.writerow(["self.curve"])
            csv_writer.writerow(["Parameters:"])
            csv_writer.writerow(self.parameters)

    # V: Noch in Arbeit
    # -> np.linspace + meshgrid verwenden?
    def polar_plot_slice(self, true_wind_speed, **kwargs):
        # polar_plot = plt.subplot(1,1,1, projection = 'polar')
        # polar_plot.set_theta_zero_location('N')
        # polar_plot.set_theta_direction('clockwise')
        # return polar_plot.plot(self._f,**kwargs)
        pass

    # V: Noch in Arbeit
    # -> np.linspace + meshgrid verwenden?
    def flat_plot_slice(self, true_wind_speed, **kwargs):
        # flat_plot = plt.subplot(1,1,1)
        # plt.xlabel("True Wind Angle")
        # plt.ylabel("Boat Speed")
        # return flat_plot(self._f, **kwargs)
        pass


class PolarDiagramPointcloud(PolarDiagram):
    def __init__(self, data=np.array([[0, 0, 0]]), **kwargs):
        if len(data[0]) != 3:
            raise PolarDiagramException("Wrong shape", (len(data[0]), "x"), (3, "x"))
        self._data = np.array(data)

    def __str__(self):
        return tabulate(self._data, headers=["TWS", "TWA", "BSP"])

    @property
    def wind_speeds(self):
        return list(dict.fromkeys(self._data[:, 0]))

    @property
    def wind_angles(self):
        return list(dict.fromkeys(self._data[:, 1]))

    @property
    def get_points(self):
        return self._data

    def to_csv(self, csv_path):
        """Creates a .csv-file with the following entries
            PolarDiagramPointcloud
            True wind speed: , True wind angle: , Boat speed:
            self.get_points

        :param csv_path:
            Path to .csv-file
        :type csv_path: ``str``
        """
        with open(csv_path, 'w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"')
            csv_writer.writerow(["PolarDiagramPointcloud"])
            csv_writer.writerow(["True wind speed: ", "True wind angle: ", "Boat speed: "])
            csv_writer.writerows(self.get_points)

    def add_points(self, points):
        try:
            self._data = np.r_[self._data, np.array(points)]
        except ValueError:
            raise PolarDiagramException("Wrong shape", (3, "x"), (len(points[0]), "x"))

    def polar_plot_slice(self, true_wind_speed, **kwargs):
        # V: Für einen gegebenen wind speed werden all
        #
        #

        # V: Funktion akzeptiert die selben Keywords wie matplotlib.pyplot.plot
        #    Falls "linestyle" bzw. "ls" nicht spezifiziert wird, setzten wir
        #    es standardmäßig auf '', damit auch wirklich eine Punktwolke
        #    dargestellt wird

        if "linestyle" not in kwargs and "ls" not in kwargs:
            kwargs["ls"] = ''

        if "marker" not in kwargs:
            kwargs["marker"] = 'o'

        # V: Wir filtern nun das Array nach Punkten deren erster Eintrag gleich dem
        #    gegebenen wind speed ist und teilen die zweiten und dritten Einträge dann
        #    in "x" und "y" Koordinaten auf.
        #    Sollte es keine solche Punkte geben, wird eine Exception geraised

        points = self._data[self._data[:, 0] == true_wind_speed][:, 1:]
        try:
            boat_speeds = points[:, 1]
            wind_angles = np.deg2rad(points[:, 0])
        except (ValueError, IndexError):
            raise PolarDiagramException("No points found", true_wind_speed)

        polar_plot = plt.subplot(1, 1, 1, projection='polar')
        polar_plot.set_theta_zero_location('N')
        polar_plot.set_theta_direction('clockwise')
        return polar_plot.plot(wind_angles, boat_speeds, **kwargs)

    def flat_plot_slice(self, true_wind_speed, **kwargs):
        # V: Für einen gegebenen wind speed werden all
        #
        #

        # V: Funktion akzeptiert die selben Keywords wie matplotlib.pyplot.plot
        #    Falls "linestyle" bzw. "ls" nicht spezifiziert wird, setzten wir
        #    es standardmäßig auf '', damit auch wirklich eine Punktwolke
        #    dargestellt wird

        if "linestyle" not in kwargs and "ls" not in kwargs:
            kwargs["ls"] = ''

        if "marker" not in kwargs:
            kwargs["marker"] = 'o'

        # V: Wir filtern nun das Array nach Punkten deren erster Eintrag gleich dem
        #    gegebenen wind speed ist und teilen die zweiten und dritten Einträge dann
        #    in "x" und "y" Koordinaten auf.
        #    Sollte es keine solche Punkte geben, wird eine Exception geraised

        points = self._data[self._data[:, 0] == true_wind_speed][:, 1:]
        try:
            boat_speeds = points[:, 1]
            wind_angles = points[:, 0]
        except (ValueError, IndexError):
            raise PolarDiagramException("No points found", true_wind_speed)

        flat_plot = plt.subplot(1, 1, 1)
        plt.xlabel("True Wind Angle")
        plt.ylabel("Boat Speed")
        return flat_plot.plot(wind_angles, boat_speeds, **kwargs)

    def plot_3d(self):
        pass

    def plot_convex_hull_slice(self, true_wind_speed, **kwargs):
        # V: Für einen gegebenen wind speed werden all
        #
        #

        # V: Funktion akzeptiert die selben Keywords wie matplotlib.pyplot.plot

        # V: Wir filtern nun das Array nach Punkten deren erster Eintrag gleich dem
        #    gegebenen wind speed ist und teilen die zweiten und dritten Einträge dann
        #    in "x" und "y" Koordinaten auf.
        #    Sollte es keine solche Punkte geben, wird eine Exception geraised

        points = np.array(sorted(list(self._data[self._data[:, 0] == true_wind_speed][:, 1:]),
                                 key=lambda x: x[0]))
        try:
            speeds = points[:, 1]
            angles = np.deg2rad(points[:, 0])
        except (ValueError, IndexError):
            raise PolarDiagramException("No points found", true_wind_speed)

        vert = sorted(convex_hull_polar(speeds.copy(), angles.copy()).vertices)
        wind_angles = []
        boat_speeds = []
        for i in vert:
            wind_angles.append(angles[i])
            boat_speeds.append(speeds[i])

        convex_hull_plot = plt.subplot(1, 1, 1, projection='polar')
        convex_hull_plot.set_theta_zero_location('N')
        convex_hull_plot.set_theta_direction('clockwise')
        return convex_hull_plot.plot(wind_angles, boat_speeds, **kwargs)

    def plot_convex_hull_3d(self):
        pass
