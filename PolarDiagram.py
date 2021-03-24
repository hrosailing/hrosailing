# import collections
import csv
# import logging
# import logging.handlers
import pickle
# import sys
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial import ConvexHull
from tabulate import tabulate


# V: Ich werde noch schauen müssen, wo ich alles logging verwende. Kommt aber noch
# logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
# LOG_FILE = "PolarDiagram.log"

# polardiagram_logger = logging.getLogger(__name__)
# console_handler = logging.StreamHandler(sys.stdout)
# file_handler = logging.handlers.TimedRotatingFileHandler(LOG_FILE, when='midnight')
# polardiagram_logger.addHandler(console_handler)
# polardiagram_logger.setLevel(logging.DEBUG)

# V: Noch in Arbeit
# def polar_diagram_from_data(, **kwargs):

# V: Noch in Arbeit
def convert(obj, convert_type):
    if type(object) is convert_type:
        return obj

    if convert_type is PolarDiagramTable:
        pass

    #    return PolarDiagramTable()

    if convert_type is PolarDiagramCurve:
        pass

    #    return PolarDiagramCurve()

    if convert_type is PolarDiagramPointcloud:
        pass

    #    return PolarDiagramPointcloud()


def convex_hull_polar(points_radians, points_angles):
    # V: Bestimmt die konvexe Hülle einer endlichen Punktmenge welche in
    #    Polarkoordinaten gegeben ist.

    #    Die Berechnung findet über die scipy.spatial.ConvexHull-Funktion
    #    statt.
    def polar_to_kartesian(radians, angles):
        # V: Wandelt die eindimensionalen Arrays der Polarkoordinaten der Punktmenge
        #    in ein zweidimensionales Array mit den kartesischen Koordinaten der
        #    Punktmenge um.
        return np.column_stack((radians * np.cos(angles), radians * np.sin(angles)))
    converted_points = polar_to_kartesian(points_radians, points_angles)
    return ConvexHull(converted_points)


def to_csv(csv_path, obj):
    # V: Erstellt eine .csv-Datei und schreibt eine menschlich lesbare
    #    Representation der Daten eines PolarDiagram-Objekts hinein

    #    Die Funktion selber ruft die interne Klassenmethode des
    #    PolarDiagram-Objektes auf
    obj.to_csv(csv_path)


def from_csv(csv_path):
    # V: Erstellt ein PolarDiagram-Objekt aus den Daten in einer
    #    .csv-Datei.
    #    Die akzeptierten Formate für die jeweiligen PolarDiagram-Objekte
    #    finden sich in den jeweiligen to_csv-Funktionen.
    #
    #    Achtung: Funktion kann kein PolarDiagramCurve-Objekt erstellen.
    #    Dazu kann die Funktion depickling verwendet werden.
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
            return PolarDiagramTable(data=np.array(data), wind_angle_resolution=wind_angle_resolution,
                                     wind_speed_resolution=wind_speed_resolution)
        elif row1 == "PolarDiagramPointcloud":
            next(csv_reader)
            data = []
            for row in csv_reader:
                data.append([eval(entry) for entry in row])
            return PolarDiagramPointcloud(data=np.array(data))


def pickling(pkl_path, obj):
    # V: Speichert die Daten eines PolarDiagram-Objekts in einer .pkl-File
    #    Diese können mittels der depickling-Funktion wieder in Python
    #    geladen werden.

    #    Die Funktion selber ruft die interne Klassenmethode des
    #    PolarDiagram-Objektes auf.
    obj.pickling(pkl_path)


def depickling(pkl_path):
    # V: Lädt gespeicherte Daten aus einer .pkl-File in Python.
    #    Hiermit können PolarDiagram-Objekte erstellt werden.
    with open(pkl_path, 'rb') as file:
        return pickle.load(file)


class PolarDiagramException(Exception):
    def __init__(self, exception_type, *args):
        message_dict = {
            "Wrong dimension": "Expecting 2 dimensional array to be viewed as Polar Diagram Tableau," +
                               f"\n got {args[0]} dimensional array instead.",
            "Wrong resolution": "Expecting resolution of type 'list' or 'int/float'," +
                                f"\n got resolution of type {args[0]} instead",
            "No data given": "No new data to update old was given",
            "Wrong shape": f"Expecting array with shape {args[0]},\n got array with shape {args[1]} instead",
            "Wind speed not in resolution": f"Expecting wind speed to be in {args[0]},\n got {args[1]} instead",
            "Wind angle not in resolution": f"Expecting wind angle to be in {args[0]},\n got {args[1]} instead",
            "No entry specified": f"Expecting to get an entry to update",
            "No points found": f"The given true wind speed {args[0]} yielded no points in the current point cloud"
        }
        if exception_type in message_dict:
            super().__init__(message_dict[exception_type])
        else:
            super().__init__(exception_type)


class PolarDiagram(ABC):
    # V: Abstrakte Basisklasse der PolarDiagram-Objekte.

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def to_csv(self, csv_path):
        pass

    def pickling(self, pkl_path):
        # V: Siehe die externe pickling-Funktion.
        with open(pkl_path, 'wb') as file:
            pickle.dump(self, file)

    @abstractmethod
    def polar_plot_slice(self, true_wind_speed, **kwargs):
        pass

    @abstractmethod
    def flat_plot_slice(self, true_wind_speed, **kwargs):
        pass


class PolarDiagramTable(PolarDiagram):
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
            if isinstance(wind_angle_resolution, list):
                self._resolution_wind_angle = wind_angle_resolution
            elif isinstance(wind_angle_resolution, int) or isinstance(wind_angle_resolution, float):
                self._resolution_wind_angle = list(np.arange(wind_angle_resolution, 360, wind_angle_resolution))
            else:
                raise PolarDiagramException("Wrong resolution", type(wind_angle_resolution))
        else:
            self._resolution_wind_angle = list(np.arange(0, 360, 5))

        if "wind_speed_resolution" in kwargs:
            wind_speed_resolution = kwargs["wind_speed_resolution"]
            if isinstance(wind_speed_resolution, list):
                self._resolution_wind_speed = wind_speed_resolution
            elif isinstance(wind_speed_resolution, int) or isinstance(wind_speed_resolution, float):
                self._resolution_wind_speed = list(np.arange(wind_speed_resolution, 40, wind_speed_resolution))
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
        # V: Erstellt eine .csv-Datei mit dem Format
        #    PolarDiagramTable
        #    Wind speed resolution:
        #    self._resolution_wind_speed
        #    Wind angle resolution:
        #    self._resolution_wind_angle
        #    Boat speeds:
        #    self._data
        with open(csv_path, 'w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"')
            csv_writer.writerow(["PolarDiagramTable"])
            csv_writer.writerow(["Wind speed resolution:"])
            csv_writer.writerow(list(self._resolution_wind_speed))
            csv_writer.writerow(["Wind angle resolution:"])
            csv_writer.writerow(list(self._resolution_wind_angle))
            csv_writer.writerow(["Boat speeds:"])
            for row in self._data:
                csv_writer.writerow(row)

    # V: Noch in Arbeit.
    #    Dieses Monster muss noch doll überarbeitet werden
    def change_entry(self, **kwargs):
        # V: Will hier irgendwie gerne das multiple Entries gleichzeitig geändert werden können
        #    Weiß aber nicht, ob das so sinnvoll ist.
        #    Hab es jetzt erstmal so gemacht, dass ganze Zeilen/Spalten geändert werden können
        #    oder mehrere Entries in der selben Zeile/Spalte
        if "data" not in kwargs:
            # V: Falls keine neuen Daten an die Funktion übergeben wurden,
            #    brechen wir sofort ab, da nichts zu tun
            #    Weiß aber nicht ob man hier tatsächlich eine Exception raisen sollte?
            raise PolarDiagramException("No data given")
        data = kwargs["data"]

        # V: Prüfen zunächst ob "true_wind_speed" übergeben wurde
        if "true_wind_speed" in kwargs:
            # V: Falls ja, prüfen wir danach ob "true_wind_angle" übergeben wurde
            true_wind_speed = kwargs["true_wind_speed"]

            if "true_wind_angle" in kwargs:
                true_wind_angle = kwargs["true_wind_angle"]

                # V: Hier gehen wir die möglichen Typen der Variablen true_wind_angle
                #    und true_wind_speed durch
                if isinstance(true_wind_speed, list) and isinstance(true_wind_angle, list):
                    # V: Falls beides Listen sind, wird eine Exception geraised.
                    #    Der Grund hierfür ist, dass es wohl nicht wirklich möglich ist
                    #    Subarrays von np-arrays zu überschreiben.
                    #    Zumindest hab ich nichts gefunden
                    raise PolarDiagramException("Multiple lists given", type(true_wind_speed), type(true_wind_angle))

                elif not isinstance(true_wind_speed, list) and not isinstance(true_wind_angle, list):
                    # V: Falls beide keine Listen (also int/float) sind (hoffentlich)
                    #    prüfen wir zunächst, ob data ebenfalls nur eine Zahl ist,
                    if np.array(data).shape != ():
                        raise PolarDiagramException("Wrong shape", (), np.array(data).shape)

                    # V: Hier versuchen wir die Indizes des übergebenen wind speeds und wind angles
                    #    zu finden.
                    #    Falls nicht vorhanden wird Exception geraised
                    try:
                        ind1 = self._resolution_wind_angle.index(true_wind_angle)
                    except ValueError:
                        raise PolarDiagramException("Wind angle not in resolution", self._resolution_wind_angle,
                                                    true_wind_angle)
                    try:
                        ind2 = self._resolution_wind_speed.index(true_wind_speed)
                    except ValueError:
                        raise PolarDiagramException("Wind speed not in resolution", self._resolution_wind_speed,
                                                    true_wind_speed)

                    # V: Schlussendlich wird der gefundene Entry überschrieben
                    self._data[ind1, ind2] = data

                # V: Nun bleiben noch die Fälle, dass jeweils eine Variable eine Liste ist und die andere nicht.
                #    Da die beiden Fälle symmetrisch sind, könnte man hier vielleicht eine interne Funktion schreiben?
                elif isinstance(true_wind_angle, list):
                    # V: Falls true_wind_angle eine Liste ist, versuchen wir wieder den Index des übergebenen
                    #    wind speeds zu finden und falls nicht vorhanden raisen wir wieder eine Exception
                    try:
                        ind = self._resolution_wind_speed.index(true_wind_speed)
                    except ValueError:
                        raise PolarDiagramException("Wind speed not in resolution", self._resolution_wind_speed,
                                                    true_wind_speed)
                    # V: Hier ermitteln wir alle Indizes der wind angles in der Liste
                    ind_list = [i for i in len(self._resolution_wind_angle)
                                if self._resolution_wind_angle[i] in true_wind_angle]
                    # V: Falls die Liste der Indizes kürzer als die Liste der wind angles ist, muss es in der
                    #    Liste wind angles geben, welche nicht in der resolution vorhanden sind
                    if len(ind_list) < len(true_wind_angle):
                        raise PolarDiagramException("Wind angle not in resolution", self._resolution_wind_angle,
                                                    true_wind_angle)

                    data = np.array(data)
                    # V: Hier wird wieder überprüft ob data die richtige Größe (Shape) hat und falls nicht
                    #    wird wieder eine Exception geraised
                    if data.shape != (len(ind_list),):
                        raise PolarDiagramException("Wrong shape", (len(ind_list), ), data.shape)

                    self._data[ind_list, ind] = data

                # V: Läuft analog zum vorherigen Fall ab
                elif isinstance(true_wind_speed, list):
                    try:
                        ind = self._resolution_wind_angle.index(true_wind_angle)
                    except ValueError:
                        raise PolarDiagramException("Wind angle not in resolution", self._resolution_wind_angle,
                                                    true_wind_angle)

                    ind_list = [i for i in len(self._resolution_wind_speed)
                                if self._resolution_wind_speed[i] in true_wind_speed]

                    if len(ind_list) < len(true_wind_speed):
                        raise PolarDiagramException("Wind speed not in resolution", self._resolution_wind_speed,
                                                    true_wind_speed)

                    data = np.array(data)
                    if data.shape != (len(ind_list),):
                        raise PolarDiagramException("Wrong shape", (len(ind_list), ), data.shape)

                    self._data[ind, ind_list] = data

            else:
                # V: Muss überarbeitet werden
                if isinstance(true_wind_speed, list):
                    if len(true_wind_speed) > 1:
                        ind_list = [i for i in len(self._resolution_wind_speed)
                                    if self._resolution_wind_speed[i] in true_wind_speed]

                        if len(ind_list) < len(true_wind_speed):
                            raise PolarDiagramException("Wind speed not in resolution", self._resolution_wind_speed,
                                                        true_wind_speed)

                        data = np.array(data)
                        if data.shape != (len(ind_list),):
                            raise PolarDiagramException("Wrong shape", (len(ind_list),), data.shape)

                        self._data[:, ind_list] = data

                    else:
                        true_wind_speed = true_wind_speed[0]
                try:
                    ind = self._resolution_wind_speed.index(true_wind_speed)
                except ValueError:
                    raise PolarDiagramException("Wind speed not in resolution", self._resolution_wind_speed,
                                                true_wind_speed)
                data = np.array(data)
                if data.shape != (len(self._resolution_wind_angle),):
                    raise PolarDiagramException("Wrong shape", (len(self._resolution_wind_angle),), data.shape)

                self._data[:, ind] = data

        # V: Muss überarbeitet werden
        elif "true_wind_angle" in kwargs:
            true_wind_angle = kwargs["true_wind_angle"]
            if isinstance(true_wind_angle, list):
                if len(true_wind_angle) > 1:
                    ind_list = [i for i in len(self._resolution_wind_angle)
                                if self._resolution_wind_angle[i] in true_wind_angle]
                    if len(ind_list) < len(true_wind_angle):
                        raise PolarDiagramException("Wind angle not in resolution", self._resolution_wind_speed,
                                                    true_wind_angle)

                    data = np.array(data)
                    if data.shape != (len(ind_list),):
                        raise PolarDiagramException("Wrong shape", (len(ind_list),), data.shape)

                    self._data[:, ind_list] = data

                else:
                    true_wind_angle = true_wind_angle[0]

            try:
                ind = self._resolution_wind_angle.index(true_wind_angle)
            except ValueError:
                raise PolarDiagramException("Wind angle not in resolution", self._resolution_wind_angle,
                                            true_wind_angle)

            data = np.array(data)
            if data.shape != (len(self._resolution_wind_speed),):
                raise PolarDiagramException("Wrong shape", (len(self._resolution_wind_speed),), data.shape)

            self._data[ind, :] = data

        # V: Falls weder true_wind_speed noch true_wind_angle in den Keywords vorhanden sind,
        #    überprüfen wir ob die Größe von data gleich der von self._data ist.
        #    Falls ja, dann wird self._data komplett überschrieben.
        #    Falls nein, dann wir eine Exception geraised
        else:
            data = np.array(data)
            if data.shape != (len(self._resolution_wind_angle), len(self._resolution_wind_speed)):
                raise PolarDiagramException("Wrong shape",
                                            (len(self._resolution_wind_angle), len(self._resolution_wind_speed)),
                                            data.shape)

            self._data = data

    def get_slice_data(self, true_wind_speed):
        # V: Für ein gegebenen wind speed wird die Spalte der Tabelle, welche zu diesem korrespondiert
        #    zurückgegeben
        #    Ist der wind speed nicht in der resolution vorhanden wird eine Exception geraised
        try:
            column = self._resolution_wind_speed.index(true_wind_speed)
            return self._data[:, column]
        except ValueError:
            raise PolarDiagramException("Wind speed not in resolution", self._resolution_wind_speed, true_wind_speed)

    def polar_plot_slice(self, true_wind_speed, **kwargs):
        boat_speeds = self.get_slice_data(true_wind_speed)
        wind_angles = np.deg2rad(self._resolution_wind_angle)
        # if all(boat_speed == np.zeros(len(self._resolution_wind_angle),)):
        #
        polar_plot = plt.subplot(1, 1, 1, projection='polar')
        polar_plot.set_theta_zero_location('N')
        polar_plot.set_theta_direction('clockwise')
        return polar_plot.plot(wind_angles, boat_speeds, **kwargs)

    def flat_plot_slice(self, true_wind_speed, **kwargs):
        boat_speeds = self.get_slice_data(true_wind_speed)
        wind_angles = self._resolution_wind_angle
        # if all(boat_speed == np.zeros(len(self._resolution_wind_angle), )):
        #
        flat_plot = plt.subplot(1, 1, 1)
        plt.xlabel("True Wind Angle")
        plt.ylabel("Boat Speed")
        return flat_plot.plot(wind_angles, boat_speeds, **kwargs)

    def plot_3d(self):
        pass

    def plot_convex_hull_slice(self, true_wind_speed, **kwargs):
        speeds = self.get_slice_data(true_wind_speed)
        angles = np.deg2rad(self._resolution_wind_angle)
        # if all(speeds == np.zeros(self._resolution_wind_angle)):
        #
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
        # self._resolution_wind_speed = ...
        # Wir sollten vielleicht eine Liste von Funktionen und Parametern abspeichern die zu bestimmten
        # Windgeschwindigkeiten passen, oder?
        self._f = f
        self._params = params

    # V: Noch in Arbeit
    def __str__(self):
        # V: Weiß nicht ob das so geprinted werden soll. Die andere Idee wäre das der Print ein Plot der Funktion ist?
        # Das man sozusagen die curve sieht, aber dafür bräuchte ich dann ja data?
        return f"Function: {self._f}\n Optimal parameters: {self._params}"

    # @property
    # def wind_speeds(self):
    #     return self._resolution_wind_speed

    @property
    def curve(self):
        return self._f

    @property
    def parameters(self):
        return self._params

    # V: Noch in Arbeit
    def to_csv(self, csv_path):
        # V: Das Format der .csv-Dateien ist
        # PolarDiagramCurve
        # #Wind speed resolution:
        # #self._resolution_wind_speed
        # Function(s):
        # self._f
        # Parameters:
        # self._params
        with open(csv_path, 'w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"')
            csv_writer.writerow(["PolarDiagramCurve"])
            # csv_writer.writerow(["Wind speed resolution:"])
            # csv_writer.writerow(list(self._resolution_wind_speed))
            csv_writer.writerow(["Function(s):"])
            csv_writer.writerow(["self._f"])
            csv_writer.writerow(["Parameters:"])
            csv_writer.writerow(list(self._params))

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
    def __init__(self, data=np.array([[0, 0, 0]])):
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
    def boat_speeds(self):
        return list(dict.fromkeys(self._data[:, 2]))

    @property
    def get_points(self):
        return self._data

    def to_csv(self, csv_path):
        # V: Das Format der .cvs-Dateien ist
        # PolarDiagramPointcloud
        # True Wind Speed: ,True Wind Angle: , Boat Speed:
        # self._data
        with open(csv_path, 'w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"')
            csv_writer.writerow(["PolarDiagramPointcloud"])
            csv_writer.writerow(["True Wind Speed: ", "True Wind Angle: ", "Boat Speed: "])
            for row in self._data:
                csv_writer.writerow(row)

    def add_points(self, points):
        try:
            self._data = np.r_[self._data, np.array(points)]
        except ValueError:
            raise PolarDiagramException("Wrong shape", (len(points[0]), "x"), (3, "x"))

    def polar_plot_slice(self, true_wind_speed, **kwargs):
        # V: Hier fange ich ab, das der User möglicherweise linestyle bzw. linewidth in den keywords hat.
        # Damit man auch wirklich eine PUNKT-Wolke plottet.
        # Sollte so funktionieren, sicher bin ich mir nicht.
        if "linestyle" in kwargs or "ls" in kwargs:
            try:
                del kwargs["linestyle"]
            except KeyError:
                del kwargs["ls"]

        if "linewidth" in kwargs or "lw" in kwargs:
            try:
                del kwargs["linewidth"]
            except KeyError:
                del kwargs["lw"]
        points = self._data[self._data[:, 0] == true_wind_speed][:, 1:]
        # V: Exception, falls es keine Punkte mit diesem Wind Speed gab
        try:
            boat_speeds = points[:, 0]
            wind_angles = np.deg2rad(points[:, 1])
        except ValueError:
            raise PolarDiagramException("No points found", true_wind_speed)

        polar_plot = plt.subplot(1, 1, 1, projection='polar')
        polar_plot.set_theta_zero_location('N')
        polar_plot.set_theta_direction('clockwise')
        return polar_plot.plot(wind_angles, boat_speeds, **kwargs, ls='')

    def flat_plot_slice(self, true_wind_speed, **kwargs):
        # V: Hier fange ich ab, das der User möglicherweise linestyle bzw. linewidth in den keywords hat
        # Damit man auch wirklich eine PUNKT-Wolke plottet
        # Sollte so funktionieren, sicher bin ich mir nicht
        if "linestyle" in kwargs or "ls" in kwargs:
            try:
                del kwargs["linestyle"]
            except KeyError:
                del kwargs["ls"]

        if "linewidth" in kwargs or "lw" in kwargs:
            try:
                del kwargs["linewidth"]
            except KeyError:
                del kwargs["lw"]
        points = self._data[self._data[:, 0] == true_wind_speed][:, 1:]
        # V: Exception, falls es keine Punkte mit diesem Wind Speed gab
        try:
            boat_speeds = points[:, 0]
            wind_angles = points[:, 1]
        except ValueError:
            raise PolarDiagramException("No points found", true_wind_speed)

        flat_plot = plt.subplot(1, 1, 1)
        plt.xlabel("True Wind Angle")
        plt.ylabel("Boat Speed")
        return flat_plot.plot(wind_angles, boat_speeds, **kwargs, ls='')

    def plot_3d(self):
        pass

    def plot_convex_hull_slice(self, true_wind_speed, **kwargs):
        points = self._data[self._data[:, 0] == true_wind_speed][:, 1:]
        # V: Exception, falls es keine Punkte mit diesem Wind Speed gab
        try:
            speeds = points[:, 0]
            angles = np.deg2rad(points[:, 1])
        except ValueError:
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
