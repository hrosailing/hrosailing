import csv
# import logging
# import logging.handlers
# import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tabulate import tabulate
from scipy.spatial import ConvexHull


# V: Ich werde noch schauen müssen, wo ich alles logging verwende. Kommt aber noch
# logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
# LOG_FILE = "PolarDiagram.log"

# polardiagram_logger = logging.getLogger(__name__)
# console_handler = logging.StreamHandler(sys.stdout)
# file_handler = logging.handlers.TimedRotatingFileHandler(LOG_FILE, when='midnight')
# polardiagram_logger.addHandler(console_handler)
# polardiagram_logger.setLevel(logging.DEBUG)


# V: Noch in Arbeit
def convert(obj, convert_type, **kwargs):
    # V: Die kwargs sollen hier verschiedenen Spezifikationen dienen. Wenn man zum Beispiel nur bestimmte
    # Punkte der PolarDiagramPointCloud für ein PolarDiagramTable verwenden will, kann man die wind_speed_resolution
    # und wind_angle_resolution selber festlegen und dann werden nur die Punkte betrachtet, die darauf passen.
    # Weiß aber nicht ob das ne gute Idee / notwendige Sache ist

    obj_type = type(obj)
    # V: Hier wird geprüft ob das object schon den gewünschten type hat. Wenn ja wird einfach das object unverändert
    # zurückgegeben
    if obj_type == convert_type:
        return obj

    # V: Gibt es eine guten Weg die verschiedenen Fälle abzuklappern ohne 6 if-Bedingungen oder ist das der beste Weg?
    if obj_type is PolarDiagramTable and convert_type is PolarDiagramPointcloud:
        wind_speed = obj.wind_speeds
        wind_angle = obj.wind_angles
        boat_speed = obj.boat_speeds
        points = []
        for tws in wind_speed:
            for twa in wind_angle:
                ind1 = wind_angle.index(twa)
                ind2 = wind_speed.index(tws)
                bsp = boat_speed[ind1, ind2]
                points.append([tws, twa, bsp])

        return PolarDiagramPointcloud(points)

    if obj_type is PolarDiagramPointcloud and convert_type is PolarDiagramTable:
        points = obj.get_points
        if "wind_speed_resolution" in kwargs:
            wind_speed_resolution = kwargs["wind_speed_resolution"]
        else:
            wind_speed_resolution = obj.wind_speeds

        if "wind_angle_resolution" in kwargs:
            wind_angle_resolution = kwargs["wind_angle_resolution"]
        else:
            wind_angle_resolution = obj.wind_angles

        data = np.zeros(len(wind_angle_resolution), len(wind_speed_resolution))
        for row in points:
            ind1 = wind_angle_resolution.index(row[1])
            ind2 = wind_speed_resolution.index(row[0])
            data[ind1, ind2] = row[2]

        return PolarDiagramTable(wind_speed_resolution=wind_speed_resolution,
                                 wind_angle_resolution=wind_angle_resolution,
                                 data=data)

    if obj_type is PolarDiagramTable and convert_type is PolarDiagramCurve:
        pass

    if obj_type is PolarDiagramCurve and convert_type is PolarDiagramTable:
        pass

    if obj_type is PolarDiagramPointcloud and convert_type is PolarDiagramCurve:
        pass

    if obj_type is PolarDiagramCurve and convert_type is PolarDiagramPointcloud:
        pass


def convex_hull_polar(points_radians, points_angles):
    def polar_to_kartesian(radians, angles):
        return np.column_stack((radians * np.cos(angles), radians * np.sin(angles)))
    converted_points = polar_to_kartesian(points_radians, points_angles)
    return ConvexHull(converted_points)


def to_csv(csv_path, obj):
    # V: Funktion ruft die jeweils interne Funktion des Objekts auf -> Nutzerfreundlichkeit?
    obj.to_csv(csv_path)


# V: Hauptsächlich nicht benutzt. .csv-Datein dienen eher der menschlichen Lesbarkeit.
# Für Speichern und Laden von Daten eher pickling und depickling benutzen
def from_csv(csv_path):
    with open(csv_path, "r", newline='') as file:
        csv_reader = csv.reader(file, delimiter=',', quotechar='"')
        # V: Die csv-Files werden in den to_csv-Funktionen so geschrieben, dass die
        # erste Zeile den Typ (PolarDiagramTable, PolarDiagramPointCloud) enthält
        # Dann wird hier anhand der ersten Zeile entschieden, was für ein PolarDiagram erstellt wird
        # Kann bisher nur PolarDiagramTable- und PolarDiagramPointCloud-Objekte erstellen
        # Für PolarDiagramCurve (aber auch für die anderen) ist cori_cycling zu verwenden
        row1 = next(csv_reader)[0]
        if row1 == 'PolarDiagramTable':
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
        elif row1 == 'PolarDiagramPointcloud':
            next(csv_reader)
            data = []
            for row in csv_reader:
                data.append([eval(entry) for entry in row])
            return PolarDiagramPointcloud(data=np.array(data))


def pickling(pkl_path, obj):
    # V: Funktion ruft die jeweils interne Funktion des Objekts auf -> Nutzerfreundlichkeit?
    obj.pickling(pkl_path)


def depickling(pkl_path):
    with open(pkl_path, "rb") as file:
        return pickle.load(file)

# def polar_diagram_from_data(, **kwargs):


class PolarDiagramException(Exception):
    # V: Custom Exceptions die noch erweitert werden, sollte ich auf weitere Sachen stoßen
    # Hab hier aber im Endeffekt die Struktur aus deiner QuadraticFormException-Klasse kopiert
    def __init__(self, exception_type, *args):
        message_dict = {
            "Wrong dimension": "Expecting 2 dimensional array to be viewed as Polar Diagram Tableau," +
                               f"\n got {args[0]} dimensional array instead.",
            "Wrong resolution": "Expecting resolution of type 'list' or 'int'," +
                                f"\n got resolution of type {args[0]} instead",
            "No data present": f"Expecting to get new data to update",
            "Wrong shape": f"Expecting array with shape {args[0]},\n got array with shape {args[1]} instead",
            "Wind speed not in resolution": f"Expecting wind speed to be in {args[0]},\n got {args[1]} instead",
            "Wind angle not in resolution": f"Expecting wind angle to be in {args[0]},\n got {args[1]} instead",
            "No entry specified": f"Expecting to get an entry to update"
        }
        if exception_type in message_dict:
            super().__init__(message_dict[exception_type])
        else:
            super().__init__(exception_type)


# V: Noch in Arbeit
class PolarDiagram(ABC):

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def polar_plot_slice(self, true_wind_speed, **kwargs):
        pass

    @abstractmethod
    def flat_plot_slice(self, true_wind_speed, **kwargs):
        pass

    @abstractmethod
    def to_csv(self, csv_path):
        pass

    def pickling(self, pkl_path):
        with open(pkl_path, "wb") as file:
            pickle.dump(self, file)


class PolarDiagramTable(PolarDiagram):
    def __init__(self, **kwargs):
        # V : Es werden 3 Keyword-Arguments unterstützt: "wind angle resolution"
        #                                                "wind speed resolution"
        #                                                "data"
        # Gibt es eine einfache Möglichkeit hier auch Sachen wie "speed resolution", "angle resolution" "boat speeds"
        # oder Abkürzungen wie "was" "wsr" "d" etc. zu unterstüzen?
        # matplotlib -> Cookbook -> define_aliases?
        if "wind_angle_resolution" in kwargs:
            wind_angle_resolution = kwargs["wind_angle_resolution"]
            if isinstance(wind_angle_resolution, list):
                self._resolution_wind_angle = wind_angle_resolution
            elif isinstance(wind_angle_resolution, int) or isinstance(wind_angle_resolution, float):
                self._resolution_wind_angle = list(np.arange(0, 360, wind_angle_resolution))
            else:
                raise PolarDiagramException("Wrong resolution", type(wind_angle_resolution))
        else:
            self._resolution_wind_angle = list(np.arange(0, 360, 5))

        if "wind_speed_resolution" in kwargs:
            wind_speed_resolution = kwargs["wind_speed_resolution"]
            if isinstance(wind_speed_resolution, list):
                self._resolution_wind_speed = wind_speed_resolution
            elif isinstance(wind_speed_resolution, int) or isinstance(wind_speed_resolution, float):
                self._resolution_wind_speed = list(np.arange(2, 42, wind_speed_resolution))
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

    def to_csv(self, csv_path):
        # V: Das Format für PolarDiagramTables .csv-Dateien ist
        # PolarDiagramTable
        # Wind speed resolution:
        # self._resolution_wind_speed
        # Wind angle resolution:
        # self._resolution_wind_angle
        # Boat speeds:
        # self._data
        with open(csv_path, "w", newline='') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"')
            csv_writer.writerow(['PolarDiagramTable'])
            csv_writer.writerow(["Wind speed resolution:"])
            csv_writer.writerow(list(self._resolution_wind_speed))
            csv_writer.writerow(["Wind angle resolution:"])
            csv_writer.writerow(list(self._resolution_wind_angle))
            csv_writer.writerow(["Boat speeds:"])
            for row in self._data:
                csv_writer.writerow(row)

    # V: Weiß nicht ob das eine gute Idee ist, aber damit man auch ohne Gefahr (also ohne sie zu verändern)
    # auf die Daten "zugreifen" kann. Zum Übergeben für andere Funktionen zum Beispiel....?
    @property
    def wind_angles(self):
        return self._resolution_wind_angle

    @property
    def wind_speeds(self):
        return self._resolution_wind_speed

    @property
    # Weiß nicht ob das eine gute Idee ist
    def boat_speeds(self):
        return self._data

    # V: Noch in Arbeit.
    # Gibt es eine Möglichkeit aus data und change_entry eine getter und setter - Methode zu machen ( auch mit kwargs)
    # Wäre das überhaupt eine gute Idee?
    def change_entry(self, **kwargs):
        # V: Will hier irgendwie gerne das multiple Entries gleichzeitig geändert werden können
        # Weiß aber nicht, ob das so sinnvoll ist. Das sieht ehrlich gesagt nach ziemlichem Overkill aus xD
        # Habs jetzt erstmal so gemacht, dass ganze Zeilen / Spalten geändert werden können
        # oder mehrere Entries in der selben Spalte / Zeile
        # Hier ist außerdem dasselbe Zeugs mit den kwargs wie in der __init__-Funktion
        if "data" not in kwargs:
            raise PolarDiagramException("No data present")
        data = kwargs["data"]

        if "true_wind_speed" in kwargs:
            true_wind_speed = kwargs["true_wind_speed"]
            if "true_wind_angle" in kwargs:
                true_wind_angle = kwargs["true_wind_angle"]
                if isinstance(true_wind_speed, list) and isinstance(true_wind_angle, list):
                    raise PolarDiagramException("Multiple lists given", type(true_wind_speed), type(true_wind_angle))

                elif not isinstance(true_wind_speed, list) and not isinstance(true_wind_angle, list):
                    if np.array(data).shape != ():
                        raise PolarDiagramException("Wrong shape", (), np.array(data).shape)
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
                    self._data[ind1, ind2] = data

                elif isinstance(true_wind_angle, list):
                    try:
                        ind = self._resolution_wind_speed.index(true_wind_speed)
                    except ValueError:
                        raise PolarDiagramException("Wind speed not in resolution", self._resolution_wind_speed,
                                                    true_wind_speed)

                    ind_list = [i for i in len(self._resolution_wind_angle)
                                if self._resolution_wind_angle[i] in true_wind_angle]
                    if len(ind_list) < len(true_wind_angle):
                        raise PolarDiagramException("Wind angle not in resolution", self._resolution_wind_angle,
                                                    true_wind_angle)

                    data = np.array(data)
                    if data.shape != (len(ind_list),):
                        raise PolarDiagramException("Wrong shape", (len(ind_list), ), data.shape)

                    self._data[ind_list, ind] = data

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
            # Alternative zu if-Statements?
            # try:
            #    self._data[ind, :]
            # except ValueError:
            #    raise PolarDiagramException("Wrong shape", (len(self._resolution_wind_speed),), data.shape)
            if data.shape != (len(self._resolution_wind_speed),):
                raise PolarDiagramException("Wrong shape", (len(self._resolution_wind_speed),), data.shape)

            self._data[ind, :] = data

        else:
            # V: Hier hab ich es jetzt so gemacht, dass wenn jemand keine Entries spezifiziert, geguckt wird, ob die
            # übergebenen Daten die richtige Größe haben ( genau die Größe von self._data ) und falls das so ist
            # self._data komplett überschrieben wird.
            # Weiß aber nicht ob das so eine gute Idee ist, oder ob man nicht
            # besser zumindest noch eine Warnung ausgibt?
            data = np.array(data)
            if data.shape != (len(self._resolution_wind_angle), len(self._resolution_wind_speed)):
                raise PolarDiagramException("Wrong shape",
                                            (len(self._resolution_wind_angle), len(self._resolution_wind_speed)),
                                            data.shape)

            self._data = data

    def get_slice_data(self, true_wind_speed):
        try:
            column = self._resolution_wind_speed.index(true_wind_speed)
            return self._data[:, column]
        except ValueError:
            raise PolarDiagramException("Wind speed not in resolution", self._resolution_wind_speed, true_wind_speed)

    def polar_plot_slice(self, true_wind_speed, **kwargs):
        boat_speed = self.get_slice_data(true_wind_speed)
        angles = np.deg2rad(self._resolution_wind_angle)
        # if all(boat_speed == np.zeros(len(self._resolution_wind_angle),)):
        #
        polar_plot = plt.subplot(1, 1, 1, projection='polar')
        polar_plot.set_theta_zero_location('N')
        polar_plot.set_theta_direction('clockwise')
        return polar_plot.plot(angles, boat_speed, **kwargs)

    def flat_plot_slice(self, true_wind_speed, **kwargs):
        boat_speed = self.get_slice_data(true_wind_speed)
        # if all(boat_speed == np.zeros(len(self._resolution_wind_angle), )):
        #    polardiagram_logger.info("The to be plotted slice has no non-zero data")

        flat_plot = plt.subplot(1, 1, 1)
        plt.xlabel("True Wind Angle")
        plt.ylabel("Boat Speed")
        return flat_plot.plot(self._resolution_wind_angle, boat_speed, **kwargs)

    def plot_3d(self):
        pass
        # V: Erstmal nur das Konzept einer Funktion, um eine 3d-Darstellung der Slices, beziehungsweise
        # auch, wenn möglich, der konvexen Hülle zu bekommen?
        # -> majavi verwenden?

    def plot_convex_hull_slice(self, true_wind_speed, **kwargs):
        slice_data = self.get_slice_data(true_wind_speed)
        angles = np.deg2rad(self._resolution_wind_angle)
        # if all(slice_data == np.zeros(self._resolution_wind_angle)):
        #
        vert = sorted(convex_hull_polar(slice_data.copy(), angles.copy()).vertices)
        wind_angles = []
        boat_speed = []
        for i in vert:
            wind_angles.append(angles[i])
            boat_speed.append(slice_data[i])
        convex_hull_plot = plt.subplot(1, 1, 1, projection='polar')
        convex_hull_plot.set_theta_zero_location('N')
        convex_hull_plot.set_theta_direction('clockwise')
        return convex_hull_plot.plot(wind_angles, boat_speed, **kwargs)

    def plot_convex_hull_3d(self, **kwargs):
        # V: Funktion zum Plotten der konvexen Hülle des 3d-Polardiagrams
        pass

    # V: Soll das hier überhaupt eine Interne Methode sein? Ist ja weniger Visualisierung, Konvertierung,
    # Erstellung und Interpretation von Polardiagrammen
    def find_cross_course(self, **kwargs):
        # V: Normalenvektoren von Facetten gebrauchen?
        pass

    def __str__(self):
        # V: Gibt eine Tabelle mit den Zeilen und Spaltenbezeichnungen
        # der Wind angle und Wind speed resolution.
        # Unschön wenn zu viele Spalten
        # Falls zu viele Spalten sind -> wird nur ein Teil der Spalten + Info ausgegeben?
        table = np.c_[self._resolution_wind_angle, self._data]
        tab = tabulate(table, headers=["TWA \\ TWS"] + self._resolution_wind_speed)
        return tab


class PolarDiagramCurve(PolarDiagram):
    # V: Noch in Arbeit
    def __init__(self, f, *params, **kwargs):
        # V: Hier noch mögliche Errorchecks?
        # self._resolution_wind_speed = ...
        # Wir sollten vielleicht eine Liste von Funktionen und Parametern abspeichern die zu bestimmten
        # Windgeschwindigkeiten passen, oder?
        self._f = f
        self._params = params

    # V: Noch in Arbeit
    def to_csv(self, csv_path):
        # V: Das Format der .csv-Dateien ist
        # PolarDiagramCurve
        # Wind speed resolution:
        # self._resolution_wind_speed
        # Function:
        # self._f
        # Parameters:
        # self._params
        with open(csv_path, 'w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"')
            csv_writer.writerow(["PolarDiagramCurve"])
            # csv_writer.writerow(["Wind speed resolution:"])
            # csv_writer.writerow(list(self._resolution_wind_speed))
            csv_writer.writerow(["Function:"])
            csv_writer.writerow(["""self._f"""])
            csv_writer.writerow(["Parameters:"])
            csv_writer.writerow(list(self._params))

    # V: Noch in Arbeit
    def polar_plot_slice(self, true_wind_speed, **kwargs):
        # polar_plot = plt.subplot(1,1,1, projection = 'polar')
        # polar_plot.set_theta_zero_location('N')
        # polar_plot.set_theta_direction('clockwise')
        # return polar_plot.plot(self._f,**kwargs)
        pass

    # V: Noch in Arbeit
    def flat_plot_slice(self, true_wind_speed, **kwargs):
        # flat_plot = plt.subplot(1,1,1)
        # plt.xlabel("True Wind Angle")
        # plt.ylabel("Boat Speed")
        # return flat_plot(self._f, **kwargs)
        pass

    # V: Noch in Arbeit
    def __str__(self):
        # V: Weiß nicht ob das so geprinted werden soll. Die andere Idee wäre das der Print ein Plot der Funktion ist?
        # Das man sozusagen die curve sieht, aber dafür bräuchte ich dann ja data?
        return f"Function {self._f} with optimal parameters {self._params}"


class PolarDiagramPointcloud(PolarDiagram):
    def __init__(self, data=np.array([[0, 0, 0]])):
        if len(data[0]) != 3:
            raise PolarDiagramException("Wrong shape", (len(data[0]), "x"), (3, "x"))
        self._data = np.array(data)

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

    @property
    def wind_speeds(self):
        return list(dict.fromkeys(self._data[:, 0]))

    @property
    def wind_angles(self):
        return list(dict.fromkeys(self._data[:, 1]))

    @property
    def boat_speed(self):
        return list(dict.fromkeys(self._data[:, 2]))

    @property
    def get_points(self):
        return self._data

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

        points = [[np.radians(point[1]), point[2]] for point in self._data if point[0] == true_wind_speed]
        # V: Warning if points = []
        wind_angles = []
        boat_speeds = []
        for point in points:
            wind_angles.append(point[0])
            boat_speeds.append(point[1])
        polar_plot = plt.subplot(1, 1, 1, projection='polar')
        polar_plot.set_theta_zero_location('N')
        polar_plot.set_theta_direction('clockwise')
        polar_plot.plot(wind_angles, boat_speeds, **kwargs)
        return polar_plot

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
        points = [[point[1], point[2]] for point in self._data if point[0] == true_wind_speed]
        # V: Warning if points = []
        wind_angles = []
        boat_speeds = []
        for point in points:
            wind_angles.append(point[0])
            boat_speeds.append(point[1])
        flat_plot = plt.subplot(1, 1, 1)
        flat_plot.plot(wind_angles, boat_speeds, **kwargs)
        plt.xlabel('True Wind Angle')
        plt.ylabel('Boat Speed')
        return flat_plot

    def plot_convex_hull_slice(self, true_wind_speed, **kwargs):
        pass

    def plot_3d(self):
        # V: Funktion zum 3d-plotten der Punktwolke.
        pass

    def __str__(self):
        # V: Gibt Tabelle mit 3 Spalten und allen Punkten zurück
        return tabulate(self._data, headers=["TWS", "TWA", "BSP"])
