import csv
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tabulate import tabulate
from scipy.spatial import ConvexHull


# V: Ich werde noch schauen müssen, wo ich alles logging verwende. Kommt aber noch
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

# V: Noch in Arbeit
def convert(object, type):
    pass


def convex_hull_polar(points):
    def polar_to_kartesian(point):
        return [point[0] * np.cos(point[1]), point[0] * np.sin(point[1])]
    # V: Die Umrechnung hier macht noch in sofern Probleme, dass sie ja nicht 100% exakt ist,
    # und daher die Koordinaten der Punkte sich etwas "ändern" und die konvexe Hülle dann etwas
    # falsch ist
    # Hier ein Beispiel zum Nachgucken:
    # import numpy as np
    # from scipy.spatial import ConvexHull
    #
    # def polar_to_kartesian(point):
    #     return [point[0] * np.cos(point[1]), point[0] * np.sin(point[1])]
    #
    # boat_speed = [4.82 5.11 5.35 5.31 4.98 4.72 4.27 3.78]
    # wind_angle = [52,60,75,90,110,120,135,150]
    # points = [[boat_speed[i], wind_angle[i] for i in range(len(boat_speed))]
    # converted_points = np.array([polar_to_kartesian(point) for point in points])
    # print(ConvexHull(converted_points).vertices)
    # -> [0,1,2,3,4,5,7] (Index 6 fehlt hier offenbar)
    # Wenn man sich dieses Slice aber plotten lässt, erkennt man, dass eigentlich alle
    # 8 Punkte Ecken sind
    points = np.array([polar_to_kartesian(point) for point in points])

    return ConvexHull(points)

def to_csv(csv_path, obj):
    # V: Funktion ruft einfach die jeweils interne Funktion des Objekts auf -> Nutzerfreundlichkeit?
    obj.to_csv(csv_path)


# V: Hauptsächlich nicht benutzt. .csv-Datein dienen eher der menschlichen Lesbarkeit. Für Speichern und Laden von Daten
# eher pickling und cori_cycling benutzen
def from_csv(csv_path):
    with open(csv_path, "r", newline = '') as file:
        csv_reader = csv.reader(file, delimiter = ',', quotechar = '"')
        # V: Die csv-Files werden in den to_csv-Funktionen so geschrieben, dass die
        # erste Zeile den Typ (PolarDiagramTable, PolarDiagramPointCloud) enthält
        # Dann wird hier anhand der ersten Zeile entschieden, was für ein PolarDiagram erstellt wird
        # Kann nur PolarDiagramTable- und PolarDiagramPointCloud-Objekte erstellen
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
            return PolarDiagramTable(data = np.array(data), wind_angle_resolution = wind_angle_resolution,
                                     wind_speed_resolution = wind_speed_resolution)
        elif row1 == 'PolarDiagramPointCloud':
            next(csv_reader)
            data = []
            for row in csv_reader:
                data.append([eval(entry) for entry in row])
            return PolarDiagramPointCloud(data = np.array(data))


def pickling(pkl_path, obj):
    #V: Funktion ruft einfach die jeweils interne Funktion des Objekts auf -> Nutzerfreundlichkeit?
    obj.pickling(pkl_path)

# V: Ich weiß der Name ist etwas verwirrend, aber das ist so ziemlich der umgekehrte pickling-Prozess,
# daher macht es irgendwo Sinn, finde ich :D
def cori_cycling(pkl_path):
    with open(pkl_path, "rb") as file:
        return pickle.load(file)


#def polar_diagram_from_data(, **kwargs):

class PolarDiagramException(Exception):
    # V: Custom Exceptions die noch erweitert werden, sollte ich auf weitere Sachen stoßen
    # Hab hier aber im Endeffekt die Struktur aus deiner QuadraticFormException-Klasse kopiert
    def __init__(self,type, *args):
        message_dict = {
            "Wrong dimension" : "Expecting 2 dimensional array to be viewed as Polar Diagram Tableau," +
                                f"\n got {args[0]} dimensional array instead.",
            "Wrong resolution" : "Expecting resolution of type 'list' or 'int'," +
                                 f"\n got resolution of type {args[0]} instead",
            "No data present" : f"Expecting to get new data to update",
            "Wrong shape" : f"Expecting array with shape {args[0]},\n got array with shape {args[1]} instead",
            "Wind speed not in resolution" : f"Expecting wind speed to be in {args[0]},\n got {args[1]} instead",
            "No entry specified" : f"Expecting to get an entry to update",
            "Wind angle not in resolution" : f"Expecting wind angle to be in {args[0]},\n got {args[1]} instead"
        }
        if type in message_dict:
            super().__init__(message_dict[type])
        else:
            super().__init__(type)

# V: Noch in Arbeit
class PolarDiagram(ABC):

    @abstractmethod
    def __str__(self):
        pass

    #@abstractmethod
    #def __iter__(self):
    #    pass

    @abstractmethod
    def to_csv(self, csv_path):
        pass

    # V: Da der Fermentationsprozess in jeder Klasse derselbe ist, hab ich das einfach in die Baseclass geschrieben.
    # Ich hoffe das geht so?
    def pickling(self, pkl_path):
        with open(pkl_path, "wb") as file:
            pickle.dump(self, file)

class PolarDiagramTable(PolarDiagram):
    def __init__(self, **kwargs):
        # V : Es werden 3 Keyword-Arguments unterstützt: "wind angle resolution"
        #                                                "wind speed resolution"
        #                                                "data"
        # Gibt es eine einfache Möglichkeit hier auch Sachen wie "speed resolution", "angle resolution" "boat speeds" oder
        # Abkürzungen wie "was" "wsr" "d" etc. zu unterstüzen?
        if "wind_angle_resolution" in kwargs:
            wind_angle_resolution = kwargs["wind_angle_resolution"]
            if isinstance(wind_angle_resolution, list):
                self._resolution_wind_angle = wind_angle_resolution
            elif isinstance(wind_angle_resolution, int):
                self._resolution_wind_angle = list(np.arange(0,360,wind_angle_resolution))
            else:
                raise PolarDiagramException("Wrong resolution", type(wind_angle_resolution))
        else:
            self._resolution_wind_angle = list(np.arange(0,360,5))

        if "wind_speed_resolution" in kwargs:
            wind_speed_resolution = kwargs["wind_speed_resolution"]
            if isinstance(wind_speed_resolution, list):
                self._resolution_wind_speed = wind_speed_resolution
            elif isinstance(wind_speed_resolution, int):
                self._resolution_wind_speed = list(np.arange(2,42,wind_speed_resolution))
            else:
                raise PolarDiagramException("Wrong resolution", type(wind_speed_resolution))
        else:
            self._resolution_wind_speed = list(np.arange(2,42,2))

        if "data" in kwargs:
            data = kwargs["data"]
            if data.ndim != 2:
                raise PolarDiagramException("Wrong dimension", data.ndim)
            if data.shape != (len(self._resolution_wind_angle), len(self._resolution_wind_speed)):
                raise PolarDiagramException("Wrong shape",
                                            (len(self._resolution_wind_angle), len(self._resolution_wind_speed)),
                                            data.shape)

            self._data = data
        else:
            self._data = np.zeros((len(self._resolution_wind_angle), len(self._resolution_wind_speed)))

    def to_csv(self,csv_path):
        # V: Das Format für PolarDiagramTables .csv-Dateien ist
        # PolarDiagramTable
        # Wind speed resolution:
        # self._resolution_wind_speed
        # Wind angle resolution:
        # self._resolution_wind_angle
        # Boat speeds:
        # self._data
        with open(csv_path, "w", newline = '') as file:
            csv_writer = csv.writer(file, delimiter = ',', quotechar = '"')
            csv_writer.writerow(['PolarDiagramTable'])
            csv_writer.writerow(["Wind speed resolution:"])
            csv_writer.writerow(list(self._resolution_wind_speed))
            csv_writer.writerow(["Wind angle resolution:"])
            csv_writer.writerow(list(self._resolution_wind_angle))
            csv_writer.writerow(["Boat speeds:"])
            for row in self._data:
                csv_writer.writerow(row)

    # V: Noch in Arbeit.
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
                    except:
                        raise PolarDiagramException("Wind angle not in resolution", self._resolution_wind_angle,
                                                    true_wind_angle)
                    try:
                        ind2 = self._resolution_wind_speed.index(true_wind_speed)
                    except:
                        raise PolarDiagramException("Wind speed not in resolution", self._resolution_wind_speed,
                                                    true_wind_speed)
                    self._data[ind1, ind2] = data

                elif isinstance(true_wind_angle, list):
                    try:
                        ind = self._resolution_wind_speed.index(true_wind_speed)
                    except:
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
                    except:
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
                if isinstance(true_wind_speed,list):
                    if len(true_wind_speed) > 1:
                        raise PolarDiagramException("")
                    else:
                        true_wind_speed = true_wind_speed[0]
                try:
                    ind = self._resolution_wind_speed.index(true_wind_speed)
                except:
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
                    raise PolarDiagramException("")
                else:
                    true_wind_angle = true_wind_angle[0]

            try:
                ind = self._resolution_wind_angle.index(true_wind_angle)
            except:
                raise PolarDiagramException("Wind angle not in resolution", self._resolution_wind_angle, true_wind_angle)

            data = np.array(data)
            if data.shape != (len(self._resolution_wind_speed),):
                raise PolarDiagramException("Wrong shape", (len(self._resolution_wind_speed),), data.shape)

            self._data[ind,:] = data

        else:
            raise PolarDiagramException("No entry specified")



    def get_slice_data(self,slice):
        try:
            column = self._resolution_wind_speed.index(slice)
            return self._data[:,column]
        except:
            raise PolarDiagramException("Wind speed not in resolution", self._resolution_wind_speed, slice)

    def polar_plot_slice(self, slice, **kwargs):
        boat_speed = self.get_slice_data(slice)
        angles = [np.radians(a) for a in self._resolution_wind_angle]
        #if all(boat_speed == np.zeros(self._resolution_wind_angle,)) is True:
        #    raise Exception("No data was available")
        polar_plot = plt.subplot(1, 1, 1, projection='polar')
        polar_plot.set_theta_zero_location('N')
        polar_plot.set_theta_direction('clockwise')
        polar_plot.plot(angles, boat_speed, **kwargs)
        return polar_plot

    def flat_plot_slice(self, slice, **kwargs):
        boat_speed = self.get_slice_data(slice)
        #if all(boat_speed == np.zeros(self._resolution_wind_angle,)) is True:
        #    raise Exception("No data was available")
        flat_plot = plt.subplot(1,1,1)
        flat_plot.plot(self._resolution_wind_angle, boat_speed,**kwargs)
        plt.xlabel('True Wind Angle')
        plt.ylabel('Boat Speed')
        return flat_plot

    def plot_3d(self):
        # V: Erstmal nur das Konzept einer Funktion, um eine 3d-Darstellung der Slices, beziehungsweise
        # auch, wenn möglich, der konvexen Hülle zu bekommen?
        # -> majavi verwenden?

    def plot_convex_hull_slice(self, slice, **kwargs):
        # V: Hat noch Macken. Siehe convex_hull_polar-Funktion
        slice_data = self.get_slice_data(slice)
        points =[[slice_data[i], self._resolution_wind_angle[i]] for i in range(len(slice_data))]
        vert = sorted(convex_hull_polar(points.copy()).vertices)
        wind_angles = []
        boat_speed = []
        for i in vert:
            wind_angles.append(np.radians(points[i][1]))
            boat_speed.append(points[i][0])
        convex_points.sort(key = lambda point: point[0])
        convex_hull_plot = plt.subplot(1,1,1, projection = 'polar')
        convex_hull_plot.set_theta_zero_location('N')
        convex_hull_plot.set_theta_direction('clockwise')
        convex_hull_plot.plot(wind_angles, boat_speed, **kwargs)
        return convex_hull_plot


    def plot_convex_hull_3d(self, **kwargs):
        # V: Funktion zum Plotten der konvexen Hülle des 3d-Polardiagrams

    def find_cross_course(self,**kwargs):
        # V: Normalenvektoren von Facetten gebrauchen?

    def __str__(self):
        # V: Gibt eine Tabelle mit den Zeilen und Spaltenbezeichnungen
        # der Wind angle und Wind speed resolution.
        # Sieht unschön aus, wenn zu viele Spalten da sind
        table = np.c_[self._resolution_wind_angle, self._data]
        tab = tabulate(table, headers=["TWA \ TWS"] + self._resolution_wind_speed)
        return tab

    #def __iter__(self):


class PolarDiagramCurve(PolarDiagram):
    # V: Noch in Arbeit
    def __init__(self, f, *params):
        # V: Hier noch mögliche Errorchecks?
        self._f = f
        self._params = params

    # V: Noch in Arbeit
    def to_csv(self,csv_path):
        # V: Das Format der .csv-Dateien ist
        # PolarDiagramCurve
        # Function:
        # self._f
        # Parameters:
        # self._params
        with open(csv_path, "w", newline = '') as file:
            csv_writer = csv.writer(file, delimiter = ',', quotechar = '"')
            csv_writer.writerow(['PolarDiagramCurve'])
            csv_writer.writerow(['Function:'])
            csv_writer.writerow(["""self._f"""])
            csv_writer.writerow(['Parameters:'])
            csv_writer.writerow(list(self._params))

    # V: Noch in Arbeit
    def __str__(self):
        # V: Weiß nicht ob das so geprinted werden soll. Die andere Idee wäre das der Print ein Plot der Funktion ist?
        # Das man sozusagen die curve sieht, aber dafür bräuchte ich dann ja data?
        return f"Function {self._f} with optimal parameters {self._params}"

    #def __iter__(self):

class PolarDiagramPointCloud(PolarDiagram):
    def __init__(self, data = np.array([[0,0,0]])):
        if len(data[0]) != 3:
            raise PolarDiagramException("Wrong shape", (len(data[0], "x"), (3, "x")))
        self._data = data

    def to_csv(self,csv_path):
        # V: Das Format der .cvs-Dateien ist
        # PolarDiagramPointCloud
        # True Wind Speed: ,True Wind Angle: , Boat Speed:
        # self._data
        with open(csv_path, "w", newline='') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"')
            csv_writer.writerow(['PolarDiagramPointCloud'])
            csv_writer.writerow(['True Wind Speed: ', 'True Wind Angle: ', 'Boat Speed: '])
            for row in self._data:
                csv_writer.writerow(row)

    def polar_plot_slice(self, slice, **kwargs):
        # V: Hier fange ich ab, das der User möglicherweise linestyle bzw. linewidth in den keywords hat.
        # Damit man auch wirklich eine PUNKT-Wolke plottet.
        # Sollte so funktionieren, sicher bin ich mir nicht.
        if "linestyle" in kwargs or "ls" in kwargs:
            try:
                del kwargs["linestyle"]
            except:
                del kwargs["ls"]

        if "linewidth" in kwargs or "lw" in kwargs:
            try:
                del kwargs["linewidth"]
            except:
                del kwargs["lw"]

        points = [[np.radians(point[1]), point[2]] for point in self._data if point[0] == slice]
        # V: Warning if points = []
        wind_angles = []
        boat_speeds = []
        for point in points:
            wind_angles.append(point[0])
            boat_speeds.append(point[1])
        polar_plot = plt.subplot(1,1,1, projection = 'polar')
        polar_plot.set_theta_zero_location('N')
        polar_plot.set_theta_direction('clockwise')
        polar_plot.plot(wind_angles, boat_speeds, **kwargs)
        return polar_plot

    def flat_plot_slice(self, slice, **kwargs):
        # V: Hier fange ich ab, das der User möglicherweise linestyle bzw. linewidth in den keywords hat
        # Damit man auch wirklich eine PUNKT-Wolke plottet
        # Sollte so funktionieren, sicher bin ich mir nicht
        if "linestyle" in kwargs or "ls" in kwargs:
            try:
                del kwargs["linestyle"]
            except:
                del kwargs["ls"]

        if "linewidth" in kwargs or "lw" in kwargs:
            try:
                del kwargs["linewidth"]
            except:
                del kwargs["lw"]
        points = [[point[1], point[2]] for point in self._data if point[0] == slice]
        # V: Warning if points = []
        wind_angles = []
        boat_speeds = []
        for point in points:
            wind_angles.append(point[0])
            boat_speeds.append(point[1])
        flat_plot = plt.subplot(1,1,1)
        flat_plot.plot(wind_angles, boat_speeds,**kwargs)
        plt.xlabel('True Wind Angle')
        plt.ylabel('Boat Speed')
        return flat_plot


   # def plot_3d(self):
        # V: Funktion zum 3d-plotten der Punktwolke.

    def __str__(self):
        # V: Gibt Tabelle mit 3 Spalten und allen Punkten zurück
        return tabulate(self._data, headers = ["TWS", "TWA", "BSP"])


    #def __iter__(self):
