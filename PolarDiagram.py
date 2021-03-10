import numpy as np
import csv
import logging
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
    # V: Konvertiert die gegebenen Punkte mit Polarkoordinaten zuerst in
    # vorausgesetzte kartesische Koordinaten und berechnet dann die Konvexe Hülle dieser.
    # Hier müssten wir eigentlich nur ConvexHull.vertices zurückgeben
    # weiß aber nicht ob das eine gute Idee ist.
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
    # 8 Punkte Ecken sind und nicht nur 7
    points = np.array([polar_to_kartesian(point) for point in points])

    return ConvexHull(points)

def to_csv(csv_path, obj):
    # V: Ich hab mich jetzt für die Extern + Interne Variante entschieden, wobei die Externe einfach nur die Internen
    # aufruft.....
    obj.to_csv(csv_path)

def from_csv(csv_path):
    with open(csv_path, "r", newline = '') as file:
        csv_reader = csv.reader(file, delimiter = ',', quotechar = '"')
        # V: Die csv-Files werden in den to_csv-Funktionen so geschrieben, dass die
        # erste Zeile den Typ (PolarDiagramTable, PolarDiagramCurve, PolarDiagramPointCloud) enthält
        # Dann wird hier anhand der ersten Zeile entschieden, was für ein PolarDiagram erstellt wird
        row1 = next(csv_reader)[0]
        if row1 == 'PolarDiagramTable':
            wind_speed_resolution = [eval(s) for s in next(csv_reader)]
            if len(wind_speed_resolution) == 1:
                wind_speed_resolution = wind_speed_resolution[0]
            wind_angle_resolution = [eval(a) for a in next(csv_reader)]
            if len(wind_angle_resolution) == 1:
                wind_angle_resolution = wind_angle_resolution[0]
            data = []
            for row in csv_reader:
                data.append([eval(entry) for entry in row])
            return PolarDiagramTable(data = np.array(data), wind_angle_resolution = wind_angle_resolution, wind_speed_resolution = wind_speed_resolution)

        elif row1 == 'PolarDiagramCurve':
            #V : Noch in Arbeit
            next(csv_reader)

            return PolarDiagramCurve()

        elif row1 == 'PolarDiagramPointCloud':
            next(csv_reader)
            data = []
            for row in csv_reader:
                data.append([eval(entry) for entry in row])
            return PolarDiagramPointCloud(data = np.array(data))

#def polar_diagram_from_data(, **kwargs):

class PolarDiagramException(Exception):
    # V: Custom Exceptions die noch erweitert werden, sollte ich auf weitere Sachen stoßen
    # Hab hier aber im Endeffekt die Struktur aus deiner QuadraticFormException-Klasse kopiert
    def __init__(self,type, *args):
        message_dict = {
            "Wrong dimension" : f"Expecting 2 dimensional array to be viewed as Polar Diagram Tableau,\n got {args[0]} dimensional array instead.",
            "Wrong resolution" : f"Expecting resolution of type 'list' or 'int',\n got resolution of type {args[0]} instead",
            "No Data present" : f"Expecting ,\n got  instead",
            "Wrong shape" : f"Expecting array with shape {args[0]},\n got array with shape {args[1]} instead",
            "Slice doesn't exist" : f"Expecting slice to be in {args[0]},\n got {args[1]} instead",
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


class PolarDiagramTable(PolarDiagram):
    def __init__(self, **kwargs):
        # V: Die "Auflösung" des Polardiagrams kann nun vom User selber bestimmt werden, d.h.
        # die User können entweder selber eine Liste von Windwinkeln und -geschwindigkeiten übergeben,
        # oder auch nur eine "Auflösungsgröße" oder die "Standardauflösung" verwenden
        if "wind_angle_resolution" in kwargs:
            wind_angle_resolution = kwargs["wind_angle_resolution"]
            if type(wind_angle_resolution) is list:
                self._resolution_wind_angle =wind_angle_resolution
            elif type(wind_angle_resolution) is int:
                self._resolution_wind_angle = list(np.arange(0,360,wind_angle_resolution))
            else:
                raise PolarDiagramException("Wrong resolution", type(wind_angle_resolution))
        else:
            self._resolution_wind_angle = list(np.arange(0,360,5))

        if "wind_speed_resolution" in kwargs:
            if type(wind_speed_resolution) is list:
                self._resolution_wind_speed = wind_speed_resolution
            elif type(wind_speed_resolution) is int:
                self._resolution_wind_speed = list(np.arange(2,42,wind_speed_resolution))
            else:
                raise PolarDiagramException("Wrong resolution", type(wind_speed_resolution))
        else:
            self._resolution_wind_speed = list(np.arange(2,42,2))

        if "data" in kwargs:
            data = kwargs["data"]
            if data.shape != (len(self._resolution_wind_angle), len(self._resolution_wind_speed)):
                raise PolarDiagramException("Wrong shape",(self._resolution_wind_angle, self._resolution_wind_speed), data.shape)
            if data.ndim != 2:
                raise PolarDiagramException("Wrong dimension", data.ndim)
            else:
                self._data = data
        else:
            self._data = np.zeros((len(self._resolution_wind_angle), len(self._resolution_wind_speed)))

    def to_csv(self,csv_path):
        # V: Das Format für PolarDiagramTables .csv-Datein ist
        # PolarDiagramTable
        # 2,4,6,8,10,......    <- True Wind Speed Resolution
        # 0,5,10,15,.......    <- True Wind Angle Resolution
        # ............        }
        # ............        } <- Boat Speed
        # ............        }
        with open(csv_path, "w", newline = '') as file:
            csv_writer = csv.writer(file, delimiter = ',', quotechar = '"')
            csv_writer.writerow(['PolarDiagramTable'])
            # V: Hier könnte man das vielleicht für bessere (menschliche) Lesbarkeit
            # der .csv-Datei in die 2 und 3 Zeile zuerst einen String
            # "True Wind Speed Resolution" bzw. "True Wind Angle Resolution"
            # einfügen, dann müsste man halt nur in der from_csv-Funktion etwas anpassen.
            # Das ist aber glaube ich alles soweit unnötig?
            csv_writer.writerow(self._resolution_wind_speed)
            csv_writer.writerow(self._resolution_wind_angle)
            for row in self._data:
                csv_writer.writerow(row)

    # V: Noch in Arbeit
    def change_entry(self, **kwargs):
        if not "data" in kwargs:
            raise PolardiagramException("No data given", )
        data = kwargs["data"]
        if "true_wind_speed" in kwargs:
            true_wind_speed = list(kwargs["true_wind_speed"])
            if "true_wind_angle" in kwargs:
                true_wind_angle = list(kwargs["true_wind_angle"])
                ind_list_speed = [i for i in range(len(true_wind_speed)) if true_wind_speed[i] in self._resolution_wind_speed]
                ind_list_angle = [i for i in range(len(true_wind_angle)) if true_wind_angle[i] in self._resolution_wind_angle]
                if data.shape != (len(ind_list_angle), len(ind_list_speed)):
                    raise PolarDiagramException("Wrong shape", (len(ind_list_angle), len(ind_list_speed)), data.shape)
                self._data[ind_list_angle,ind_list_speed] = data

        elif "true_wind_angle" in kwargs:

        else:
            if data.shape != (len(self._resolution_wind_angle), len(self._resolution_wind_speed)):
                raise PolarDiagramException("Wrong shape", (len(self._resolution_wind_angle), len(self._resolution_wind_speed)), data.shape)

    def get_slice_data(self,slice):
        try:
            column = self._resolution_wind_speed.index(slice)
            return self._data[:,column]
        except:
            raise PolarDiagramException("Slice doesn't exist", self._resolution_wind_speed, slice)

    def polar_plot_slice(self, slice, **kwargs):
        boat_speed = self.get_slice_data(slice)
        angles = [np.radians(a) for a in self._resolution_wind_angle]
        #if all(boat_speed == np.zeros(self._resolution_wind_angle,)) is True:
            # raise Exception("No data was available")
        polar_plot_diagram = plt.subplot(1, 1, 1, projection='polar')
        # V: Polarplot hat standardmäßig 0° nach Osten und geht gegen den Uhrzeigersinn
        # Setzen 0° auf Norden und gehen im Uhrzeigersinn.
        polar_plot_diagram.set_theta_zero_location('N')
        polar_plot_diagram.set_theta_direction('clockwise')
        # V: User können jetzt selber das Aussehen des Plots bestimmen
        polar_plot_diagram.plot(angles, boat_speed, **kwargs)
        # V: Gibt jetzt nur noch den Plot zurück -> man könnte nun also mehrere Plots übereinanderlegen
        return polar_plot_diagram

    def flat_plot_slice(self, slice, **kwargs):
        # V: Hier die gewünschte flat_plot-Funktion (mit Achsenbezeichnung! :) )
        boat_speed = self.get_slice_data(slice)
        flat_plot_diagram = plt.subplot(1,1,1)
        flat_plot_diagram.plot(self._resolution_wind_angle, boat_speed,**kwargs)
        plt.xlabel('True Wind Angle')
        plt.ylabel('Boat Speed')
        return flat_plot_diagram

    def plot_3d(self):
        # V: Erstmal nur das Konzept einer Funktion, um eine 3d-Darstellung der Slices, beziehungsweise
        # auch, wenn möglich, der konvexen Hülle zu bekommen?
        # -> majavi verwenden?

    def plot_convex_hull_slice(self, slice, **kwargs):
        # V: Hier die überarbeitete plot_convex_hull_slice-Funktion.
        # Hat noch Macken.
        boat_speed = self.get_slice_data(slice)
        points =[[boat_speed[i], self._resolution_wind_angle[i]] for i in range(len(boat_speed))]
        # V: Ich übergebe hier nur eine Kopie der Punkte, damit man nicht später nochmal von
        # kartesischen in Polarkoordinaten umrechnen muss
        convex_hull = convex_hull_polar(points.copy())
        # V: Von der convex_hull benötigen wir hier eigentlich nur die Indizes des vertices,
        # wie oben also angemerkt ist es möglicherweise besser einfach nur diese zurückzugeben?
        vert = sorted(convex_hull.vertices)
        # V: Wir sortieren hier die Indizies, da ja die Wind-Winkel der Größe nach geordnet sind
        # und andernfalls der Plot etwas seltsam aussieht
        angles = []
        speed = []
        for i in vert:
            angles.append(np.radians(points[i][1]))
            speed.append(points[i][0])
        convex_points.sort(key = lambda point: point[0])
        convex_hull_plot = plt.subplot(1,1,1, projection = 'polar')
        convex_hull_plot.set_theta_zero_location('N')
        convex_hull_plot.set_theta_direction('clockwise')
        convex_hull_plot.plot(points, **kwargs)
        return convex_hull_plot


    def plot_convex_hull_3d(self, **kwargs):
        # V: Funktion zum Plotten der konvexen Hülle des 3d-Polardiagrams

    def find_cross_course(self,**kwargs):
        # V: Normalenvektoren von Facetten gebrauchen?

    def __str__(self):
        # V: Wenn man jetzt ein PolarDiagramTable-Objekt printen will, sollte es eine
        # (bisher noch zu große um gut dargestellt zu werden) "schöne" Tabelle ausgeben.
        table = np.c_[self._resolution_wind_angle, self._data]
        tab = tabulate(table, headers=["TWA \ TWS"] + self._resolution_wind_speed)
        return tab

    #def __iter__(self):


class PolarDiagramCurve(PolarDiagram):
    def __init__(self,data, f, *params):
        # V: Noch in Arbeit
        # Ich bin mir hier nicht so wirklich sicher, ob wir hier f(data, params ) (also ein np.nd_array) speichern wollen
        # oder mehr so eine lambda-Funktion?
        self._f = f
        self._params = params
        self._data = data

    def to_csv(self,csv_path):
        # V: Noch in Arbeit
        with open(csv_path, "w", newline = '') as file:
            csv_writer = csv.writer(file, delimiter = ',', quotechar = '"')
            csv_writer.writerow(['PolarDiagramCurve'])
            csv_writer.writerow(['Data'])
            for row in self._data:
                csv_writer.writerow(row)
            csv_writer.writerow(['Function'] + [self._f])
            csv_writer.writerow(['Parameters:'] + self._params)

    # V: Noch in Arbeit
    def __str__(self):
        pass


    #def __iter__(self):

class PolarDiagramPointCloud(PolarDiagram):
    def __init__(self, data = np.array([[0,0,0]])):
        self._data = data

    def to_csv(self,csv_path):
        with open(csv_path, "w", newline='') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"')
            csv_writer.writerow(['PolarDiagramPointCloud'])
            # V: die Zeile ['True Wind Speed', 'True Wind Angle', 'Boat Speed'] dient vorallendingen
            # der besseren Lesbarkeit für Menschen.
            # Bei der Read-Funktion wird diese aber übersprungen
            # -> Problem wenn nicht vorhanden in User-supplied Datei -> Wie lösen?
            csv_writer.writerow(['True Wind Speed', 'True Wind Angle', 'Boat Speed'])
            for row in self._data:
                csv_writer.writerow(row)

    def plot_3d(self):
        # V: Funktion zum 3d-plotten der Punktwolke.

    # V: Noch in Arbeit
    def __str__(self):
        pass


    #def __iter__(self):
