import numpy as np
import csv
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

# V: Algorithmus zum Berechnen der konvexen Hülle einer 2d-Punkt-Wolke (mit Punkten in Polarkoordinaten!)
# Muss noch bearbeitet werden
def convex_hull_2d(point_cloud):
    point_cloud = sorted(set(point_cloud))
    if len(point_cloud) <= 1:
        return point_cloud
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    def polar_to_kartesian(point):
        return [point[0] * np.cos(point[1]), point[0] * np.sin(point[1])]
        
    lower_hull_list = []
    for point in point_cloud:
        while len(lower_hull_list) >= 2 and cross(polar_to_kartesian(lower_hull_list[-2]),                                                                                         polar_to_kartesian(lower_hull_list[-1]),
                                                  polar_to_kartesian(point)) <= 0:
            lower_hull_list.pop()
        lower_hull_list.append(point)
        
    upper_hull_list = []
    for point in reversed(point_cloud):
        while len(upper_hull_list) >= 2 and cross(polar_to_kartesian(upper_hull_list[-2]),                                                                                         polar_to_kartesian(lower_hull_list[-1]), 
                                                  polar_to_kartesian(point)) <= 0:
            upper_hull_list.pop()
        upper_hull_list.append(point)
    
    return lower_hull_list[:-1] + upper_hull_list[:-1]
    


# V: Statt von einer csv-Datei zu lesen, kann man soweit ich gesehen habe auch gleich die NMEA-Datei als solche einlesen
# Dabei kann man wahrscheinlich auch besser kontrollieren, welche Werte man mitnehmen will
#def from_nmea(nmea_path):
    
    
    

#def from_pol(pol_path):
    
def convert_apparent_wind_angle_to_true(data):
    #true_wind_angle = arcsin(true_wind_speed * sin(apparent_wind_angle) / apparent_wind_speed)?
       
def convert_apparent_wind_speed_to_true(data):
    #true_wind_speed = apparent_wind_speed - speed_over_ground? (wenn course_over_ground = apparent_wind_direction?)
    



class PolarDiagramTable():
    def __init__(self,data = np.zeros((72,20))):
        self.__data = data
        #self.plot = None
    
    def set_data(self, data, true_wind_speed = None, true_wind_angle = None):
        if true_wind_speed == None and true_wind_angle == None:
            self.__data = data
        elif true_wind_speed == None:
            self.__data[true_wind_angle,:] = data
        elif true_wind_angle == None:
            self.__data[:,true_wind_speed] = data
        else:
            self.__data[true_wind_angle, true_wind_speed] = data
            
        
    
    def get_data(self):
        return self.__data
    
    
    def convert_to_table(self):
        return
        
    def plot_slice(self,true_wind_speed):
        true_wind_angles = [np.radians(a) for a in np.arange(0,360,5)]
        # V: Hier müsste vielleicht noch nen Error abgefangen werden, wenn true_wind_speed ungerade ist
        # oder man macht die Tabelle so, dass sie true_wind_speed in 1er-Schritten dokumentiert,
        # obwohl sich wahrscheinlich bei einem Knoten Unterschied nicht viel ändert?
        boat_speed = self.__data[:,true_wind_speed/2]
        plot_diagram = plt.subplot(1,1,1, projection = 'polar')
        # V: Polarplot hat standardmäßig 0° nach Osten und geht gegen den Uhrzeigersinn
        # Setzen 0° auf Norden und gehen im Uhrzeigersinn.
        plot_diagram.set_theta_zero_location('N')
        plot_diagram.set_theta_direction('clockwise')
        plot_diagram.plot(true_wind_angles, boat_speed, 'ro-', ms = 1, lw = 0.75)
        plt.show()
        # V: Man könnte hier vielleicht zusätzlich die Plots als eine Property der Klasse verwenden?
        # das müsste man dann oben noch setzen, sodass man dann mit .plot[...] oder so darauf zugreifen kann.
        # Ich hab aber auch keine Ahnung, ob sowas möglich ist, weil man die Plots dafür ja irgendwie
        # speichern muss?
    
    def plot_3d(self):
        # V: Erstmal nur das Konzept einer Funktion, um eine 3d-Darstellung der Slices, beziehungsweise
        # auch, wenn möglich, der konvexen Hülle zu bekommen?
    
    def convex_hull_slice(self, true_wind_speed):
        # V: Hier wird die konvexe Hülle eines Slices des Polardiagrams berechnet -> für find_cross_course
        # Kann vielleicht auch komplett rausgelassen werden und durch einfaches Aufrufen der convex_hull_2d - Funktion
        # erledigt werden, allerdings stellt die Funktion die Hülle auch dar (Plot)
        points = tuple([(self.__data[int(a/5), true_wind_speed], a) for a in np.arange(0,360,5)])
        convex_hull = convex_hull_2d(points.copy())
        true_wind_angles = [np.radians(point[1]) for point in convex_hull]
        boat_speed = [point[0] for point in convex_hull]
        plot_convex_hull = plt.subplot(1,1,1, projection = 'polar')
        plot_convex_hull.set_theta_zero_location('N')
        plot_convex_hull.set_theta_direction('clockwise')
        # V: Der Plot hat noch ein paar Macken .... 
        plot_convex_hull.plot(true_wind_angles, boat_speed, 'ro-', ms = 1, lw = 0.75)
        plt.show()
        return convex_hull
    
    def convex_hull(self):
        # V: Funktion zum Plotten der konvexen des 3d-Polardiagrams
 

    def find_cross_course(self,true_wind_speed,true_wind_angle):
        
    
    
    
class PolarDiagramParam(PolarDiagramTable):
    # V: Hier hab ich absolut keine Ahnung wie man sowas abspeichern könnte....? -> plot(s)?? Funktion(en)??
    def __init__(self, data = ...):
        self.__data =

    def convert_to_table(self):
        # V: Hier kann man wahrscheinlich an den jeweiligen Punkten dann ablesen und daraus
        # EINE Spalte von so einer Tabelle erhalten, es sei denn wir machen dass so, dass man
        # mehrere parametrisierte(?) Darstellungen hat?



class PolarDiagramPointCloud(PolarDiagramTable):
    def __init__(self, data = [[0,0,0]]):
        # Ich bin noch unentschlossen, ob man hier nicht auf einfach ein np.array verwenden sollte,
        # ähnlich wie bei PolarDiagramTable, oder ob man bei beiden ein pandas Dataframe verwendet,
        # obwohl das bei der Table-Sache glaub ich Blödsinn ist?
        self.__data = pd.Dataframe(data = data , columns = ['TWS', 'TWA', 'BSP'])
    
    def plot_3d(self):
        # V: Funktion zum 3d-plotten der Punktwolke.
        # Noch in Bearbeitung
        
    
    def convert_to_table(self):
        # V: Hier sollen Interpolationsmethoden/Machine Learning benutzt werden, um Daten an den
        # TWAs in jeweils 5er Schritten (0°,5°,10°,15°,......,355°) zu erhalten
        # und das Objekt dann zu einem PolarDiagramTable-Objekt zu konvertieren?
        # -> Forschung!
        # V: --> Ergibt aber nur eine Spalte der Tabelle, da Windgeschwindigkeit "konstant"?
        
        
    def convert_to_param(self):
        # V: Hier könnte man versuchen, aus der Punktwolke ( oder auch nur aus einem Slice der Punktwolke ) eine
        # parametrisierte Darstellung zu berechnen?
        # -> Forschung?
        
        
    
        
        
   
        
    
    