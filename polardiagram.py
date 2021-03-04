import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import convex hull

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

# V: Algorithmus zum Berechnen der konvexen Hülle einer 2d-Punkt-Wolke.
# Gibt Liste der Ecken aus
# Muss aber noch für die Zwecke hier bearbeitet werden -> Ich denke an eine interne Konvertierung von
# Polar- zu kartesischen Koordinaten, die sich aber nicht auf die Punkte auswirken soll...
#def convex_hull_2d(point_cloud):
#    point_cloud = sorted(set(point_cloud))
#    if len(point_cloud) <= 1:
#        return point_cloud
    
#    def cross(o, a, b):
#        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
#    lower_hull_list = []
#    for point in point_cloud:
#        while len(lower_hull_list) >= 2 and cross(lower_hull_list[-2], lower_hull_list[-1],point) <= 0:
#            lower_hull_list.pop()
#        lower_hull_list.append(point)
        
#    upper_hull_list = []
#    for point in reversed(point_cloud):
#        while len(upper_hull_list) >= 2 and cross(upper_hull_list[-2], lower_hull_list[-1], point) <= 0:
#            upper_hull_list.pop()
#        upper_hull_list.append(point)
    
#    return lower_hull_list[:-1] + upper_hull_list[:-1]
    


def from_file(path):



def convert_apparent_wind_speed_to_true(data):
    #true_wind_speed = apparent_wind_speed * sin(true_wind_angle) / sin(apparent_wind_angle)
    
#def convert_apparent_wind_angle_to_true(data):
    


class PolarDiagramTable:
    def __init__(self,data = np.zeros((72,20))):
        self.data = data
        #self.plot = None
   
    def _convert_to_table(self):
        return self.data
        
    def _plot_slice(self,true_wind_speed):
        true_wind_angles = [np.radians(a) for a in np.arange(0,360,5)]
        # V: Hier müsste vielleicht noch nen Error abgefangen werden, wenn true_wind_speed ungerade ist
        # oder man macht die Tabelle so, dass sie true_wind_speed in 1er-Schritten dokumentiert,
        # obwohl sich wahrscheinlich bei einem Knoten Unterschied nicht viel ändert?
        boat_speed = self.data[:,true_wind_speed/2]
        plot_diagram = plt.subplot(1,1,1, projection = 'polar')
        # V: Polarplot hat standardmäßig 0° nach Osten und geht gegen den Uhrzeigersinn
        # Setzen 0° auf Norden und gehen im Uhrzeigersinn.
        plot_diagram.set_theta_zero_location('N')
        plot_diagram.set_theta_direction('clockwise')
        plot_diagram.plot(true_wind_angles, boat_speed, 'ro-', ms = 1, lw = 0.75)
        plt.show()
        # V: Man könnte hier vielleicht zusätzlich den Plot als eine Eigenschaft der Klasse sehen
        # das müsste man dann oben noch setzen, sodass man dann mit .plot oder so darauf zugreifen kann.
        # Dabei müsste man aber auch wieder die Windgeschwindigkeit welche man geplottet haben möchte noch
        # angeben, also eher sowas wie .plot(true_wind_speed).
        # Das könnte vielleicht Aufwand sparen, wenn wir das ganze nicht mehr nur "diskret" plotten wollen,
        # sondern durch die Tabelle eine Art Approximation an das "kontinuierliche" Polardiagram kriegen wollen,
        # was ja möglicherweise immer dauern kann, sowas zu plotten.
        # Ich hab aber auch keine Ahnung, ob sowas möglich ist, weil man die Plots dafür ja irgendwie
        # speichern muss.
    
    def _plot_3d(self):
        # V: Erstmal nur das Konzept einer Funktion, um eine 3d-Darstellung der Slices, beziehungsweise
        # auch, wenn möglich, der konvexen Hülle zu bekommen?
    
    def _convex_hull_slice(self, true_wind_speed):
        # V: Hier soll die konvexe Hülle eines Slices des Polardiagrams berechnet werden -> für _find_cross_course
        # Kann vielleicht auch komplett rausgelassen werden und durch einfaches Aufrufen der convex_hull_2d - Funktion
        # erledigt werden? 
    
    
    # V: Funktion um zu einem gegebenen true_wind_speed und true_wind_angle/Richtung wo man hinwill, einen 
    # Kreuz-kurs zu berechnen
    def _find_cross_course(self,true_wind_speed,true_wind_angle):
        convex_hull = convex_hull_2d(self.data[true_wind_speed/2])
        
        # Schnittpunkt mit konvexer Hülle finden
        # Konvexkombination der Ecken berechnen
        # Daraus Kurs angeben
        
        
        
        
    
    
    
class PolarDiagramParam(PolarDiagramTable):
    # V: Hier hab ich absolut keine Ahnung wie man sowas abspeichern könnte....?
    # Außer vielleicht als plot..., aber da müsste man ja auch irgendwie eine Möglichkeit habe
    # die Daten daraus automatisch zu extrahieren, damit man das überhaupt in eine Tabelle
    # konvertieren kann
    def __init__(self):
        self.data =

    def _convert_to_table(self):
        # V: Hier sollen Interpolationsmethoden benutzt werden, um Daten an den
        # TWAs in jeweils 5er Schritten (0°,5°,10°,15°,......,355°) zu erhalten
        # und da Objekt dann zu einem PolarDiagramTable-Objekt zu konvertieren
        # V: --> Ergibt aber nur eine Spalte der Tabelle, da Windgeschwindigkeit "konstant"?



class PolarDiagramPointCloud(PolarDiagramTable):
    def __init__(self, data = [[0,0,0]]):
        # Ich bin noch unentschlossen, ob man hier nicht auf einfach ein np.array verwenden sollte,
        # ähnlich wie bei PolarDiagramTable, oder ob man bei beiden ein pandas Dataframe verwendet,
        # obwohl das bei der Table-Sache glaub ich Blödsinn ist?
        self.data = pd.Dataframe(data = data , columns = ['TWS', 'TWA', 'BSP'])
        # V: So wäre Punktwolke "3-dimensional"......
        # Muss man glaub ich aber auch so machen, da sonst Konvertierung zu Tabelle schwer wird? (siehe unten)
        
    
    def _convert_to_table(self):
        # V: Hier sollen Interpolationsmethoden benutzt werden, um Daten an den
        # TWAs in jeweils 5er Schritten (0°,5°,10°,15°,......,355°) zu erhalten
        # und das Objekt dann zu einem PolarDiagramTable-Objekt zu konvertieren?
        # V: --> Ergibt aber nur eine Spalte der Tabelle, da Windgeschwindigkeit "konstant"?
        
        
    def _convert_to_param(self):
        # V: Hier könnte man versuchen, aus der Punktwolke ( oder auch nur aus einem Slice der Punktwolke ) eine
        # parametrisierte Darstellung zu berechnen?
        
        
    
        
        
   
        
    
    
