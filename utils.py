import numpy as np


def polar_to_kartesian(radians, angles):
    # V: Wandelt die eindimensionalen Arrays der Polarkoordinaten der Punktmenge
    #    in ein zweidimensionales Array mit den kartesischen Koordinaten der
    #    Punktmenge um.
    return np.column_stack((radians * np.cos(angles), radians * np.sin(angles)))

