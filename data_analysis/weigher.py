"""
Collection of various function to
determine weights of data points

Each function returns an array
containing the weights of
the corresponding data point
"""

# Author: Valentin F. Dannenberg / Ente


from abc import ABC, abstractmethod
import numpy as np

from exceptions import ProcessingException
from utils import convert_wind, polar_to_kartesian


class WeightedPoints:
    """
    """
    def __init__(self, points, weights=None,
                 weigher=None, tw=True):
        points = np.asarray(points)

        if len(points[0]) != 3:
            try:
                points = points.reshape(-1, 3)
            except ValueError:
                raise ProcessingException(
                    "points could not be broadcasted "
                    "to an array of shape (n,3)")

        w_dict = convert_wind(
            {"wind_speed": points[:, 0],
             "wind_angle": points[:, 1],
             "boat_speed": points[:, 2]},
            tw)

        points = np.column_stack(
            (w_dict["wind_speed"],
             w_dict["wind_angle"],
             points[:, 2]))
        self._points = points

        if weigher is None:
            weigher = cylindric_mean_weights

        if not isinstance(weigher, Weigher):
            raise ProcessingException("")

        if weights is None:
            self._weights = weigher.weigh(points)
        else:
            weights = np.asarray(weights)
            no_pts = len(points)

            if len(weights) != no_pts:
                try:
                    weights = weights.reshape(no_pts, )
                except ValueError:
                    raise ProcessingException(
                        f"weights could not be broadcasted"
                        f"to an array of shape ({no_pts}, )")

            self._weights = weights

    @property
    def points(self):
        return self._points.copy()

    @property
    def weights(self):
        return self._weights.copy()

    def __repr__(self):
        return f"""WeightedPoints(points={self.points},
        weights={self.weights})"""

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.points):
            self.index += 1
            return self.points[self.index], \
                self.weights[self.index]

        raise StopIteration


class Weigher(ABC):

    @abstractmethod
    def weigh(self, points):
        pass


def inverted_standard_deviation_weight(points, **w_func_kw):
    st_points = w_func_kw.get('st_points', 13)
    out = w_func_kw.get('out', 5)

    std_list = [[], [], []]
    weights = []

    for i in range(st_points, len(points)):
        std_list[0].append(points[i-st_points:i, 0].std())
        std_list[1].append(points[i-st_points:i, 1].std())
        std_list[2].append(points[i-st_points:i, 2].std())

    for i in range(3):
        mask_1 = std_list[i] >= np.percentile(std_list[i], out)
        mask_2 = std_list[i] >= np.percentile(std_list[i], 100-out)
        mask = np.logical_not(np.logical_or(mask_1, mask_2))
        weights.append([1 / std_list[i][j]**2 if mask[j] else 0
                        for j in range(len(std_list[i]))])

    sum_weights = np.array([
        (ws_w + wa_w + bsp_w)/3 for ws_w, wa_w, bsp_w
        in zip(weights[0], weights[1], weights[2])])
    normed_weights = sum_weights / max(sum_weights)
    return np.concatenate([np.array([1] * st_points), normed_weights])


def cubiod_members_weights(points, **w_func_kw):
    radius = w_func_kw.get('radius', 1)
    ws_weight = w_func_kw.get('ws_weight', 1)
    weights = [0] * len(points)

    for i, point in enumerate(points):
        mask_WS = np.abs(points[:, 0] - point[0]) <= ws_weight
        # Hier nicht Degree sondern Radians?
        # Kartesische Koordinaten?
        mask_R = np.linalg.norm(
            polar_to_kartesian(points[:, 1:] - point[1:]),
            axis=1) <= radius
        weights[i] = len(points[np.logical_and(mask_R, mask_WS)]) - 1

    weights = np.array(weights)
    # Andere Normierungen?
    # weights = weights / max(weights)
    weights = len(points) * weights / sum(weights)
    return weights


def cylindric_mean_weights(points, **w_func_kw):
    weights = [0] * len(points)
    ws_weight = w_func_kw.get('ws_weight', 1)
    wa_weight = w_func_kw.get('wa_weight', 1)

    for i, point in enumerate(points):
        mask_WS = np.abs(points[:, 0] - point[0]) <= ws_weight
        mask_WA = np.abs(points[:, 1] - point[1]) <= wa_weight
        cylinder = points[np.logical_and(mask_WS, mask_WA)][:, 2]
        std = np.std(cylinder) or 1
        mean = np.mean(cylinder) or 0
        weights[i] = np.abs(mean - point[2]) / std

    weights = np.array(weights)
    weights = weights / max(weights)
    # weights = len(points) * weights / sum(weights)
    return weights
