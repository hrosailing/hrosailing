import unittest
import io

import numpy as np

import matplotlib.pyplot as plt

from hrosailing.plotting.projections import _plot


class PlotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.slices = [
            np.array([
                [1, 1, 1, 1, 1, 1], [0, 45, 90, 180, 270, 315], [1, 2, 2, 2, 2, 2]
            ]),
            np.array([
                [2, 2, 2, 2, 2, 2], [0, 45, 90, 180, 270, 315], [10, 1, 2, 2, 2, 1]
            ])
        ]
        self.wa_rad = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 7*np.pi/4]
        self.info = [
            ["A", "A", "B", "B", "B"],
            ["A", "B", "A", "B", "A"]
        ]

    def plot_test(self):
        #Input/Output Test with different parameters

        ax = plt.subplot(projection="hro polar")
