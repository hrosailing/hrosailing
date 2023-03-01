import unittest
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images

from hrosailing.plotting.projections import _plot


def save_plot(path):
    plt.savefig(path)
    plt.close()


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
            ["A", "A", "B", "B", "B", "B"],
            ["A", "B", "A", "B", "A", "B"]
        ]

    def test_plot(self):
        #Input/Output Tests

        with self.subTest("regular plot"):
            #creating resulting plot
            ax = plt.subplot(projection="hro flat")
            _plot(ax, self.slices, None, False)
            save_plot("./result.png")

            #creating exspected plot
            plt.plot([0, 45, 90, 180, 270, 315], [1, 2, 2, 2, 2, 2])
            plt.plot([0, 45, 90, 180, 270, 315], [10, 1, 2, 2, 2, 1])
            save_plot("./expected.png")

            compare_images(
                "./expected.png",
                "./result.png",
                tol=5
            )

        with self.subTest("using info"):
            #creating resulting plot
            ax = plt.subplot(projection="hro flat")
            _plot(ax, self.slices, self.info, False)
            save_plot("./result.png")

            #creating exspected plot
            plt.plot([0, 45], [1, 2], color="C0")
            plt.plot([90, 180, 270, 315], [2, 2, 2, 2], color="C0")
            plt.plot([0, 90, 270], [10, 2, 2], color="C1")
            plt.plot([45, 180, 315], [1, 2, 1], color="C1")
            save_plot("./expected.png")

            compare_images(
                "./expected.png",
                "./result.png",
                tol=5
            )

    def tearDown(self) -> None:
        os.remove("./expected.png")
        os.remove("./result.png")