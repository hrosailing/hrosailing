import unittest

import matplotlib.pyplot as plt
import numpy as np

import hrosailing.plotting
from hrosailing.polardiagram import PolarDiagramTable


class TestAxes3D(unittest.TestCase):
    def setUp(self) -> None:
        self.pd = PolarDiagramTable(
            [1, 2, 3], [0, 90, 180], [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        )
        self.wind = (np.array([1, 2, 3]), np.array([0, 90, 180]))

    def test_prepare_points(self):
        # Input/Output
        ax = plt.subplot(projection="hro 3d")
        resx, resy, resws = ax._prepare_points(self.pd, wind=self.wind)
        np.testing.assert_array_almost_equal(
            resx, np.array([0, 0, 0, 1, 2, 3, 0, 0, 0])
        )
        np.testing.assert_array_almost_equal(
            resy, np.array([0, 1, 2, 0, 0, 0, -2, -3, -4])
        )
        np.testing.assert_array_equal(
            resws, np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        )

    def test_plot3d(self):
        # Execution Test
        ax = plt.subplot(projection="hro 3d")
        ax._plot3d(
            np.array([1, 2, 3, 4]),
            np.array([2, 3, 4, 5]),
            np.array([3, 4, 5, 6]),
            ("green", "blue"),
            marker="H",
        )

    def test_plot_polar_diagram(self):
        # Execution Test
        ax = plt.subplot(projection="hro 3d")
        ax.plot(self.pd, wind=self.wind, colors=("green", "blue"), shade=False)

    def test_plot_other(self):
        # Execution Test
        ax = plt.subplot(projection="hro 3d")
        ax.plot([1, 2, 3], [1, 2, 3], [1, 2, 3], color="blue")

    def test_plot_surface_polar_diagram(self):
        # Execution Test
        ax = plt.subplot(projection="hro 3d")
        ax.plot_surface(
            self.pd, wind=self.wind, colors=("green", "blue"), shade=False
        )

    def test_plot_surface_other(self):
        # Execution Test
        ax = plt.subplot(projection="hro 3d")
        ax.plot_surface(
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]),
            linewidth=0,
        )

    def test_scatter_polar_diagram(self):
        # Execution Test
        ax = plt.subplot(projection="hro 3d")
        ax.scatter(
            self.pd, wind=self.wind, colors=("green", "blue"), marker="H"
        )

    def test_scatter_other(self):
        # Execution Test
        ax = plt.subplot(projection="hro 3d")
        ax.scatter([1, 2, 3], [1, 2, 3], marker="H")
