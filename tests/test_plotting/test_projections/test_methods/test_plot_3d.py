import unittest

from hrosailing.plotting.projections import plot_3d
from hrosailing.polardiagram import PolarDiagramTable


class TestPlot3D(unittest.TestCase):
    def test_polar_diagram_plot(self):
        # Execution test
        keywords = {"shade": False}
        pd = PolarDiagramTable(
            [1, 2, 3], [0, 90, 180], [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        )
        plot_3d(pd, **keywords)

    def test_regular_plot(self):
        # Execution test
        plot_3d([1, 2, 3], [1, 2, 3], [1, 2, 3])
