# pylint: disable-all
import unittest

import matplotlib.pyplot as plt
import numpy as np

from hrosailing.plotting.projections import plot_flat
from hrosailing.polardiagram import PolarDiagramTable
from tests.test_plotting.image_testcase import ImageTestcase


class TestPlotFlat(unittest.TestCase):
    def test_polar_diagram_plot(self):
        # Execution test
        keywords = {"marker": "H", "linestyle": "--"}
        pd = PolarDiagramTable(
            [1, 2, 3], [0, 90, 180], [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        )
        plot_flat(pd, **keywords)

    def test_regular_plot(self):
        # Execution test
        plot_flat([1, 2, 3], [1, 2, 3])
