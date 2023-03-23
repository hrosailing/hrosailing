import unittest
import matplotlib.pyplot as plt

import hrosailing.plotting.projections
from hrosailing.polardiagram import PolarDiagramTable
from tests.test_plotting.image_testcase import ImageTestcase

class TestHROPolar(ImageTestcase):
    def test_prepare_plot(self):
        #Execution Test
        ax = plt.subplot(projection="hro polar")
        pd = PolarDiagramTable([1, 2, 3], [1,2,3], [[0,1,2], [1,2,3], [2,3,4]])
        ax._prepare_plot((pd,), [1,2,3], 20, ("black", "white"), True, {"location": "left"}, marker="H")