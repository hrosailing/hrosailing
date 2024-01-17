# pylint: disable-all

import unittest

import matplotlib.pyplot as plt

from hrosailing.plotting.projections import _set_3d_axis_labels


class TestSet3DAxisLabels(unittest.TestCase):
    def test_regular_input(self):
        _set_3d_axis_labels(plt.subplot(projection="3d"))
