import unittest

import matplotlib.pyplot as plt

from hrosailing.plotting.projections import _set_color_cycle


class TestSetColorCycle(unittest.TestCase):
    def test_execution(self):
        _set_color_cycle(plt.subplot(), [1, 2, 3, 4], ("red", "green"))
