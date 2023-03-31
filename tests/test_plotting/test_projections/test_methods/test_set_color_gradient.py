# pylint: disable-all
import unittest

import matplotlib.pyplot as plt

from hrosailing.plotting.projections import _set_color_gradient


class TestSetColorGradient(unittest.TestCase):
    def test_execution(self):
        # Execution Test
        _set_color_gradient(
            plt.subplot(), [1, 2, 3], ["red", "green", "orange"]
        )
