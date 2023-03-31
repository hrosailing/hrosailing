import unittest

import matplotlib.pyplot as plt

from hrosailing.plotting.projections import _set_polar_axis


class TestSetPolarAxis(unittest.TestCase):
    def test_execution(self):
        # Execution Test
        _set_polar_axis(plt.subplot(projection="polar"))
