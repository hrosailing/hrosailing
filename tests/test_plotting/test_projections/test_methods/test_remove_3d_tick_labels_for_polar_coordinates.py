import unittest

import matplotlib.pyplot as plt

from hrosailing.plotting.projections import (
    _remove_3d_tick_labels_for_polar_coordinates,
)


class TestRemove3DTickLabelsForPolarCoordinates(unittest.TestCase):
    def test_regular_input(self):
        # Execution Test
        _remove_3d_tick_labels_for_polar_coordinates(
            plt.subplot(projection="3d")
        )
