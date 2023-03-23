import unittest

from hrosailing.plotting.projections import _create_color_map

class TestCreateColorMap(unittest.TestCase):
    def test_regular_input(self):
        # Execution Test
        _create_color_map(["red", "green", "blue"])