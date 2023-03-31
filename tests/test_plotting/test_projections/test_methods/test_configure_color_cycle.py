import unittest

from hrosailing.plotting.projections import _configure_color_cycle


class TestConfigureColorCycle(unittest.TestCase):
    def test_tuple(self):
        # Input/Output
        color_cycle = ["blue"] * 3
        _configure_color_cycle(
            color_cycle, [(3, "green"), (1, "red"), (2, "blue")], [1, 2, 3]
        )
        self.assertEqual(color_cycle, ["red", "blue", "green"])

    def test_list(self):
        # Input/Output
        color_cycle = ["blue"] * 3
        _configure_color_cycle(
            color_cycle, ["red", "blue", "green", "orange"], [1, 2, 3]
        )
        self.assertEqual(color_cycle, ["red", "blue", "green"])
