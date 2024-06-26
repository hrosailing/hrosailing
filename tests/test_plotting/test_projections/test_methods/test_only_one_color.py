# pylint: disable-all

import unittest

from hrosailing.plotting.projections import _only_one_color


class TestOnlyOneColor(unittest.TestCase):
    def test_string_input(self):
        self.assertTrue(_only_one_color("red"))

    def test_2_color_input(self):
        self.assertFalse(_only_one_color(["red", "blue"]))

    def test_3_tuple_input(self):
        self.assertTrue(_only_one_color((1, 0, 0)))

    def test_4_tuple_input(self):
        self.assertTrue(_only_one_color((1, 0, 0, 1)))
