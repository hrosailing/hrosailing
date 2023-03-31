# pylint: disable-all
import unittest

from hrosailing.plotting.projections import _get_info_intervals


class TestGetInfoIntervals(unittest.TestCase):
    def test_elaborate_input(self):
        # Input/Output test using input satisfying all kinds of cases

        result = _get_info_intervals(
            ["A", "A", "A", "B", "B", "C", "B", "B", "A"]
        )
        expected = [[0, 1, 2, 8], [3, 4, 6, 7], [5]]
        self.assertEqual(result, expected)

    def test_edge_case_empty_list(self):
        # Input/Output

        result = _get_info_intervals([])
        expected = []

        self.assertEqual(result, expected)
