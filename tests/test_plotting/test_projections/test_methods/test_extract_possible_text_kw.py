# pylint: disable-all
import unittest

from hrosailing.plotting.projections import _extract_possible_text_kw


class TestExtractPossibleTextKW(unittest.TestCase):
    def test_regular_input(self):
        # Input/Output
        self.assertEqual(
            _extract_possible_text_kw({"hallo": 5}), ({}, {"hallo": 5})
        )
