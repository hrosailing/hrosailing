# pylint: disable-all

from unittest import TestCase

import numpy as np

import hrosailing.core.data as dt
import hrosailing.processing.smoother as smt


class TestLazySmoother(TestCase):
    def setUp(self) -> None:
        self.data = dt.Data().from_dict(
            {"TWS": [12, None, 15], "TWA": [None, 34, 40], "BSP": [15, 17, 16]}
        )

    def test_smooth(self):
        """
        Input/Output-Test.
        """

        result = smt.LazySmoother().smooth(self.data)._data
        expected_result = self.data._data
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )
