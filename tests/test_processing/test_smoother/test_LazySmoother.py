# pylint: disable-all

from unittest import TestCase

import hrosailing.core.data as dt
import hrosailing.processing.smoother as smt


class TestLazySmoother(TestCase):
    def setUp(self):
        self.data = dt.Data().from_dict(
            {"TWS": [12, None, 15], "TWA": [None, 34, 40], "BSP": [15, 17, 16]}
        )

    def test_smooth(self):
        result = smt.LazySmoother().smooth(self.data)._data
        expected_result = self.data._data

        self.assertEqual(result, expected_result)
