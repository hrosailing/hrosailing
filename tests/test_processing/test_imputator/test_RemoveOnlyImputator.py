# pylint: disable-all

from unittest import TestCase

import hrosailing.core.data as dt
import hrosailing.processing.imputator as imp


class TestRemoveOnlyImputator(TestCase):
    def setUp(self):
        self.data = dt.Data().from_dict(
            {"TWS": [12, None, 15], "TWA": [None, 34, 40], "BSP": [15, 17, 16]}
        )

    def test_impute(self):
        result = imp.RemoveOnlyImputator().impute(self.data)._data
        expected_result = (
            dt.Data().from_dict({"TWS": [15], "TWA": [40], "BSP": [16]})._data
        )

        self.assertDictEqual(result, expected_result)

    def test_impute_edge_empty_data(self):
        data = dt.Data()
        result = imp.RemoveOnlyImputator().impute(data)._data
        expected_result = dt.Data()._data

        self.assertDictEqual(result, expected_result)
