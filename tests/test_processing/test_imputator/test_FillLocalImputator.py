# pylint: disable-all

from datetime import datetime, timedelta
from unittest import TestCase

import hrosailing.core.data as dt
import hrosailing.processing.imputator as imp


class TestFillLocalImputator(TestCase):
    def setUp(self):
        self.fill_before = lambda name, right, mu: right - 0.1
        self.fill_between = lambda name, left, right, mu: 0.5 * (left + right)
        self.fill_after = lambda name, left, mu: left - 0.1
        self.max_time_diff = timedelta(minutes=1, seconds=30)

        self.data = dt.Data().from_dict(
            {
                "datetime": [
                    datetime(2023, 3, 13, 8),
                    datetime(2023, 3, 13, 8, 1),
                    datetime(2023, 3, 13, 8, 2, 30),
                    None,
                    datetime(2023, 3, 13, 8, 3),
                    datetime(2023, 3, 13, 8, 3, 30),
                ],
                "None_col": [None, None, None, None, None, None],
                "TWS": [None, 14.6, 16.9, 17.2, None, 17.6],
                "TWA": [67, 74, 77, 77, 75, None],
                "BSP": [14, 16, None, None, None, 14],
                "idx": [1, 2, 3, 4, 5, 6],
            }
        )

    def test_impute_default(self):
        result = imp.FillLocalImputator().impute(self.data)._data
        expected_result = {
            "datetime": [
                datetime(2023, 3, 13, 8),
                datetime(2023, 3, 13, 8, 1),
                datetime(2023, 3, 13, 8, 2, 30),
                datetime(2023, 3, 13, 8, 2, 45),
                datetime(2023, 3, 13, 8, 3),
                datetime(2023, 3, 13, 8, 3, 30),
            ],
            "TWS": [14.6, 14.6, 16.9, 17.2, 17.2, 17.6],
            "TWA": [67, 74, 77, 77, 75, 75],
            "BSP": [14, 16, 16, 16, 14, 14],
            "idx": [1, 2, 3, 4, 5, 6],
        }

        self.assertDictEqual(result, expected_result)

    def test_impute_custom_fill_before(self):
        result = (
            imp.FillLocalImputator(fill_before=self.fill_before)
            .impute(self.data)
            ._data
        )
        expected_result = {
            "datetime": [
                datetime(2023, 3, 13, 8),
                datetime(2023, 3, 13, 8, 1),
                datetime(2023, 3, 13, 8, 2, 30),
                datetime(2023, 3, 13, 8, 2, 45),
                datetime(2023, 3, 13, 8, 3),
                datetime(2023, 3, 13, 8, 3, 30),
            ],
            "TWS": [14.5, 14.6, 16.9, 17.2, 17.2, 17.6],
            "TWA": [67, 74, 77, 77, 75, 75],
            "BSP": [14, 16, 16, 16, 13.9, 14],
            "idx": [1, 2, 3, 4, 5, 6],
        }

        self.assertDictEqual(result, expected_result)

    def test_impute_custom_fill_after(self):
        result = (
            imp.FillLocalImputator(fill_after=self.fill_after)
            .impute(self.data)
            ._data
        )
        expected_result = {
            "datetime": [
                datetime(2023, 3, 13, 8),
                datetime(2023, 3, 13, 8, 1),
                datetime(2023, 3, 13, 8, 2, 30),
                datetime(2023, 3, 13, 8, 2, 45),
                datetime(2023, 3, 13, 8, 3),
                datetime(2023, 3, 13, 8, 3, 30),
            ],
            "TWS": [14.6, 14.6, 16.9, 17.2, 17.2, 17.6],
            "TWA": [67, 74, 77, 77, 75, 74.9],
            "BSP": [14, 16, 15.9, 15.9, 14, 14],
            "idx": [1, 2, 3, 4, 5, 6],
        }

        self.assertDictEqual(result, expected_result)

    def test_impute_custom_fill_between(self):
        result = (
            imp.FillLocalImputator(fill_between=self.fill_between)
            .impute(self.data)
            ._data
        )
        expected_result = {
            "datetime": [
                datetime(2023, 3, 13, 8),
                datetime(2023, 3, 13, 8, 1),
                datetime(2023, 3, 13, 8, 2, 30),
                datetime(2023, 3, 13, 8, 2, 45),
                datetime(2023, 3, 13, 8, 3),
                datetime(2023, 3, 13, 8, 3, 30),
            ],
            "TWS": [14.6, 14.6, 16.9, 17.2, 17.4, 17.6],
            "TWA": [67, 74, 77, 77, 75, 75],
            "BSP": [14, 16, 16, 16, 14, 14],
            "idx": [1, 2, 3, 4, 5, 6],
        }

        self.assertDictEqual(result, expected_result)

    def test_impute_edge_empty_Data(self):
        with self.assertRaises(KeyError):
            imp.FillLocalImputator().impute(dt.Data())
