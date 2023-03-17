from unittest import TestCase
import numpy as np
from datetime import datetime, timedelta

import hrosailing.processing.imputator as imp
import hrosailing.core.data as dt


class TestFillLocalImputator(TestCase):
    def setUp(self) -> None:
        self.fill_before = lambda name, right, mu: right - 0.1
        self.fill_between = lambda name, left, right, mu: .5 * (left + right)
        self.fill_after = lambda name, left, mu: left - 0.1
        self.max_time_diff = timedelta(minutes=1, seconds=30)

        self.data = dt.Data().from_dict({
            "datetime": [datetime(2023, 3, 13, 8), datetime(2023, 3, 13, 8, 1), datetime(2023, 3, 13, 8, 2, 30), None,
                         datetime(2023, 3, 13, 8, 3), datetime(2023, 3, 13, 8, 3, 30)],
            "None_col": [None, None, None, None, None, None],
            "TWS": [None, 14.6, 16.9, 17.2, None, 17.6],
            "TWA": [67, 74, 77, 77, 75, None],
            "BSP": [14, 16, None, None, None, 14],
            "idx": [1, 2, 3, 4, 5, 6]
        })

    def test_impute_default(self):
        """
        Input/Output-Test.
        """
        # TODO: why does this delete the last row? (it is not far apart in time), BSP values in result are strange
        result = imp.FillLocalImputator().impute(self.data)._data
        expected_result = dt.Data().from_dict({
            "datetime": [datetime(2023, 3, 13, 8), datetime(2023, 3, 13, 8, 2, 30),
                         datetime(2023, 3, 13, 8, 2, 45), datetime(2023, 3, 13, 8, 3),
                         datetime(2023, 3, 13, 8, 3, 30)],
            "TWS": [14.6, 16.9, 17.2, 17.2, 17.6],
            "TWA": [67, 77, 77, 75, 75],
            "BSP": [14, 16, 16, 14, 14],
            "idx": [1, 3, 4, 5, 6]

        })._data
        self.assertDictEqual(result, expected_result,
                             f"Expected {expected_result} but got {result}!")

    def test_impute_custom_fill_before(self):
        """
        Input/Output-Test.
        """
        # TODO
        result = imp.FillLocalImputator(fill_before=self.fill_before).impute(self.data)._data
        expected_result = dt.Data().from_dict({
            "datetime": [datetime(2023, 3, 13, 8), datetime(2023, 3, 13, 8, 2, 30),
                         datetime(2023, 3, 13, 8, 2, 45), datetime(2023, 3, 13, 8, 3),
                         datetime(2023, 3, 13, 8, 3, 30)],
            "TWS": [14.5, 16.9, 16.9, 17.6],
            "TWA": [67, 77, 75, 75],
            "BSP": [14, 14, 14, 14]
        })._data
        self.assertDictEqual(result, expected_result,
                             f"Expected {expected_result} but got {result}!")

    def test_impute_custom_fill_after(self):
        """
        Input/Output-Test.
        """
        # TODO
        result = imp.FillLocalImputator().impute(self.data)._data
        expected_result = dt.Data().from_dict({
            "datetime": [datetime(2023, 3, 13, 8), datetime(2023, 3, 13, 8, 2, 30),
                         datetime(2023, 3, 13, 8, 2, 45), datetime(2023, 3, 13, 8, 3),
                         datetime(2023, 3, 13, 8, 3, 30)],
            "TWS": [14.6, 16.9, 16.9, 17.6],
            "TWA": [67.0, 77.0, 75.0, 74.9],
            "BSP": [14, 14, 14, 14]
        })._data
        self.assertDictEqual(result, expected_result,
                             f"Expected {expected_result} but got {result}!")

    def test_impute_custom_fill_between(self):
        """
        Input/Output-Test.
        """
        # TODO
        result = imp.FillLocalImputator(fill_between=self.fill_between).impute(self.data)._data
        expected_result = dt.Data().from_dict({
            "datetime": [datetime(2023, 3, 13, 8), datetime(2023, 3, 13, 8, 2, 30),
                         datetime(2023, 3, 13, 8, 2, 45), datetime(2023, 3, 13, 8, 3),
                         datetime(2023, 3, 13, 8, 3, 30)],
            "TWS": [14.6, 16.9, 16.9, 17.6],
            "TWA": [67.0, 77.0, 75.0, 74.9],
            "BSP": [14, 14, 14, 14]
        })._data
        self.assertDictEqual(result, expected_result,
                             f"Expected {expected_result} but got {result}!")

    def test_impute_edge_empty_Data(self):
        """
        EdgeCase: Data is empty.
        """
        # TODO: remove or use assertRaises to make sure Error is thrown in core.data?
        result = imp.FillLocalImputator().impute(dt.Data())._data
        expected_result = dt.Data()._data
        self.assertDictEqual(result, expected_result,
                             f"Expected {expected_result} but got {result}!")
