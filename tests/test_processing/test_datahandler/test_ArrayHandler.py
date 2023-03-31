"""
Tests
"""

import csv
from datetime import datetime
from unittest import TestCase

import numpy as np
import pandas as pd

import hrosailing.core.data as dt
import hrosailing.processing.datahandler as dth


class TestArrayHandler(TestCase):
    def setUp(self) -> None:
        self.pd_dataframe = pd.DataFrame(
            data=np.array([[12, 34, 15], [13, 40, 17]]),
            columns=["TWS", "TWA", "BSP"],
        )
        self.tuple = (
            np.array([[12, 34, 15], [13, 40, 17]]),
            ("TWS", "TWA", "BSP"),
        )

    def test_handle_pdDataFrame(self):
        """
        Input/Output-Test.
        """
        result = dth.ArrayHandler().handle(self.pd_dataframe)._data
        expected_result = {"TWS": [12, 13], "TWA": [34, 40], "BSP": [15, 17]}
        self.assertDictEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_handle_array_like_and_ordered_iterable(self):
        """
        Input/Output-Test.
        """
        result = dth.ArrayHandler().handle(self.tuple)._data
        expected_result = (
            dt.Data()
            .from_dict({"TWS": [12, 13], "TWA": [34, 40], "BSP": [15, 17]})
            ._data
        )
        self.assertDictEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )
