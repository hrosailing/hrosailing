import csv
from unittest import TestCase
import numpy as np
import pandas as pd
from datetime import datetime

import hrosailing.core.data as dt
import hrosailing.processing.datahandler as dth


class TestArrayHandler(TestCase):
    def setUp(self) -> None:
        self.pd_dataframe = pd.DataFrame([["TWS", "TWA", "BSP"], [12, 34, 15], [13, 40, 17]])
        self.tuple = (np.array([[12, 34, 15], [13, 40, 17]]), ("TWS", "TWA", "BSP"))


    def test_ArrayHandler_handle(self):
        """
        Input/Output-Test.
        """
        # TODO: this still causes problems
        '''
        with self.subTest("pandas DataFrame"):
            result = dth.ArrayHandler().handle(self.pd_dataframe)._data
            expected_result = dt.Data().from_dict({"TWS": [12, 13], "TWA": [34, 40], "BSP": [15, 17]})._data
            self.assertDictEqual(result, expected_result,
                                 f"Expected {expected_result} but got {result}!")
        '''
        with self.subTest("array_like and ordered iterable"):
            result = dth.ArrayHandler().handle(self.tuple)._data
            expected_result = dt.Data().from_dict({"TWS": [12, 13], "TWA": [34, 40], "BSP": [15, 17]})._data
            self.assertDictEqual(result, expected_result,
                                 f"Expected {expected_result} but got {result}!")
