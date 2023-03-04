from unittest import TestCase
import numpy as np
from datetime import datetime
import csv
import os

import hrosailing.core.data as dt
import hrosailing.processing.datahandler as dth


class TestCsvFileHandler(TestCase):
    def setUp(self) -> None:
        self.data = np.array(
            [["datetime", "TWS", "TWA", "BSP"], [datetime(2023, 3, 4), 12, 34, 15], [datetime(2023, 3, 5), 13, 40, 17]])
        with open('TestCsvFileHandler.csv', mode='w') as file:
            csv.writer(file).writerows(self.data)

    def test_CsvFileHandler_handle_Errors(self):
        """
        Error occurs if date format in Handler does not match the one in the file.
        """
        with self.assertRaises(RuntimeError):
            dth.CsvFileHandler().handle("TestCsvFileHandler.csv")

    def test_CsvFileHandler_handle(self):
        """
        Input/Output-Test.

        ATTENTION: the parameter `date_format` has to be set in order for the Handler to work
        """

        result = dth.CsvFileHandler("%Y-%m-%d %H:%M:%S").handle("TestCsvFileHandler.csv")._data
        expected_result = dt.Data().from_dict({"datetime": [datetime(2023, 3, 4), datetime(2023, 3, 5)],
                                               "TWS": [12, 13], "TWA": [34, 40], "BSP": [15, 17]})._data
        self.assertDictEqual(result, expected_result,
                             f"Expected {expected_result} but got {result}!")


# os.remove("TestCsvFileHandler.csv")
