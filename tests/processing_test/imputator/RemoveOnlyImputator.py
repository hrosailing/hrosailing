from unittest import TestCase

import hrosailing.processing.imputator as imp
import hrosailing.core.data as dt


class TestRemoveOnlyImputator(TestCase):
    def setUp(self) -> None:
        self.data = dt.Data().from_dict({"TWS": [12, None, 15], "TWA": [None, 34, 40], "BSP": [15, 17, 16]})

    def test_RemoveOnlyImputator_impute(self):
        """
        Input/Output-Test.
        """

        result = imp.RemoveOnlyImputator().impute(self.data)._data
        expected_result = dt.Data().from_dict({"TWS": [15], "TWA": [40], "BSP": [16]})._data
        self.assertDictEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")