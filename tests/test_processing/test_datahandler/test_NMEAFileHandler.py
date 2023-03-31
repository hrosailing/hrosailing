import os
from datetime import date, datetime, time
from unittest import TestCase

import numpy as np

import hrosailing.core.data as dt
import hrosailing.processing.datahandler as dth
from tests.utils_for_testing import hroTestCase


class TestNMEAFileHandler(hroTestCase):
    def setUp(self) -> None:
        self.wanted_sen = ["MWV"]
        self.unwanted_sen = ["GLL"]
        self.all_sen = ["MWV", "GLL"]
        self.wanted_attr = ["TWA", "TWS"]
        self.unwanted_attr = ["lat", "lon", "time"]

        self.field = np.array(
            [["Wind angle", 199], ["TWA", 199], ["TWS", 24.3], ["lat", 54]]
        )

        self.data_all = {
            "TWA": [199.0, 195.0, None, 196.0, 195.0, None],
            "TWS": [18.6, 24.3, None, 18.0, 24.0, None],
            "lat": [None, None, 54.480183, None, None, 54.48019],
            "lon": [None, None, 12.571283, None, None, 12.5713],
            "time": [None, None, time(8, 34, 54), None, None, time(8, 34, 55)],
        }

        self.data_all_comp = {
            "TWA": [199.0, 195.0, 196.0, 195.0],
            "TWS": [18.6, 24.3, 18.0, 24.0],
            "lat": [None, 54.480183, None, 54.48019],
            "lon": [None, 12.571283, None, 12.5713],
            "time": [None, time(8, 34, 54), None, time(8, 34, 55)],
        }

        self.data_attr_sel = dt.Data().from_dict(
            {
                "TWA": [199.0, 195.0, 196.0, 195.0],
                "TWS": [18.6, 24.3, 18.0, 24.0],
            }
        )

        self.sentences = np.array(
            [
                "$IIMWV,199,T,18.6,N,A*1B\n",
                "$IIMWV,195,T,24.3,N,A*1D\n",
                "$IIGLL,5428.811,N,01234.277,E,083454,A,A*5B\n",
                "$IIMWV,196,T,18.0,N,A*12\n",
                "$IIMWV,195,T,24.0,N,A*1E\n",
                "$IIGLL,5428.812,N,01234.280,E,083455,A,A*51\n",
            ]
        )

        with open("TestNMEAFileHandler.vdr", "w") as file:
            file.writelines(self.sentences)

    def tearDown(self) -> None:
        os.remove("TestNMEAFileHandler.vdr")

    def test_init_senfilter_wanted_sentences_is_not_None(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler(
            wanted_sentences=self.wanted_sen
        )._sentence_filter(self.sentences[2])
        expected_result = False
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_init_senfilter_unwanted_sentences_is_not_None(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler(
            unwanted_sentences=self.unwanted_sen
        )._sentence_filter(self.sentences[1])
        expected_result = True
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_init_senfilter_both_set(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler(
            unwanted_sentences=self.unwanted_sen,
            wanted_sentences=self.wanted_sen,
        )._sentence_filter(self.sentences[5])
        expected_result = False
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_init_attrfilter_wanted_attributes_is_not_None(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler(
            wanted_attributes=self.wanted_attr
        )._attribute_filter(self.field[0])
        expected_result = True
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_init_attrfilter_unwanted_attributes_is_not_None(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler(
            unwanted_attributes=self.unwanted_attr
        )._attribute_filter(self.field[3])
        expected_result = False
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_init_attrfilter_both_None(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler()._attribute_filter(self.field[2])
        expected_result = True
        self.assertEqual(
            result,
            expected_result,
            f"Expected {expected_result} but got {result}!",
        )

    def test_handle_default(self):
        """
        Input/Output-Test.
        """
        handler = dth.NMEAFileHandler(
            post_filter_types=(float, date, time, datetime, str)
        )
        result = handler.handle("TestNMEAFileHandler.vdr")._data
        expected_result = self.data_all
        self.assertListEqual(
            result["TWS"], expected_result["TWS"], f"Unexpected TWS!"
        )
        self.assertListEqual(
            result["TWA"], expected_result["TWA"], f"Unexpected TWA!"
        )

        self.assert_list_almost_equal(
            result=result["lat"],
            expected_result=expected_result["lat"],
            places=3,
            msg=f"Unexpected lat!",
        )
        self.assert_list_almost_equal(
            result=result["lon"],
            expected_result=expected_result["lon"],
            places=3,
            msg=f"Unexpected lon!",
        )

        self.assert_time_list_equal(
            result["time"], expected_result["time"], f"Unexpected time!"
        )

    def test_handle_custom_wanted_sentences(self):
        """
        Input/Output-Test.
        """
        result = (
            dth.NMEAFileHandler(wanted_sentences=self.wanted_sen)
            .handle("TestNMEAFileHandler.vdr")
            ._data
        )
        expected_result = self.data_attr_sel._data
        self.assertListEqual(
            result["TWA"],
            expected_result["TWA"],
            f"Expected {expected_result['TWA']} but got {result['TWA']}!",
        )
        self.assertListEqual(
            result["TWS"],
            expected_result["TWS"],
            f"Expected {expected_result['TWS']} but got {result['TWS']}!",
        )

    def test_handle_custom_unwanted_sentences(self):
        """
        Input/Output-Test.
        """
        result = (
            dth.NMEAFileHandler(unwanted_sentences=self.unwanted_sen)
            .handle("TestNMEAFileHandler.vdr")
            ._data
        )
        expected_result = self.data_attr_sel._data
        self.assertListEqual(
            result["TWA"],
            expected_result["TWA"],
            f"Expected {expected_result['TWA']} but got {result['TWA']}!",
        )
        self.assertListEqual(
            result["TWS"],
            expected_result["TWS"],
            f"Expected {expected_result['TWS']} but got {result['TWS']}!",
        )

    def test_handle_custom_wanted_attributes(self):
        """
        Input/Output-Test.
        """
        result = (
            dth.NMEAFileHandler(wanted_attributes=self.wanted_attr)
            .handle("TestNMEAFileHandler.vdr")
            ._data
        )
        expected_result = self.data_attr_sel
        self.assertListEqual(
            result["TWA"],
            expected_result["TWA"],
            f"Expected {expected_result['TWA']} but got {result['TWA']}!",
        )
        self.assertListEqual(
            result["TWS"],
            expected_result["TWS"],
            f"Expected {expected_result['TWS']} but got {result['TWS']}!",
        )

    def test_handle_custom_unwanted_attributes(self):
        """
        Input/Output-Test.
        """
        result = (
            dth.NMEAFileHandler(unwanted_attributes=self.unwanted_attr)
            .handle("TestNMEAFileHandler.vdr")
            ._data
        )
        expected_result = self.data_all
        self.assertListEqual(
            result["TWA"],
            expected_result["TWA"],
            f"Expected {expected_result['TWA']} but got {result['TWA']}!",
        )
        self.assertListEqual(
            result["TWS"],
            expected_result["TWS"],
            f"Expected {expected_result['TWS']} but got {result['TWS']}!",
        )
