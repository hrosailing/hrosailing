from unittest import TestCase
import numpy as np

import hrosailing.processing.datahandler as dth
import hrosailing.core.data as dt
from datetime import datetime, timedelta, time


class TestNMEAFileHandler(TestCase):
    def setUp(self) -> None:
        self.wanted_sen = ["MWV"]
        self.unwanted_sen = ["GLL"]
        self.all_sen = ["MWV", "GLL"]
        self.wanted_attr = ["wind angle", "reference", "wind speed"]
        self.unwanted_attr = ["wind speed units", "status", "checksum"]
        self.all_attr = ["wind angle", "reference", "wind speed", "reference", "wind speed units", "status", "checksum"]

        self.field = np.array([["wind angle", 199], ["TWS", 24.3], ["status", 'quick']])

        self.data = dt.Data().from_dict(
            {"TWA": [199.0, 195.0, None, 196.0, 195.0, None],
             "TWS": [18.6, 24.3, None, 18.0, 24.0, None],
             "lat": [None, None, 5428.811, None, None, 5428.812],
             "lon": [None, None, 01234.277, None, None, 01234.280],
             "time": [None, None, time(8, 34, 54), None, None, time(8, 34, 55)]})

        self.sentences = np.array(["$IIMWV,199,T,18.6,N,A*1B\n",
                                   "$IIMWV,195,T,24.3,N,A*1D\n",
                                   "$IIGLL,5428.811,N,01234.277,E,083454,A,A*5B\n",
                                   "$IIMWV,196,T,18.0,N,A*12\n",
                                   "$IIMWV,195,T,24.0,N,A*1E\n",
                                   "$IIGLL,5428.812,N,01234.280,E,083455,A,A*51\n"])

        with open("TestNMEAFileHandler.vdr", 'w') as file:
            file.writelines(self.sentences)

    def test_init_senfilter_wanted_sentences_is_not_None(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler(wanted_sentences=self.wanted_sen)._sentence_filter(self.sentences[2])
        expected_result = False
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_init_senfilter_unwanted_sentences_is_not_None(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler(unwanted_sentences=self.unwanted_sen)._sentence_filter(self.sentences[1])
        expected_result = True
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_init_senfilter_both_set(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler(unwanted_sentences=self.unwanted_sen,
                                     wanted_sentences=self.wanted_sen)._sentence_filter(self.sentences[5])
        expected_result = False
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_init_attrfilter_wanted_sentences_is_not_None(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler(wanted_attributes=self.wanted_attr)._attribute_filter(self.field[0])
        expected_result = True
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_init_attrfilter_unwanted_sentences_is_not_None(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler(unwanted_attributes=self.unwanted_attr)._attribute_filter(self.field[2])
        expected_result = False
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_init_attrfilter_both_None(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler()._attribute_filter(self.field[2])
        expected_result = True
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_handle_default(self):
        """
        Input/Output-Test.
        """

        result = dth.NMEAFileHandler().handle("TestNMEAFileHandler.vdr")._data
        expected_result = self.data._data
        self.maxDiff = None
        for key in expected_result.keys():
            with self.subTest(f"test {key}"):
                np.testing.assert_almost_equal(result[key], expected_result[key], decimal=6,
                                               err_msg=f"Expected {expected_result} but got {result}!")
