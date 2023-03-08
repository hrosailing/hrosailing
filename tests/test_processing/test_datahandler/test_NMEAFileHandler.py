from unittest import TestCase
import numpy as np

import hrosailing.processing.datahandler as dth
import hrosailing.core.data as dt


class TestNMEAFileHandler(TestCase):
    def setUp(self) -> None:
        self.wanted_sen = ["MWV"]
        self.unwanted_sen = ["GLL", "RMC"]
        self.all_sen = ["MWV", "GLL", "RMC"]
        self.wanted_attr = ["wind angle", "reference", "wind speed"]
        self.unwanted_attr = ["reference", "wind speed units", "status", "checksum"]
        self.all_attr = ["wind angle", "reference", "wind speed", "reference", "wind speed units", "status", "checksum"]

        self.data = np.array(["$IIMWV,199,R,18.6,N,A*1D\n",
                              "$IIMWV,195,T,24.3,N,A*1D\n",
                              "$IIRMC,083453,A,5428.811,N,01234.275,E,05.7,066,020821,03,E,A*0A\n",
                              "$IIGLL,5428.811,N,01234.277,E,083454,A,A*5B\n",
                              "$IIMWV,196,R,18.0,N,A*14\n",
                              "$IIMWV,195,T,24.0,N,A*1E\n",
                              "$IIRMC,083454,A,5428.811,N,01234.277,E,05.6,070,020821,03,E,A*09\n",
                              "$IIGLL,5428.812,N,01234.280,E,083455,A,A*51\n"])

        with open("TestNMEAFileHandler.vdr", 'w') as file:
            file.writelines(self.data)

    def test_init_senfilter_wanted_sentences_is_not_None(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler(wanted_sentences=self.wanted_sen)._sentence_filter(self.data[3])
        expected_result = False
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_init_senfilter_unwanted_sentences_is_not_None(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler(unwanted_sentences=self.unwanted_sen)._sentence_filter(self.data[1])
        expected_result = True
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_init_senfilter_both_set(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler(unwanted_sentences=self.unwanted_sen, wanted_sentences=self.wanted_sen)._sentence_filter(self.data[6])
        expected_result = False
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

    def test_init_attrfilter_wanted_sentences_is_not_None(self):
        """
        Input/Output-Test.
        """
        result = dth.NMEAFileHandler(wanted_attributes=self.wanted_attr)._attribute_filter(self.data[])
        expected_result = False
        self.assertEqual(result, expected_result,
                         f"Expected {expected_result} but got {result}!")

'''
    def test_handle_custom(self):
        """
        Input/Output-Test.
        """

        result = dth.NMEAFileHandler().handle("TestNMEAFileHandler.vdr")
        expected_result =
        testing_function(result, expected_result,
                         f"Expected {expected_result} but got {result}!")
'''
