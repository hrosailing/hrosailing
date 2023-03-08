from unittest import TestCase
import numpy as np

import hrosailing.processing.datahandler as dth
import hrosailing.core.data as dt


class TestNMEAFileHandler(TestCase):
    def setUp(self) -> None:
        self.wanted_sen = ["MWV"]
        self.unwanted_sen = ["GLL", "RMC"]
        self.wanted_attr = ["wind angle", "reference", "wind speed"]
        self.unwanted_attr = ["reference", "wind speed", "wind speed units", "status", "checksum"]

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

    def test_init(self):
        """
        Input/Output-Test.
        """
        # TODO: sentence filter?
        pass


'''
    def test_handle(self):
        """
        Input/Output-Test.
        """

        with self.subTest("custom NMEAFileHandler"):
            result = dth.NMEAFileHandler().handle("TestNMEAFileHandler.vdr")
            expected_result =
            testing_function(result, expected_result,
                             f"Expected {expected_result} but got {result}!")
'''