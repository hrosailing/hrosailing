from unittest import TestCase
import numpy as np


from hrosailing.processing.weigher import hrosailing_standard_scaled_euclidean_norm


class TestWeigherFunctions(TestCase):
    def setUp(self) -> None:
        self.dimensions = ["TWS", "TWA", "BSP"]

    def test_hrosailing_standard_scaled_euclidean_norm(self):
        pass