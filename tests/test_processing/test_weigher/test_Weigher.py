# pylint: disable-all
from unittest import TestCase

import numpy as np

from hrosailing.processing import ArithmeticMeanInterpolator, Weigher
from tests.utils_for_testing import parameterized


class DummyWeigher(Weigher):

    def weigh(self, points) -> (np.ndarray, dict):
        return np.zeros(1), {}


class TestWeigher(TestCase):

    def testAdditionThrowsTypeErrorForInvalidTypes(self):
        with self.assertRaises(TypeError):
            DummyWeigher() + "Hello world"

    def testMultiplicationThrowsTypeErrorForInvalidTypes(self):
        with self.assertRaises(TypeError):
            DummyWeigher() * "Hello world"

    def testSubtractionThrowsTypeErrorForInvalidTypes(self):
        with self.assertRaises(TypeError):
            DummyWeigher() - "Hello world"

    def testDivisionThrowsTypeErrorForInvalidTypes(self):
        with self.assertRaises(TypeError):
            DummyWeigher() / "Hello world"

    def testDivisionByThrowsTypeErrorForInvalidTypes(self):
        with self.assertRaises(TypeError):
            "Hello world" / DummyWeigher()
