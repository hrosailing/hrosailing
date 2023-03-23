import unittest

from hrosailing.plotting.projections import _get_gradient_coefficients

class TestGetGradientCoefficients(unittest.TestCase):
    def test_regular_input(self):
        # Input/Output Test
        result = _get_gradient_coefficients(
            [4, 1, 6, 9, 2]
        )
        self.assertEqual(
            result,
            [0.375, 0, 0.625, 1, 0.125]
        )