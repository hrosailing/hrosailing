import math
import unittest

from hrosailing.cruising import convex_direction

from hrosailing.polardiagram import PolarDiagramTable


class TestHROPolar(unittest.TestCase):
    def setUp(self):
        self.pd = PolarDiagramTable(
            [1], [45, 90, 180, 270, 315], [[1], [1], [1], [1], [2]]
        )

    def test_convex_direction(self):
        direction = convex_direction(self.pd, 1, 0)

        self.assertEqual(2, len(direction))
        self.assertEqual(45, direction[0].angle)
        self.assertEqual(315, direction[1].angle)

        self.assertAlmostEqual(2/3, direction[0].proportion)
        self.assertAlmostEqual(1/3, direction[1].proportion)