# pylint: disable-all

import unittest

import numpy as np

from hrosailing.cruising import convex_direction
from hrosailing.polardiagram import PolarDiagram, PolarDiagramTable


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

        self.assertAlmostEqual(2 / 3, direction[0].proportion)
        self.assertAlmostEqual(1 / 3, direction[1].proportion)

    def test_convex_direction_port_side_angles(self):
        direction = convex_direction(self.pd, 1, 350)

        self.assertEqual(2, len(direction))
        self.assertGreaterEqual(direction[0].proportion, 0)
        self.assertGreaterEqual(direction[1].proportion, 0)
        self.assertAlmostEqual(
            1, direction[0].proportion + direction[1].proportion
        )

    def test_convex_direction_exact_angle_match(self):
        """Test that convex_direction returns exact angles when target matches a vertex."""
        # Test with 90° - should return exactly 90°, not a neighboring angle
        direction = convex_direction(self.pd, 1, 90)
        
        self.assertEqual(1, len(direction), "Should return single direction for exact match")
        self.assertEqual(90, direction[0].angle, "Should return exact target angle 90°")
        
        # Test with 180° - should return exactly 180°, not a neighboring angle  
        direction = convex_direction(self.pd, 1, 180)
        
        self.assertEqual(1, len(direction), "Should return single direction for exact match")
        self.assertEqual(180, direction[0].angle, "Should return exact target angle 180°")
        
        # Test with 45° - should return exactly 45°, not a neighboring angle
        direction = convex_direction(self.pd, 1, 45)
        
        self.assertEqual(1, len(direction), "Should return single direction for exact match")
        self.assertEqual(45, direction[0].angle, "Should return exact target angle 45°")
