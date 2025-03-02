# pylint: disable-all

from unittest.mock import patch

import numpy as np

from hrosailing.polardiagram import PolarDiagramPointcloud
from tests.utils_for_testing import hroTestCase


class TestPolarDiagramPointCloud(hroTestCase):

    def setUp(self) -> None:
        self.pd = PolarDiagramPointcloud(
            np.array([
                [1, 2, 3],
                [1, 3, 3],
                [1, 4, 3]
            ])
        )

    def test_should_return_exact_point_if_contained(self):
        bs = self.pd(1, 2)
        self.assertEqual(bs, 3)

    def test_should_return_interpolated_point_if_not_contained(self):
        bs = self.pd(1, 2.5)
        self.assertIsInstance(bs, float)

    def test_ws_to_slices_without_interpolator_should_not_interpolate(self):
        slices = self.pd.ws_to_slices([1])
        np.testing.assert_almost_equal(slices,
                                       np.array([[[1, 1, 1], [2, 3, 4], [3, 3, 3]]]))

    @patch("hrosailing.processing.Interpolator")
    def test_ws_to_slices_with_interpolator_should_interpolate(self, interpolator_mock):
        wa_resolution = 100
        interpolator_mock.interpolate.return_value = 5

        slices = self.pd.ws_to_slices([1], wa_resolution=wa_resolution,
                                      interpolator=interpolator_mock)
        self.assertEquals(slices[0].shape[0], wa_resolution)
        self.assertTrue(interpolator_mock.interpolate.called)
