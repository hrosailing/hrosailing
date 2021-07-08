import unittest
import numpy as np
import hrosailing.polardiagram as pol

from hrosailing.polardiagram.polardiagram import PolarDiagramException


def equal_arrays(a, b, msg=None):
    if np.all(a == b):
        return True
    raise AssertionError(msg)


class PolarDiagramTableTest(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, equal_arrays)
        self.ws_res = np.array([2, 4, 6, 8])
        self.wa_res = np.array([10, 15, 20, 25])
        self.bsps = np.array(
            [
                [1, 2, 3, 4],
                [1.5, 2.4, 3.1, 4.1],
                [1.7, 2.6, 3.5, 4.4],
                [2, 3, 3.8, 4.6],
            ]
        )
        self.polar_diagram = pol.PolarDiagramTable(
            self.ws_res, self.wa_res, self.bsps
        )

    def test_default_init(self):
        polar_diagram = pol.PolarDiagramTable()
        self.assertEqual(polar_diagram.wind_speeds, np.arange(2, 42, 2))
        self.assertEqual(polar_diagram.wind_angles, np.arange(0, 360, 5))
        self.assertEqual(polar_diagram.boat_speeds, np.zeros((72, 20)))

    def test_init(self):
        self.assertEqual(self.polar_diagram.wind_speeds, self.ws_res)
        self.assertEqual(self.polar_diagram.wind_angles, self.wa_res)
        self.assertEqual(self.polar_diagram.boat_speeds, self.bsps)

    def test_init_exception_empty_bsps(self):
        with self.assertRaises(PolarDiagramException):
            pol.PolarDiagramTable(bsps=[])

    def test_init_exception_not_ndim2(self):
        with self.assertRaises(PolarDiagramException):
            pol.PolarDiagramTable(bsps=[[[0]]])

    def test_init_exception_wrong_shape(self):
        with self.assertRaises(PolarDiagramException):
            pol.PolarDiagramTable(bsps=[[0]])

    def test_wind_speeds(self):
        self.assertEqual(self.polar_diagram.wind_speeds, self.ws_res)

    def test_wind_angles(self):
        self.assertEqual(self.polar_diagram.wind_angles, self.wa_res)

    def test_boat_speeds(self):
        self.assertEqual(self.polar_diagram.boat_speeds, self.bsps)

    def test_symmetric_polar_diagram_no_180(self):
        polar_diagram = pol.symmetric_polar_diagram(self.polar_diagram)

        sym_wa_res = np.concatenate([self.wa_res, 360 - np.flip(self.wa_res)])
        sym_bsps = np.row_stack((self.bsps, np.flip(self.bsps, axis=0)))

        self.assertEqual(
            polar_diagram.wind_speeds, self.polar_diagram.wind_speeds
        )
        self.assertEqual(polar_diagram.wind_angles, sym_wa_res)
        self.assertEqual(polar_diagram.boat_speeds, sym_bsps)

    def test_symmetric_polar_diagram_w_180_and_0(self):
        wa_res = [0, 90, 180]
        polar_diagram = pol.PolarDiagramTable(
            ws_res=self.ws_res, wa_res=wa_res, bsps=self.bsps[:3, :]
        )
        polar_diagram = pol.symmetric_polar_diagram(polar_diagram)
        self.assertEqual(polar_diagram.wind_speeds, self.ws_res)
        sym_wa_res = np.array([0, 90, 180, 270])
        self.assertEqual(polar_diagram.wind_angles, sym_wa_res)
        sym_bsps = np.row_stack((self.bsps[:3, :], self.bsps[1, :]))
        self.assertEqual(polar_diagram.boat_speeds, sym_bsps)

    def test_change_one_entry(self):
        self.polar_diagram.change_entries(new_bsps=2.1, ws=2, wa=25)
        self.bsps[3, 1] = 2.1
        self.assertEqual(self.polar_diagram.boat_speeds, self.bsps)

    def test_change_one_column(self):
        self.polar_diagram.change_entries(new_bsps=[3.5, 3.7, 3.9, 4.1], ws=8)
        self.bsps[:, 3] = [3.5, 3.7, 3.9, 4.1]
        self.assertEqual(self.polar_diagram.boat_speeds, self.bsps)

    def test_change_one_row(self):
        self.polar_diagram.change_entries(new_bsps=[1.9, 2.7, 3.6, 4.4], wa=20)
        self.bsps[2, :] = [1.9, 2.7, 3.6, 4.4]
        self.assertEqual(self.polar_diagram.boat_speeds, self.bsps)

    def test_change_subarray(self):
        self.polar_diagram.change_entries(
            new_bsps=[2.3, 3.0, 2.5, 3.4], ws=[4, 6], wa=[15, 20]
        )
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        mask[1, 2] = True
        mask[2, 1] = True
        mask[2, 2] = True
        self.bsps[mask] = np.array([[2.3, 3.0], [2.5, 3.4]]).flat

    def test_change_entries_exceptions_empty(self):
        with self.assertRaises(PolarDiagramException):
            self.polar_diagram.change_entries(new_bsps=[])

    def test_change_entries_exception_wrong_shape(self):
        with self.assertRaises(PolarDiagramException):
            self.polar_diagram.change_entries(new_bsps=[1])

    def test_get_one_slice(self):
        self.assertEqual(
            self.polar_diagram._get_slice_data(2), self.bsps[:, [0]]
        )

    def test_get_multiple_slices(self):
        self.assertEqual(
            self.polar_diagram._get_slice_data([2, 4, 8]),
            self.bsps[:, [0, 1, 3]],
        )

    def test_get_all_slices(self):
        self.assertEqual(self.polar_diagram._get_slice_data(None), self.bsps)

    def test_get_slice_exceptions(self):
        slices = [[], 0, [0, 2, 4]]
        for i, slice_ in enumerate(slices):
            with self.subTest(i=i):
                with self.assertRaises(PolarDiagramException):
                    self.polar_diagram._get_slice_data(slice_)


def polar_table_suite():
    suite = unittest.TestSuite()
    suite.addTests(
        [
            PolarDiagramTableTest("test_default_init"),
            PolarDiagramTableTest("test_init"),
            PolarDiagramTableTest("test_init_exception_empty_bsps"),
            PolarDiagramTableTest("test_init_exception_not_ndim2"),
            PolarDiagramTableTest("test_init_exception_wrong_shape"),
            PolarDiagramTableTest("test_wind_speeds"),
            PolarDiagramTableTest("test_wind_angles"),
            PolarDiagramTableTest("test_boat_speeds"),
            PolarDiagramTableTest("test_symmetric_polar_diagram_no_180"),
            PolarDiagramTableTest("test_symmetric_polar_diagram_w_180_and_0"),
            PolarDiagramTableTest("test_change_one_entry"),
            PolarDiagramTableTest("test_change_one_row"),
            PolarDiagramTableTest("test_change_one_column"),
            PolarDiagramTableTest("test_change_entries_exceptions_empty"),
            PolarDiagramTableTest("test_change_entries_exception_wrong_shape"),
            PolarDiagramTableTest("test_change_subarray"),
            PolarDiagramTableTest("test_get_one_slice"),
            PolarDiagramTableTest("test_get_multiple_slices"),
            PolarDiagramTableTest("test_get_all_slices"),
            PolarDiagramTableTest("test_get_slice_exceptions"),
        ]
    )

    return suite


class PolarDiagramPointCloudTest(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, equal_arrays)
        self.points = np.array(
            [
                [2, 10, 1.1],
                [2, 15, 1.5],
                [2, 20, 1.7],
                [2, 25, 2.1],
                [4, 10, 2],
                [4, 15, 2.4],
                [4, 20, 2.6],
                [4, 25, 3],
                [6, 10, 3],
                [6, 15, 3.1],
                [6, 20, 3.5],
                [6, 25, 3.8],
                [8, 10, 4],
                [8, 15, 4.1],
                [8, 20, 4.4],
                [8, 25, 4.6],
            ]
        )
        self.point_cloud = pol.PolarDiagramPointcloud(self.points)

    def test_default_init(self):
        point_cloud = pol.PolarDiagramPointcloud()
        self.assertEqual(point_cloud.points, np.array([]))

    def test_init(self):
        self.assertEqual(self.point_cloud.points, self.points)

    def test_init_exception_wrong_size(self):
        with self.assertRaises(PolarDiagramException):
            pol.PolarDiagramPointcloud(pts=[0])

    def test_wind_speeds(self):
        self.assertEqual(self.point_cloud.wind_speeds, [2, 4, 6, 8])

    def test_wind_angles(self):
        self.assertEqual(self.point_cloud.wind_angles, [10, 15, 20, 25])

    def test_points(self):
        self.assertEqual(self.point_cloud.points, self.points)

    def test_add_points_with_no_points(self):
        point_cloud = pol.PolarDiagramPointcloud()
        point_cloud.add_points(self.points)
        self.assertEqual(point_cloud.points, self.points)

    def test_add_points(self):
        self.point_cloud.add_points([[2.3, 15.5, 1.65], [3.7, 20.1, 2.43]])
        self.points = np.row_stack(
            (self.points, np.array([[2.3, 15.5, 1.65], [3.7, 20.1, 2.43]]))
        )
        self.assertEqual(self.point_cloud.points, self.points)

    def test_add_points_exception_empty_new_pts(self):
        with self.assertRaises(PolarDiagramException):
            self.point_cloud.add_points(new_pts=[])

    def test_add_points_exception_wrong_shape(self):
        with self.assertRaises(PolarDiagramException):
            self.point_cloud.add_points(new_pts=[0])

    def test_symmetric_polar_diagram_no_points(self):
        point_cloud = pol.PolarDiagramPointcloud()
        point_cloud = pol.symmetric_polar_diagram(point_cloud)
        self.assertEqual(point_cloud.points.size, False)

    def test_symmetric_polar_diagram(self):
        point_cloud = pol.symmetric_polar_diagram(self.point_cloud)
        sym_pts = self.point_cloud.points
        sym_pts[:, 1] = 360 - sym_pts[:, 1]
        pts = np.row_stack((self.point_cloud.points, sym_pts))
        self.assertEqual(point_cloud.points, pts)

    def test_get_slice(self):
        wa, bsp = self.point_cloud._get_slice_data(4)
        self.assertEqual(wa, self.points[self.points[:, 0] == 4][:, 1])
        self.assertEqual(bsp, self.points[self.points[:, 0] == 4][:, 2])

    def test_get_slice_exception(self):
        with self.assertRaises(PolarDiagramException):
            self.point_cloud._get_slice_data(0)

    def test_get_slices_list(self):
        ws, wa, bsp = self.point_cloud._get_slices([4, 8])
        self.assertEqual(ws, [4, 8])
        self.assertEqual(type(wa), list)
        self.assertEqual(type(bsp), list)
        self.assertEqual(len(wa), 2)
        self.assertEqual(len(bsp), 2)
        self.assertEqual(wa[0], self.point_cloud._get_slice_data(4)[0])
        self.assertEqual(bsp[0], self.point_cloud._get_slice_data(4)[1])
        self.assertEqual(wa[1], self.point_cloud._get_slice_data(8)[0])
        self.assertEqual(bsp[1], self.point_cloud._get_slice_data(8)[1])

    def test_get_slices_exception_empty(self):
        with self.assertRaises(PolarDiagramException):
            self.point_cloud._get_slices([])

    def test_get_slices_exception_no_slices(self):
        with self.assertRaises(PolarDiagramException):
            self.point_cloud._get_slices([0, 2])

    def test_get_slices_range(self):
        ws, wa, bsp = self.point_cloud._get_slices((3, 9))
        self.assertEqual(ws, [4, 6, 8])
        self.assertEqual(type(wa), list)
        self.assertEqual(type(bsp), list)
        self.assertEqual(len(wa), 3)
        self.assertEqual(len(bsp), 3)
        self.assertEqual(wa[0], self.point_cloud._get_slice_data(4)[0])
        self.assertEqual(bsp[0], self.point_cloud._get_slice_data(4)[1])
        self.assertEqual(wa[1], self.point_cloud._get_slice_data(6)[0])
        self.assertEqual(bsp[1], self.point_cloud._get_slice_data(6)[1])
        self.assertEqual(wa[2], self.point_cloud._get_slice_data(8)[0])
        self.assertEqual(bsp[2], self.point_cloud._get_slice_data(8)[1])

    def test_get_slices_range_empty(self):
        ws, wa, bsp = self.point_cloud._get_slices((0, 1))
        self.assertEqual(ws, [])
        self.assertEqual(wa, [])
        self.assertEqual(bsp, [])


def point_cloud_suite():
    suite = unittest.TestSuite()
    suite.addTests(
        [
            PolarDiagramPointCloudTest("test_default_init"),
            PolarDiagramPointCloudTest("test_init"),
            PolarDiagramPointCloudTest("test_init_exception_wrong_size"),
            PolarDiagramPointCloudTest("test_wind_speeds"),
            PolarDiagramPointCloudTest("test_wind_angles"),
            PolarDiagramPointCloudTest("test_points"),
            PolarDiagramPointCloudTest("test_add_points_with_no_points"),
            PolarDiagramPointCloudTest("test_add_points"),
            PolarDiagramPointCloudTest(
                "test_add_points_exception_empty_new_pts"
            ),
            PolarDiagramPointCloudTest(
                "test_add_points_exception_wrong_shape"
            ),
            PolarDiagramPointCloudTest(
                "test_symmetric_polar_diagram_no_points"
            ),
            PolarDiagramPointCloudTest("test_symmetric_polar_diagram"),
            PolarDiagramPointCloudTest("test_get_slice"),
            PolarDiagramPointCloudTest("test_get_slice_exception"),
            PolarDiagramPointCloudTest("test_get_slices_list"),
            PolarDiagramPointCloudTest("test_get_slices_exception_empty"),
            PolarDiagramPointCloudTest("test_get_slices_exception_no_slices"),
            PolarDiagramPointCloudTest("test_get_slices_range"),
            PolarDiagramPointCloudTest("test_get_slices_range_empty"),
        ]
    )

    return suite


class PolarDiagramCurveTest(unittest.TestCase):
    def setUp(self):
        def test_func(ws, wa, *params):
            return params[0] * ws * wa + params[1]

        self.f = test_func
        self.params = (1, 2)
        self.radians = False
        self.polar_curve = pol.PolarDiagramCurve(
            self.f, self.params, radians=self.radians
        )

    def test_init(self):
        self.assertEqual(self.polar_curve.curve.__name__, "test_func")
        self.assertEqual(self.polar_curve.parameters, (1, 2))
        self.assertEqual(self.polar_curve.radians, False)

    def test_init_exception(self):
        with self.assertRaises(PolarDiagramException):
            f = 5
            params = 1, 2
            pol.PolarDiagramCurve(f, params)

    def test_curve(self):
        self.assertEqual(self.polar_curve.curve.__name__, "test_func")

    def test_parameters(self):
        self.assertEqual(self.polar_curve.parameters, (1, 2))

    def test_radians(self):
        self.assertEqual(self.polar_curve.radians, False)

    def test_get_wind_angles(self):
        self.addTypeEqualityFunc(np.ndarray, equal_arrays)
        self.assertEqual(
            self.polar_curve._get_wind_angles(), np.linspace(0, 360, 1000)
        )

    def test_get_wind_angles_radians(self):
        self.addTypeEqualityFunc(np.ndarray, equal_arrays)
        polar_curve = pol.PolarDiagramCurve(self.f, self.params, radians=True)
        self.assertEqual(
            polar_curve._get_wind_angles(),
            np.deg2rad(np.linspace(0, 360, 1000)),
        )


def polar_curve_suite():
    suite = unittest.TestSuite()
    suite.addTests(
        [
            PolarDiagramCurveTest("test_init"),
            PolarDiagramCurveTest("test_init_exception"),
            PolarDiagramCurveTest("test_curve"),
            PolarDiagramCurveTest("test_parameters"),
            PolarDiagramCurveTest("test_radians"),
            PolarDiagramCurveTest("test_get_wind_angles"),
            PolarDiagramCurveTest("test_get_wind_angles_radians"),
        ]
    )

    return suite
