# pylint: disable-all

import numpy as np

from hrosailing.polardiagram import PolarDiagram


class DummyPolarDiagram(PolarDiagram):
    def to_csv(self, csv_path):
        with open(csv_path, "w", encoding="utf-8") as file:
            file.write("DummyPolarDiagram")

    @classmethod
    def __from_csv__(cls, file):
        return cls()

    def __call__(self, ws, wa):
        return 1

    def symmetrize(self):
        pass

    @property
    def default_points(self):
        return [[1, 1, 1]]

    @property
    def default_slices(self):
        return [1, 2, 3]

    def ws_to_slices(self, ws, **kwargs):
        return [np.array([[1], [1], [1]])]

    def __str__(self):
        return "Dummy"

    def __repr__(self):
        return "Dummy()"
