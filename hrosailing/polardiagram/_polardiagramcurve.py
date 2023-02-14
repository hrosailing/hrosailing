# pylint: disable=missing-module-docstring

import csv
from ast import literal_eval
from inspect import getmembers, isfunction

import numpy as np

import hrosailing.pipelinecomponents.modelfunctions as model

from ._basepolardiagram import (
    PolarDiagram,
    PolarDiagramException,
    PolarDiagramInitializationException,
)
from ._plotting import (
    plot_color_gradient,
    plot_convex_hull,
    plot_flat,
    plot_polar,
    plot_surface,
)

MODEL_FUNCTIONS = dict(getmembers(model, isfunction))


class PolarDiagramCurve(PolarDiagram):
    """A class to represent, visualize and work with a polar diagram
    given by a fitted curve/surface.

    Parameters
    ----------
    f : function
        Curve/surface that describes the polar diagram, given as
        a function, with the signature `f(ws, wa, *params) -> bsp`,
        where

        - `ws` and `wa` should be `array_like` of shape `(n,)`
        and refer to wind speeds and wind angles,
        - `params` is a list of additional parameters,
        - `bsp` is `array_like` of shape `(n,)` and contains the resulting
        boat speeds.

    params : Sequence
        Optimal parameters for `f`.

    radians : bool, optional
        Specifies if `f` takes the wind angles in radians or degrees.

        Defaults to `False`.

    Raises
    ------
    PolarDiagramInitializationException
        If `f` is not callable.
    PolarDiagramInitializationException
        If `params` contains not enough parameters for `f`.
    """

    @property
    def default_points(self):
        ws = np.linspace(5, 20, 128)
        wa = np.linspace(5, 355, 144)
        ws, wa = np.meshgrid(ws, wa)
        ws, wa = ws.ravel(), wa.ravel()
        bsp = np.array([self(ws_, wa_) for ws_, wa_ in zip(ws, wa)])
        return np.column_stack([ws, wa, bsp])

    def get_slices(self, ws=None, n_steps=None, full_info=False, **kwargs):
        """
        Other Parameters
        ----------------
        wa_resolution : int, optional
            The number of wind angles that will be used for estimation if an
            interpolator is given.

            Defaults to 100.

        See also
        -------
        `PolarDiagram.get_slices`
        """
        return super().get_slices(ws, n_steps, full_info, **kwargs)

    def ws_to_slices(self, ws, wa_resolution=100, **kwargs):
        """
        See also
        -------
        `PolarDiagramCurve.get_slices`
        `PolarDiagram.ws_to_slices`
        """
        wa_ls = np.linspace(0, 360, wa_resolution)
        return [
            np.row_stack([
                [ws_]*len(wa_ls), wa_ls, self(ws_, wa_ls)
            ])
            for ws_ in ws
        ]

    def __init__(self, f, *params, radians=False):
        if not callable(f):
            raise PolarDiagramInitializationException("`f` is not callable")

        if not self._check_enough_params(f, params):
            raise PolarDiagramInitializationException(
                "`params` is an incorrect amount of parameters for `f`"
            )

        self._f = f
        self._params = params
        self._rad = radians

    @staticmethod
    def _check_enough_params(func, params):
        try:
            func(1, 1, *params)
            return True
        except (IndexError, TypeError):
            return False

    def __repr__(self):
        return (
            f"PolarDiagramCurve(f={self._f.__name__},"
            f"{self._params}, radians={self._rad})"
        )

    def __call__(self, ws, wa):
        # do not change the input wa
        if isinstance(wa, np.ndarray):
            wa = wa.copy()

        if np.any((ws < 0)):
            raise PolarDiagramException("`ws` is negative")

        if self.radians:
            wa = np.rad2deg(wa)
            wa %= 360
            wa = np.deg2rad(wa)
        else:
            wa %= 360

        return self.curve(ws, wa, *self.parameters)

    @property
    def curve(self):
        """Returns a read only version of `self._f`."""
        return self._f

    @property
    def default_slices(self):
        return np.linspace(5, 20, 16)

    @property
    def parameters(self):
        """Returns a read only version of `self._params`."""
        return self._params

    @property
    def radians(self):
        """Returns a read only version of `self._rad`."""
        return self._rad

    def to_csv(self, csv_path):
        """Creates a .csv file with delimiter ':' and the
        following format:

            `PolarDiagramCurve`
            Function: `self.curve.__name__`
            Radians: `self.radians`
            Parameters: `self.parameters`

        Parameters
        ----------
        csv_path : path-like
            Path to a .csv file or where a new .csv file will be created.

        Raises
        ------
        OSError
            If no permission to write is given for file.
        """
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file, delimiter=":")
            csv_writer.writerow([self.__class__.__name__])
            csv_writer.writerow(["Function"] + [self.curve.__name__])
            csv_writer.writerow(["Radians"] + [str(self.radians)])
            csv_writer.writerow(["Parameters"] + list(self.parameters))

    @classmethod
    def __from_csv__(cls, file):
        csv_reader = csv.reader(file, delimiter=":")
        function = next(csv_reader)[1]
        radians = literal_eval(next(csv_reader)[1])
        params = [literal_eval(value) for value in next(csv_reader)[1:]]

        if function not in MODEL_FUNCTIONS:
            raise PolarDiagramInitializationException(
                f"no valid function, named {function}"
            )

        function = MODEL_FUNCTIONS[function]
        return PolarDiagramCurve(function, *params, radians=radians)

    def symmetrize(self):
        """Constructs a symmetric version of the polar diagram,
        by mirroring it at the 0° - 180° axis and returning a new instance.
        """

        def sym_func(ws, wa, *params):
            y = self._f(ws, wa, *params)
            y_symm = self._f(ws, (360 - wa) % 360, *params)
            return 0.5 * (y + y_symm)

        return PolarDiagramCurve(
            sym_func, *self.parameters, radians=self.radians
        )
