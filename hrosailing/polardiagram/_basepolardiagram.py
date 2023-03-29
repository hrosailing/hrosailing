# pylint: disable=missing-module-docstring

from abc import ABC, abstractmethod

import numpy as np


class PolarDiagram(ABC):
    """Base class for all polar diagrams.

    Attributes
    -----------
    default_slices (abstract property) : numpy.ndarray of shape (n)
        Should give the default windspeeds for `get_slices`.

    default_points (abstract property) : numnpy.ndarray of shape (n, 3)
        Should give the 'points' (consisting of wind speed, wind angle and boat speed) at which the
        polar diagram is evaluated by default.
    """

    @abstractmethod
    def to_csv(self, csv_path):
        """This method should, given a path, write a .csv file in
        the location, containing human-readable information about the
        polar diagram object that called the method.

        Parameters
        ----------
        csv_path: str
            Path of the created csv file.
        """

    @classmethod
    @abstractmethod
    def __from_csv__(cls, file):
        """"""

    @abstractmethod
    def __call__(self, ws, wa):
        pass

    @abstractmethod
    def symmetrize(self):
        """This method should return a new `PolarDiagram` object that is a
        symmetric (i.e. mirrored along the 0 - 180° axis) version of the
        `PolarDiagram` object called the method.
        """

    def get_slices(self, ws=None, n_steps=None, full_info=False, **kwargs):
        """This method should produce a list of 'slices' describing the
        performance of specified wind_speed as well as wind angles and
        corresponding boat speeds, that reflect how the vessel behaves near
        the given wind speeds.

        Parameters
        ----------
        ws : int, float or iterable thereof, optional
            The wind speeds corresponding to the requested slices.

            Defaults to `self.default_slices`

        n_steps : int, optional
            If set, a total of `n_steps` wind_speeds between each value given
            in `ws` is taken into account. For example `n_steps = 1` adds all
            midpoints.

            Defaults to 0.

        full_info : bool, optional
            Specifies wether the additional value `info` will be
            returned or not

            Defaults to `False`.

        **kwargs :
            Additional keyword arguments whose functionality depends on the
            inheriting class. Are forwarded to `ws_to_slices`.

        Returns
        ---------
        labels : np.ndarray
            The wind speeds corresponding to the slices

        slices : list of `numpy.ndarray` of shape (3, *)
            A list of slices.
            A slice stores row wise data points consisting of actual
            wind speed, wind angle and boat speed.
            Note that the actual wind speed can differ (slightly) from the
            wind speed corresponding to the slice, depending on the inheriting
            class.
        """
        ws = self._get_windspeeds(ws, n_steps)
        kwargs["full_info"] = full_info
        slices = self.ws_to_slices(ws, **kwargs)
        if full_info:
            info = self.get_slice_info(ws, slices, **kwargs)
            return ws, slices, info
        return ws, slices

    def _get_windspeeds(self, ws, n_steps):
        if ws is None:
            ws = self.default_slices
        if isinstance(ws, (int, float)):
            return [ws]
        try:
            all_numbers = all(isinstance(ws_, (int, float, np.integer, np.floating)) for ws_ in ws)
        except TypeError as exp:
            raise TypeError(
                "`ws` has to be an int a float or an iterable"
            ) from exp
        if not all_numbers:
            raise TypeError(
                "If `ws` is an iterable, it needs to iterate over int or float"
            )
        if n_steps is None:
            return np.array(ws)
        if n_steps <= 0:
            raise ValueError("`n_steps` has to be positive")
        return np.concatenate(
            [
                np.linspace(ws_before, ws_next, n_steps + 2)[:-1]
                for ws_before, ws_next in zip(ws, ws[1:])
            ]
            + [[ws[-1]]]
        )

    def _get_wind(self, wind):
        if isinstance(wind, np.ndarray):
            if wind.shape[1] == 2:
                return wind
            if wind.shape[0] == 2:
                return wind.T
            raise ValueError(
                "`wind` should be a tuple or an array with a dimension of"
                f" shape 2,\ngot an array of shape {wind.shape} instead."
            )
        if isinstance(wind, tuple):
            if len(wind) != 2:
                raise ValueError(
                    "`wind` should be a tuple of size 2 or an array,\n"
                    f"got a tuple of size {len(wind)} instead."
                )
            ws, wa = wind
            ws, wa = np.meshgrid(ws, wa)
            return np.array(list(zip(ws.ravel(), wa.ravel())))
        raise TypeError(
            f"`wind` should be a tuple or an array, got {type(wind)} instead."
        )

    @property
    @abstractmethod
    def default_points(self):
        pass

    @property
    @abstractmethod
    def default_slices(self):
        pass

    @abstractmethod
    def ws_to_slices(self, ws, **kwargs):
        """
        Should produce slices for the windspeeds given in ws.

        Parameters
        ---------
        ws : 1 dimensional numpy.ndarray
            The wind speeds corresponding to the requested slices.

        **kwargs :
            Further keyword arguments that may be used by custom
            implementations. Are forwarded by the `get_slices` method.
            Supports the key 'full_info'.

        Returns
        --------
        slices : list of (3, *) numpy.ndarrays
            List of the requested slices. The three rows of a slice
            should correspond to the actual
            wind speeds, the wind angles and the boat speeds

        See also
        -----------
        `get_slices`
        """

    def get_points(self, wind=None):
        """
        Returns a read only version of all relevant points specified by wind

        Parameters
        ----------
        wind : (2, d) or (d, 2) numpy.ndarray or tuple of 2 (1, d) numpy.ndarray or `None`, optional
            A specification of the wind.

            - If given as a (2, d) or (d, 2) numpy.ndarray the columns (rows)
                are interpreted as different wind records
                (for d=2 we assume the shape (d, 2))
            - If given as a tuple, the first array defines the wind speed
             resolution, the second array defines the wind angle resolution.
             The wind records are all combinations thereof.

        Returns
        ---------
        points : (m, 3) numpy.ndarray
            Row-wise contains all records specified in `wind`
            consisting of wind speed, wind angle, boat speed.
            If `wind` is not specified, `self.default_points` is returned.
        """

        if wind is None:
            return self.default_points
        wind = self._get_wind(wind)
        bsps = [self(wa, ws) for wa, ws in wind]
        return np.column_stack([wind, bsps])

    def get_slice_info(self, ws, slices, **kwargs):
        """
        Should produce additional information about slices depending on the
        inheriting class which is returned when the `full_info` parameter of
        the `get_slices` method is set to `True`. If no further information
        is known, return None.

        Parameters
        ----------
        ws : 1 dimensional numpy.ndarray
            The wind speeds corresponding to the requested slices.

        slices : list of (3, *) `np.ndarray`
            Slices produced by `ws_to_slices`

        **kwargs :
            Further keyword arguments that may be used by custom
            implementations. Are forwarded by the `get_slices` method.
            Supports the key 'full_info'.

        Returns
        -------
        info : list of lists
            List of additional information for each record of each slice
            organized in the same manner as a slice.

        See also
        -----------
        `get_slices`
        """
        # I do something with the variables to make codefactor happy, feel free to refactor
        kwargs["ws"] = ws
        kwargs["slices"] = slices
        return None
