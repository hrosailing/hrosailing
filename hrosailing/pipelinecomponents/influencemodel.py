"""
Defines the `InfluenceModel` abstract base class that can be used
to create custom influence models.

Subclasses of `InfluenceModel` can be used with

- the `PolarPipeline` class in the `hrosailing.pipeline` module,
- various functions in the `hrosailing.cruising` module.
"""

from abc import ABC, abstractmethod

import numpy as np

from hrosailing.pipelinecomponents._utils import ComponentWithStatistics
from hrosailing.wind import convert_apparent_wind_to_true

from ._utils import data_dict_to_numpy


class InfluenceException(Exception):
    """Raised when removing or adding influence does not work."""


class InfluenceModel(ComponentWithStatistics, ABC):
    """Base class for all influence model classes.


    Abstract Methods
    ----------------
    remove_influence(data)

    add_influence(pd, influence_data)

    fit(training_data)
    """

    @abstractmethod
    def remove_influence(self, data):
        """This method should be used to create a `numpy.ndarray`
        output from a given `Data` object where the columns correspond to wind speed, wind angle and
        boat speed respectively.

        Parameters
        ----------
        data : Data
            Should contain at least keys for wind speed, wind angle
            and either speed over ground, speed over water or boat speed.

        Returns
        -------
        out : numpy.ndarray
        """

    @abstractmethod
    def add_influence(self, pd, influence_data):
        """This method should be used, given a polar diagram and a
        dictionary, to obtain a modified boat speed of that given
        in the polar diagram, based on the influences presented in
        the given dictionary, such as wave height, underlying currents etc.

        Parameters
        ----------
        pd : PolarDiagram

        influence_data : Data or dict or list of Data or list of dict
            Further influences to be considered.

        Returns
        -------
        speeds : float or list of floats
            The boat speed if `influence_data` contained values,
            a list of respective boat speeds if `influence_data` contained
            lists.
        """

    def fit(self, training_data):
        """
        This method can be overwritten in order
        to fit parameters of the influence
        model to given training data.

        Parameters
        ----------
        training_data : Data
        """


class IdentityInfluenceModel(InfluenceModel):
    """An influence model which ignores most influences and just calculates
    the true wind if only the apparent wind is given.
    IF 'BSP' is not provided by the data, 'SOG' is used instead."""

    def remove_influence(self, data: dict):
        """
        Ignores most influences as described above.

        Parameters
        ----------
        data : dict
            Data dictionary, must either provide 'BSP' or 'SOG' key as well as
            either the keys 'TWS', 'TWA' or 'AWS', 'AWA'.

        Returns
        -------
        (n,3)-array, {}

        See also
        --------
        `InfluenceModel.remove_influence`
        """
        return _get_true_wind_data(data)

    def add_influence(self, pd, influence_data: dict):
        """
        Ignores most influences and strictly calculates the boat speed using the
        polar diagram.

        Parameters
        ----------
        pd : PolarDiagram

        influence_data : dict
            Either a dictionary of lists or a dictionary of values containing
            one or more sets of influence data.
            At least the keys 'TWS' and 'TWA' should be provided.

        See also
        --------
        `InfluenceModel.add_influence`
        """
        if isinstance(influence_data["TWS"], list):
            wind = zip(influence_data["TWS"], influence_data["TWA"])
            speed = [pd(ws, wa) for ws, wa in wind]
        else:
            ws, wa = influence_data["TWS"], influence_data["TWA"]
            speed = pd(ws, wa)
        return speed

    def fit(self, training_data: dict):
        """
        Does nothing.

        See also
        --------
        `InfluenceModel.fit`
        """


class WindAngleCorrectingInfluenceModel(InfluenceModel):
    """
    An influence model which corrects a structural measurement error in 'TWA'.

    Parameters
    ----------
    wa_shift: int or float, optional
        Difference between real wind angle and measured wind angle (correction value).

        Defaults to 0.

    interval_size: int or float, optional
        Interval size used in the fitting method.

        Defaults to 30.
    """

    def __init__(self, interval_size=30, wa_shift=0):
        super().__init__()
        self._wa_shift = wa_shift
        self._interval_size = interval_size

    def remove_influence(self, data):
        """
        Removes the correction value from the data.

        See also
        -------
        `InfluenceModel.remove_influence`
        """
        wind_data = _get_true_wind_data(data)
        wa = wind_data[:, 1]
        wind_data[:, 1] = (wa - self._wa_shift) % 360
        return wind_data

    def add_influence(self, pd, influence_data):
        """
        Adds the correction value to the data.

        See also
        -------
        `InfluenceModel.add_influence`

        """
        if isinstance(influence_data["TWS"], (list, np.ndarray)):
            wind = zip(influence_data["TWS"], influence_data["TWA"])
            speed = [pd(ws, (wa + self._wa_shift) % 360) for ws, wa in wind]
        else:
            ws, wa = influence_data["TWS"], influence_data["TWA"]
            speed = pd(ws, wa)
        return speed

    def fit(self, training_data):
        """
        The wind angle with the lowest density of measured wind angles is assumed to be the
        actual zero. The data density is computed using gauss kernel functions
        :math:`e^{\frac{wa - wa'}{l}}`, where :math:`l` can be seen as the size of
        an interval on which the gauss kernel is stretched.

        Parameter
        ---------
        training_data: Data containing key "TWA"

        See also
        --------
        `InfluenceModel.fit`
        """
        wind_angles = training_data["TWA"]
        wind_speeds = training_data["TWS"]
        sample = np.linspace(0, 359, 10 * 360)
        counts = [
            sum(
                other_ws
                * np.exp(
                    -(((abs(wa - other_wa) % 360) / self._interval_size) ** 2)
                )
                for other_wa, other_ws in zip(wind_angles, wind_speeds)
            )
            for wa in sample
        ]
        min_angle, _ = min(zip(sample, counts), key=lambda x: x[1])
        self._wa_shift = min_angle
        self.set_statistics(wa_shift=min_angle)


def _get_true_wind_data(data: dict):
    speed = "BSP" if "BSP" in data else "SOG"
    if "TWA" in data and "TWS" in data:
        if isinstance(data["TWS"], list):
            return data_dict_to_numpy(data, ["TWS", "TWA", speed])

        return np.array([data["TWS"], data["TWA"], data[speed]])

    if "AWA" in data and "AWS" in data:
        apparent_data = data_dict_to_numpy(data, ["AWS", "AWA", speed])
        return convert_apparent_wind_to_true(apparent_data)

    raise InfluenceException(
        "No sufficient wind data is given in order to apply influence"
        "model. Either give 'AWA' and 'AWS' or 'TWA' and 'TWS'"
    )
