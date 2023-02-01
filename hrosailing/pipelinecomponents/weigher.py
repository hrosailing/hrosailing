"""
Contains the baseclass `Weigher` used in the `PolarPipeline` class
that can also be used to create custom weighers.

Also contains two predefined and usable weighers, the `CylindricMeanWeigher`
and the `CylindricMemberWeigher`, as well as the `WeightedPoints` class, used to
represent data points together with their respective weights.
"""


from abc import ABC, abstractmethod
from datetime import timedelta

import numpy as np

from hrosailing.pipelinecomponents.constants import NORM_SCALES
from hrosailing.pipelinecomponents.data import Data

from ._utils import (
    ComponentWithStatistics,
    _safe_operation,
    data_dict_to_numpy,
    euclidean_norm,
    scaled_norm,
)


class WeightedPointsInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of `WeightedPoints`.
    """


class WeigherInitializationException(Exception):
    """Exception raised if an error occurs during
    initialization of a `Weigher`.
    """


class WeighingException(Exception):
    """Exception raised if an error occurs during the calling
    of the `.weigh()`-method."""


class WeightedPoints:
    """A class to weigh data points and represent them together
    with their respective weights.

    Parameters
    ----------
    data : Data, dict or numpy.ndarray
        Points that will be weight or paired with given weights.
        If given as a dictionary, each value should be a list. Data points are interpreted as
        `[data[key][i] for key in data.keys()]` for each suitable `i`.
        If given as a `numpy.ndarray`, the rows will be interpreted as data points.

    weights : scalar or array_like of shape (n,)
        If the weights of the points are known beforehand,
        they can be given as an argument. If weights are
        passed, they will be assigned to the points
        and no further weighing will take place.

        If a scalar is passed, the points will all be assigned
        the same weight.
    """

    def __init__(self, data, weights):
        self.data = data
        if isinstance(weights, (float, int)):
            if isinstance(data, dict):
                length = len(list(data.values())[0])
            else:
                length = len(data)
            self.weights = weights * np.ones(length)
        else:
            self.weights = np.asarray(weights)

    def __getitem__(self, mask):
        if isinstance(self.data, dict):
            return WeightedPoints(
                data={
                    key: list(np.array(value)[mask])
                    for key, value in self.data.items()
                },
                weights=self.weights[mask],
            )
        return WeightedPoints(data=self.data[mask], weights=self.weights[mask])

    def extend(self, other):
        """
        Extends the weighted points by other weighted points.
        The value of the `data` attribute should be of the same type in both respective `WeightedPoints` objects.
        If both data is given as a dictionary of lists, the respective lists
        will be extended.
        Keys that are not present in both dictionaries are discarded.

        Parameters
        ----------
        other : WeightedPoints
            Points to be appended.
        """

        if isinstance(self.data, dict):
            self.data = {
                key: value + other.data[key]
                for key, value in self.data.items()
                if key in other.data
            }
        elif isinstance(self.data, Data):
            self.data = Data.concatenate([self.data, other.data])
        else:
            self.data = np.row_stack([self.data, other.data])

        self.weights = np.concatenate([self.weights, other.weights])


class Weigher(ComponentWithStatistics, ABC):
    """Base class for all weigher classes.
    Basic arithmetic operations may be performed among weighers.


    Abstract Methods
    ----------------
    weigh(self, points)
    """

    @abstractmethod
    def weigh(self, points) -> (np.ndarray, dict):
        """This method should be used to determine a weight for each point
        where the points might be given as `Data` or as `numpy.ndarray` of floating point numbers storing records
        in a row wise fashion and return the resulting weights.

        Parameters
        ----------
        points : Data or numpy.ndarray
        """

    def __add__(self, other):
        if isinstance(other, Weigher):
            sum_ = _BinaryMapWeigher(self, other, lambda x, y: x + y)
        elif isinstance(other, (int, float, np.ndarray)):
            sum_ = _UnaryMapWeigher(self, lambda x: x + other)
        return sum_

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, Weigher):
            prod_ = _BinaryMapWeigher(self, other, lambda x, y: x * y)
        elif isinstance(other, (int, float, np.ndarray)):
            prod_ = _UnaryMapWeigher(self, lambda x: x * other)
        return prod_

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        if isinstance(other, Weigher):
            sub = _BinaryMapWeigher(self, other, lambda x, y: x - y)
        elif isinstance(other, np.ndarray):
            sub = _UnaryMapWeigher(self, lambda x: x - other)
        return sub

    def __neg__(self):
        neg = _UnaryMapWeigher(self, lambda x: -x)
        return neg

    def __truediv__(self, other):
        if isinstance(other, Weigher):
            div = _BinaryMapWeigher(self, other, lambda x, y: x / y)
        elif isinstance(other, (int, float, np.ndarray)):
            div = _UnaryMapWeigher(self, lambda x: x / other)
        return div

    def __rtruediv__(self, other):
        if isinstance(other, Weigher):
            div = _BinaryMapWeigher(self, other, lambda x, y: y / x)
        elif isinstance(other, np.ndarray):
            div = _UnaryMapWeigher(self, lambda x: other / x)
        return div

    def __pow__(self, power, modulo=None):
        pow_ = _UnaryMapWeigher(self, lambda x: x**power)
        return pow_

    def set_statistics(self, weights):
        """
        Produces hrosailing standard statistics for weighers, namely the average,
        minimal and maximal weight as well as a list of percentages of how many
        weights are contained in each 10th of the span between the minimal and
        maximal weight.
        The respective keys are `average_weight`, `minimal_weight`, `maximal_weight` and
        `quantiles`.

        Parameters
        ----------
        weights : array_like of floats
            The weights to be analyzed.
        """
        minw = _safe_operation(np.min, weights)
        maxw = _safe_operation(np.max, weights)
        span = _safe_operation(lambda x: x[1] - x[0], [minw, maxw])

        def get_quantiles(args):
            minw, span = args
            return [
                round(
                    100
                    * len(
                        [
                            w
                            for w in weights
                            if (
                                minw + span * (i + 1) / 10
                                >= w
                                >= minw + span * i / 10
                            )
                        ]
                    )
                    / len(weights),
                    2,
                )
                for i in range(10)
            ]

        super().set_statistics(
            average_weight=_safe_operation(np.mean, weights),
            minimal_weight=minw,
            maximal_weight=maxw,
            quantiles=_safe_operation(get_quantiles, [minw, span]),
        )


class _UnaryMapWeigher(Weigher):
    def __init__(self, sub_weigher, map_):
        super().__init__()
        self._sub_weigher = sub_weigher
        self._map = map_

    def weigh(self, points) -> (np.ndarray, dict):
        weights, _ = self._sub_weigher.weigh(points)
        self.set_statistics(weights)
        return self._map(weights)


class _BinaryMapWeigher(Weigher):
    def __init__(self, sub_weigher, other_sub_weigher, map_):
        super().__init__()
        self._sub_weigher = sub_weigher
        self._other_sub_weigher = other_sub_weigher
        self._map = map_

    def weigh(self, points) -> (np.ndarray, dict):
        weights, _ = self._sub_weigher.weigh(points)
        other_weights, _ = self._other_sub_weigher.weigh(points)
        new_weights = self._map(weights, other_weights)
        self.set_statistics(new_weights)
        return new_weights


class AllOneWeigher(Weigher):
    """A weigher that functions mainly as a stand in if the weighing options
    of the pipeline are set to `False`. Weighs everything as `1`."""

    def weigh(self, points) -> (np.ndarray, dict):
        """
        Assigns weights according to the procedure defined above.

        See also
        --------
        `Weigher.weigh`
        """
        if isinstance(points, np.ndarray):
            size = len(points)
        else:
            size = points.n_rows
        return np.ones(size)


class CylindricMeanWeigher(Weigher):
    """A weigher that weighs given points according to the
    following procedure:

    For a given point `p` and points `pts` we look at all the points
    `pt` in `pts` such that :math:`||pt[:d-1] - p[:d-1]|| \\leq r`.

    Then we take the mean `m_p` and standard deviation `std_p`
    of the dth component of all those points and set
    :math:`w_p = | m_p - p[d-1] | / std_p`.

    This weigher can only be used on floating point data.

    Parameters
    ----------
    radius : positive int or float, optional
        The radius of the considered cylinder, with infinite height, i.e. :math:`r`.

        Defaults to `0.05`.

    norm : function or callable, optional
        Norm with which to evaluate the distances, i.e. :math:`||.||`.

        If nothing is passed, it will automatically detect a scaled euclidean norm with respect to the used dimensions.

    Raises
    ------
    WeigherInitializationException
        If radius is non-positive.
    """

    def __init__(self, radius=0.05, norm=None, dimensions=None):
        super().__init__()
        if radius <= 0:
            raise WeigherInitializationException(
                f"Invalid value for `radius`: {radius}. Non-positive number."
                " Use a positive value for `radius`."
            )

        self._radius = radius

        self._norm = norm
        self._dimensions = dimensions

    def __repr__(self):
        return (
            f"CylindricMeanWeigher(radius={self._radius}, "
            f"norm={self._norm.__name__})"
        )

    def weigh(self, points):
        """Weighs given points according to the procedure described above.

        Parameters
        ----------
        points : numpy.ndarray of shape (n, d) with d>=3 or Data
            Points to be weighed. Data has to be given in type `float`.

        Returns
        -------
        WeightedPoints : numpy.ndarray of shape (n,)
            Normalized weights of the input points.
        """
        dimensions, points, bsps = _set_points_from_data(
            points, self._dimensions
        )
        weights = [
            self._calculate_weight(point, points, bsps, bsp, dimensions)
            for point, bsp in zip(points, bsps)
        ]
        weights = np.array(weights)
        weights = 1 - _normalize(weights, np.max)

        self.set_statistics(weights)

        return weights

    def _calculate_weight(self, point, points, bsps, bsp, dimensions):
        points_in_cylinder = self._determine_points_in_cylinder(
            point, points, bsps, dimensions
        )

        std = _standard_deviation_of(points_in_cylinder)
        mean = _mean_of(points_in_cylinder)

        return np.abs(mean - bsp) / std

    def _determine_points_in_cylinder(self, point, points, bsps, dimensions):
        if self._norm is None:
            self._norm = hrosailing_standard_scaled_euclidean_norm(dimensions)
        in_cylinder = self._norm(points - point) <= self._radius
        return bsps[in_cylinder]


def _standard_deviation_of(points_in_cylinder):
    return np.std(points_in_cylinder) or 1  # in case that there are no points


def _mean_of(points_in_cylinder):
    return np.mean(points_in_cylinder)


def _normalize(weights, normalizer):
    if len(weights) == 0:
        return weights
    normalizer_ = normalizer(weights)
    if normalizer_ == 0:
        return weights
    return weights / normalizer(weights)


class CylindricMemberWeigher(Weigher):
    """A weigher that weighs given points according to the
    following procedure:

    For a given point `p` and points `pts`
    we look at all the points `pt` in `pts` such that
    :math:`|pt[0] - p[0]| \\leq h` and :math:`||pt[1:] - p[1:]|| \\leq r`.

    If we call the set of all such points `P`, then :math:`w_p = |P| - 1`.

    Parameters
    ----------
    radius : positive int or float, optional
        The radius of the considered cylinder, i.e. :math:`r`.

        Defaults to `0.05`

    length : non-negative int or float, optional
        The height of the considered cylinder, i.e. :math:`h`.

        If length is 0, the cylinder is a d-1 dimensional ball.

        Defaults to `0.05`.

    norm : function or callable, optional
        Norm with which to evaluate the distances, i.e. :math:`||.||`.

        If nothing is passed, it will default to :math:`||.||_2`.

    dimensions : [str] or None, optional
        If the data is given as `Data`, `dimensions` contains the keys
        which should be used in order to create the data space.
        If `None`, all keys of the given `Data` are used.

        Defaults to `None`.

    Raises
    ------
    WeigherInitializationException
        If radius is non-positive.
    WeigherInitializationException
        If length is negative.
    """

    def __init__(self, radius=0.05, length=0.05, norm=None, dimensions=None):
        super().__init__()
        if radius <= 0:
            raise WeigherInitializationException("`radius` is non-positive")

        if length < 0:
            raise WeigherInitializationException(
                "`length` is not non-negative"
            )

        self._radius = radius
        self._length = length

        if norm is None:
            self._norm = hrosailing_standard_scaled_euclidean_norm(dimensions)
        else:
            self._norm = norm
        self._dimensions = dimensions

    def __repr__(self):
        return (
            f"CylindricMemberWeigher(radius={self._radius},"
            f"length={self._length}, norm={self._norm.__name__})"
        )

    def weigh(self, points):
        """Weighs given points according to the procedure described above.

        Parameters
        ----------
        points : numpy.ndarray of shape (n, d)
            Points to be weighed.

        Returns
        -------
        weights : numpy.ndarray of shape (n,)
            Normalized weights of the input points.
        """

        self._dimensions, points, bsps = _set_points_from_data(
            points, self._dimensions
        )

        weights = [self._calculate_weight(point, points) for point in points]
        weights = np.array(weights)
        weights = _normalize(weights, np.max)

        self.set_statistics(weights)
        return weights

    def _calculate_weight(self, point, points):
        points_in_cylinder = self._count_points_in_cylinder(point, points)
        return len(points_in_cylinder) - 1

    def _count_points_in_cylinder(self, point, points):
        height = np.abs(points[:, 0] - point[0]) <= self._length
        radius = self._norm(points - point) <= self._radius

        cylinder = height & radius
        points_in_cylinder = cylinder[cylinder]
        return points_in_cylinder


class FluctuationWeigher(Weigher):
    """
    A weigher that benefits data with (locally) small fluctuation.

    If a single attribute is used, we use the following procedure:

    For each data point, the weigher considers the standard variation of the data restricted to an interval
    before and after the datestamp of the considered point.
    Given an upper bound on the standard variation the weight is computed by a ReLu function mirrored and stretched
    such that a standard variation of 0 gets the weight 1 and if the standard variation exceeds the upper bound, the
    weight is set to 0.

    If more than one attribute is used, the weights described above will be computed for each such attribute
    and then multiplied.

    Parameters
    ----------
    dimensions : list of str
        The names of the numeric attributes to be considered.

    timespan : timedelta or tuple of two timedelta

        - If it is a single value, the weigher will consider the corresponding duration before the examined
        data point,
        - If it is a tuple, the weigher will consider points in the interval
        (`time` - `timespan[0]`, `time` + `timespan[1]`) where `time` is the timestamp of the currently examined data
        point.

    upper_bounds : list of int or float
        The upper bounds on the standard deviation for each considered attribute.
    """

    def __init__(self, dimensions, timespan, upper_bounds):
        super().__init__()
        if isinstance(timespan, timedelta):
            self._timespan_before = timespan
            self._timespan_after = timedelta(seconds=0)
        else:
            self._timespan_before = timespan[0]
            self._timespan_after = timespan[1]
        self._dimensions = dimensions
        self._upper_bounds = np.array(upper_bounds)

    def weigh(self, points):
        """
        Weighs points by the procedure described above.

        Parameters
        ----------
        points : Data
            Should contain the key "datetime" and all keys contained in the "dimension" parameter during initialization.
            Fields which do not contain `float` values are ignored.

        See also
        --------
        `Weigher.weigh`
        """
        times = points["datetime"]
        dimensions, points = _set_points_from_data(
            points, self._dimensions, False
        )
        upper_bounds = self._upper_bounds[
            [i for i, key in enumerate(self._dimensions) if key in dimensions]
        ]

        weights = [1] * len(points)
        start_idx = 0

        for curr_idx, (dt, pt) in enumerate(zip(times, points)):
            while dt - times[start_idx] > self._timespan_before:
                start_idx += 1
            end_idx = start_idx
            while (
                end_idx + 1 < len(times)
                and times[end_idx + 1] - dt < self._timespan_after
            ):
                end_idx += 1
            for col, ub in enumerate(upper_bounds):
                curr_pts = points[start_idx : end_idx + 1, col]
                std = np.std(curr_pts)
                if std > ub:
                    weights[curr_idx] = 0
                else:
                    weights[curr_idx] *= (ub - std) / ub

        self.set_statistics(weights)

        return weights


def _set_points_from_data(data, dimensions, reduce=True):
    if isinstance(data, np.ndarray):
        if reduce:
            bsp = data[:, -1]
            data = data[:, :-1]
            return dimensions, data, bsp
        return dimensions, data

    if dimensions is None:
        dimensions = list(data.keys()) or list(data.keys)

    if reduce:
        if "BSP" in dimensions:
            bsp = data["BSP"]
            dimensions.remove("BSP")
        elif "SOG" in dimensions:
            bsp = data["SOG"]
            dimensions.remove("SOG")
        else:
            raise ValueError(
                "Data has to support one of the keys 'BSP' or 'SOG'"
            )

    if isinstance(data, dict):
        data = data_dict_to_numpy(data, dimensions)

    if isinstance(data, Data):
        dimensions, data = data[dimensions].numerical

    if reduce:
        return dimensions, data, np.asarray(bsp)
    return dimensions, data


class FuzzyBool:
    """
    Class representing a fuzzy truth statement, i.e. a function with values
    between 0 and 1 (truth function).
    Supports the operators `&`, `|`, `~` for the operators and, or, not.
    The easiest way to initialize a `FuzzyBool` object is by using the operators of the
    `FuzzyVariables` class, but it can also be initialized with a custom truth
    function.

    Parameters
    ----------
    eval_fun : callable
        The truth function used. Truth values between 0 and 1 are recommended.

    Attributes
    ----------
    repr : str
        A human readable representation of the `FuzzyBool` object.
        Is returned by `__str__`. First set to 'Fuzzy-Bool'.

    See also
    --------
    For recommendations on how to use a `FuzzyBool` see also `FuzzyVariable`.
    """

    def __init__(self, eval_fun):
        self._fun = eval_fun
        self.repr = "Fuzzy-Bool"

    def __call__(self, x):
        return self._fun(x)

    def __and__(self, other):
        new = FuzzyBool.fuzzy_and(self, other)
        new._concat_repr(self, other, "&")
        return new

    def __or__(self, other):
        new = FuzzyBool.fuzzy_or(self, other)
        new._concat_repr(self, other, "|")
        return new

    def __invert__(self):
        new = FuzzyBool.fuzzy_not(self)
        new.repr = f"~({self})"
        return new

    def __getitem__(self, item):
        def eval_fun(x):
            return self._fun(x[item])

        new = FuzzyBool(eval_fun)
        new.repr = f"{self}[{item}]"
        return new

    def __str__(self):
        return self.repr

    @classmethod
    def fuzzy_and(cls, one, other):
        """
        Parameters
        ----------
        one : FuzzyBool

        other : FuzzyBool

        Returns
        -------
        one_and_other : FuzzyBool
            Representation of the fuzzy 'and' operation of `one` and `other`
            realized via taking the minimum.
        """

        def eval_fun(x):
            concat = np.row_stack([one(x), other(x)])
            return np.min(concat, axis=0)[0]

        return cls(eval_fun)

    @classmethod
    def fuzzy_or(cls, one, other):
        """
        Parameters
        ----------
        one : FuzzyBool

        other : FuzzyBool

        Returns
        -------
        one_or_other : FuzzyBool
            Representation of the fuzzy 'or' operation of `one` and `other`
            realized via taking the maximum.
        """

        def eval_fun(x):
            concat = np.row_stack([one(x), other(x)])
            return np.max(concat, axis=0)[0]

        return cls(eval_fun)

    @classmethod
    def fuzzy_not(cls, one):
        """
        Parameters
        ----------
        one : FuzzyBool

        Returns
        -------
        not_one : FuzzyBool
            Representation of the fuzzy 'not' operation of `one`
            realized via taking the difference to 1.
        """
        return cls(lambda x: 1 - one(x))

    @classmethod
    def sigmoid(cls, center, sharpness, sigma):
        """
        Classical activation function.

        Parameters
        ----------
        center : int or float

        sharpness : int or float
            Controls the slope of the sigmoid function
            (higher sharpness yields higher slope).

        sigma : {1, -1}
            The direction of the sigmoid function; -1 yields the classical
            sigmoid, 1 yields the inverted sigmoid.

        Returns
        -------
        sigmoid : FuzzyBool
            A `FuzzyBool` object with truth function
            :math:`x → 1/(1+e^{sigma * sharpness * (x - center)})`.
        """

        def eval_fun(x):
            return 1 / (1 + np.exp(sigma * sharpness * (x - center)))

        new = cls(eval_fun)
        new.repr = (
            f"sigmoid(center={center},sharpness={sharpness},sigma={sigma})"
        )
        return new

    def _concat_repr(self, origin, other, concat):
        """
        Set the representation string to f"{origin} {concat} {other}". Use brackets if necessary.

        Parameters
        ----------
        origin : FuzzyBool

        other : FuzzyBool

        concat : str
        """
        self_repr = f"({origin})" if " " in f"{origin}" else f"{origin}"
        other_repr = f"({other})" if " " in f"{other}" else f"{other}"
        self.repr = f"{self_repr} {concat} {other_repr}"


class FuzzyVariable:
    """
    Refers to Variables in the fuzzy logic.
    It's main purpose is to easily create `FuzzyBool` instances.

    For example, the following notations work for a `FuzzyVariable` `x`, `int` or `float`
    Variables `a`, `b`, `s` and `key` such that `x.__getitem__(key)` works:

        - x < a, x <= a, x > a, x >= a, x == a
            refers to the respective truth function
            (using sigmoid activation function),

        - x(s) < a, ... (same as above, but with sharpness `s` used),

        - x[key] < a, ... (same as above, but the truth function will be
            applied after getting the item referenced by `key`),

        - x[key](s) < a, ... (the both notations above combined),

        - (x < a) & (x > b), ... ('and' concatenation),

        - (x < a) | (x > b), ... ('or' concatenation),

        - ~(x < a), ... ('not' operation).

    Parameters
    ----------
    key : None or str
        If `key` is not `None`, all generated `FuzzyBool` instances apply the
        truth function to `x`[`key`] instead of `x`.

        Defaults to `None`.

    sharpness : int
        Defines the default sharpness of all `FuzzyBool` instances generated
        by `FuzzyBool.sigmoid`.
        This sharpness will be used if no other sharpness is given via the
        `__call__`-method.

    Properties
    ----------
    sharpness : int
        The sharpness that will be used for the next generated `FuzzyBool` instance.

    See also
    --------
    `FuzzyBool`
    """

    def __init__(self, sharpness=10, key=None):
        self.key = key
        self._sharpness = sharpness
        self._next_sharpness = sharpness

    @property
    def sharpness(self):
        next_sharpness = self._next_sharpness
        self._next_sharpness = self._sharpness
        return next_sharpness

    def _truth(self, other, sigma):
        sigmoid = FuzzyBool.sigmoid(float(other), self.sharpness, sigma)
        if self.key is None:
            return sigmoid

        return sigmoid[self.key]

    def __gt__(self, other):
        new = self._truth(other, -1)
        new.repr = f"{self} > {other}"
        return new

    def __lt__(self, other):
        new = self._truth(other, 1)
        new.repr = f"{self} < {other}"
        return new

    def __ge__(self, other):
        return self > other

    def __le__(self, other):
        return self < other

    def __eq__(self, other):
        new = FuzzyBool.fuzzy_and(self < other, self > other)
        new.repr = f"{self} == {other}"
        return new

    def __getitem__(self, item):
        return FuzzyVariable(key=item, sharpness=self.sharpness)

    def __call__(self, sharpness):
        self._next_sharpness = sharpness
        return self

    def __str__(self):
        if self.key is None:
            return "x"
        return f"x[{self.key}]"

    def __repr__(self):
        if self.key is None:
            return f"x({self._sharpness})"
        return f"x({self._sharpness})[{self.key}]"


class FuzzyWeigher(Weigher):
    """
    Weigher that uses the truth function of a `FuzzyBool` object to create
    weights.

    Parameters
    ----------
    fuzzy : FuzzyBool
        The object wrapped around the truth function.

    See also
    ---------
    `FuzzyBool`, `FuzzyVariable`
    """

    def __init__(self, fuzzy):
        super().__init__()
        self.fuzzy = fuzzy

    def weigh(self, points):
        """
        See also
        --------
        `Weigher.weigh`
        """
        if isinstance(points, Data):
            weights = np.array([self.fuzzy(point) for point in points.rows()])
        elif isinstance(points, np.ndarray):
            weights = np.array([self.fuzzy(point) for point in points])
        else:
            raise TypeError(
                "FuzzyWeigher only takes numpy arrays or"
                " `hrosailing.pipelinecomponents.data.Data` objectsgot"
                f" {type(points)} instead."
            )
        self.set_statistics(weights)
        return weights


def hrosailing_standard_scaled_euclidean_norm(dimensions=None):
    """
    Returns a scaled euclidean norm function where the scales are chosen with respect to `constants.NORM_SCALES`
    (or 1 if `constants.NORM_SCALES` does not contain the respective key).

    Parameters
    ----------
    dimensions : iterator of str or None, optional
        Iterates over the names of the attributes used.
        If set to `None`, a two-dimensional norm with scalings 1/40 (for wind speed) and 1/360 (for wind angle) is
        returned.

        Defaults to `None`.
    """
    if dimensions is None:
        dimensions = ["TWS", "TWA"]
    scales = [
        NORM_SCALES[key] if key in NORM_SCALES else 1 for key in dimensions
    ]
    return scaled_norm(euclidean_norm, scales)
