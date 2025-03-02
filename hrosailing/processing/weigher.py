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

from hrosailing.core.computing import (
    data_dict_to_numpy,
    euclidean_norm,
    safe_operation,
    scaled_norm,
)
from hrosailing.core.constants import NORM_SCALES
from hrosailing.core.data import Data
from hrosailing.core.statistics import ComponentWithStatistics


class Weigher(ComponentWithStatistics, ABC):
    """Base class for all weigher classes.
    Basic arithmetic operations may be performed among weighers.
    """

    @abstractmethod
    def weigh(self, points) -> (np.ndarray, dict):
        """This method should be used to determine a weight for each point
        where the points might be given as `Data` or as `numpy.ndarray` of floating point numbers storing records
        in a row wise fashion and return the resulting weights.

        The following arithmetic operations are supported for `Weigher` instances `weigher`, `weigher1`, `weigher2`,
         integers `k` and `a` of type `float`, `int` or array like over `float` or `int`:

        - `weigher1 + weigher2`, weigher + a`
        - `-weigher`, `weigher1 - weigher2`, `weigher - a`
        - `a*weigher`, `weigher1*weigher2`
        - `weigher1/weigher2`, `a/weigher`, `weigher/a`
        - `weigher**k`

        The result will be a weigher producing weights according to the used formula.

        Parameters
        ----------
        points : Data or numpy.ndarray
        """

    def __add__(self, other):
        if isinstance(other, Weigher):
            return _BinaryMapWeigher(self, other, lambda x, y: x + y)
        if isinstance(other, (int, float, np.ndarray)):
            return _UnaryMapWeigher(self, lambda x: x + other)
        raise TypeError(
            f"Invalid type for addition with Weigher."
            f"Expected Weigher, int, float or np.ndarray. Got {type(other)} instead")

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, Weigher):
            return _BinaryMapWeigher(self, other, lambda x, y: x * y)
        if isinstance(other, (int, float, np.ndarray)):
            return _UnaryMapWeigher(self, lambda x: x * other)
        raise TypeError(
            f"Invalid type for multiplication with Weigher."
            f"Expected Weigher, int, float or np.ndarray. Got {type(other)} instead")

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        if isinstance(other, Weigher):
            return _BinaryMapWeigher(self, other, lambda x, y: x - y)
        if isinstance(other, np.ndarray):
            return _UnaryMapWeigher(self, lambda x: x - other)
        raise TypeError(
            f"Invalid type for subtraction with Weigher."
            f"Expected Weigher or np.ndarray. Got {type(other)} instead")

    def __neg__(self):
        neg = _UnaryMapWeigher(self, lambda x: -x)
        return neg

    def __truediv__(self, other):
        if isinstance(other, Weigher):
            return _BinaryMapWeigher(self, other, lambda x, y: x / y)
        if isinstance(other, (int, float, np.ndarray)):
            return _UnaryMapWeigher(self, lambda x: x / other)
        raise TypeError(
            f"Invalid type for division with Weigher."
            f"Expected Weigher or np.ndarray. Got {type(other)} instead")

    def __rtruediv__(self, other):
        if isinstance(other, Weigher):
            return _BinaryMapWeigher(self, other, lambda x, y: y / x)
        if isinstance(other, np.ndarray):
            return _UnaryMapWeigher(self, lambda x: other / x)
        raise TypeError(
            f"Invalid type for division with Weigher."
            f"Expected Weigher or np.ndarray. Got {type(other)} instead")

    def __pow__(self, power, modulo=None):
        pow_ = _UnaryMapWeigher(self, lambda x: x ** power)
        return pow_

    def set_statistics(self, weights):
        """
        Produces hrosailing standard statistics for weighers, namely the average,
        minimal and maximal weight as well as a list of percentages of how many
        weights are contained in each 10th of the span between the minimal and
        maximal weight.
        The respective keys are `"average_weight"`, `"minimal_weight"`, `"maximal_weight"` and
        `"quantiles"`.

        Parameters
        ----------
        weights : array_like of floats
            The weights to be analyzed.
        """
        minw = safe_operation(np.min, weights)
        maxw = safe_operation(np.max, weights)
        span = safe_operation(lambda x: x[1] - x[0], [minw, maxw])

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
            average_weight=safe_operation(np.mean, weights),
            minimal_weight=minw,
            maximal_weight=maxw,
            quantiles=safe_operation(get_quantiles, [minw, span]),
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
    of the pipeline are set to `False`. Weighs everything as `1`.

    See also
    ----------
    `Weigher`
    """

    def weigh(self, points) -> (np.ndarray, dict):
        """
        Assigns weights according to the procedure described above.

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
    of the d-th component of all those points and set
    :math:`w_p = | m_p - p[d-1] | / std_p`.

    This weigher can only be used on floating point data.

    Parameters
    ----------
    radius : positive int or float, optional
        The radius of the considered cylinder, with infinite height, i.e. :math:`r`.

        Defaults to `0.05`.

    norm : function or callable, optional
        Norm with which to evaluate the distances, i.e. :math:`||.||`.

        If nothing is passed, it will automatically detect a scaled euclidean norm with respect to the used attributes.

    dimensions : list of str, optional
        If not `None` and the input of `weigh` is of type `Data`, only the attributes mentioned in
        `attributes` will be considered.

        Defaults to `None`.

    See also
    ----------
    `Weigher`
    """

    def __init__(self, radius=0.05, norm=None, dimensions=None):
        super().__init__()
        if radius <= 0:
            raise ValueError(
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

        See also
        ----------
        `Weigher.weigh`
        """
        if isinstance(points, np.ndarray) and len(points) == 0:
            return []
        if isinstance(points, Data) and points.n_rows == 0:
            return []
        dimensions, points, bsps = _set_points_from_data(
            points, self._dimensions
        )
        weights = [
            self._calculate_weight(point,
                                   points=points,
                                   bsps=bsps,
                                   bsp=bsp,
                                   dimensions=dimensions)
            for point, bsp in zip(points, bsps)
        ]
        weights = np.array(weights)
        weights = 1 - _normalize(weights, np.max)

        self.set_statistics(weights)

        return weights

    def _calculate_weight(self, point, *, points, bsps, bsp, dimensions):
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
    :math:`|pt[-1] - p[-1]| \\leq h` and :math:`||pt[:d-1] - p[:d-1]|| \\leq r`.

    If we call the set of all such points `P`, then :math:`w_p = |P| - 1`
    normalized by dividing by the maximum weight.

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
        If the data is given as `Data`, `attributes` contains the keys
        which will be used in order to create the data space.
        If `None`, all keys of the given `Data` are used.

        Defaults to `None`.

    See also
    ----------
    `Weigher`
    """

    def __init__(self, radius=0.05, length=0.05, norm=None, dimensions=None):
        super().__init__()
        if radius <= 0:
            raise ValueError("`radius` is non-positive")

        if length < 0:
            raise ValueError("`length` should be non-negative.")

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

        See also
        ----------
        `Weigher.weigh`
        """

        if isinstance(points, np.ndarray) and len(points) == 0:
            return []
        if isinstance(points, Data) and points.n_rows == 0:
            return []

        self._dimensions, points = _set_points_from_data(
            points, self._dimensions, reduce=False
        )

        weights = [self._calculate_weight(point, points) for point in points]
        weights = np.array(weights)
        weights = _normalize(weights, np.max)

        self.set_statistics(weights)
        return weights

    def _calculate_weight(self, point, points):
        n_points_in_cylinder = self._count_points_in_cylinder(point, points)
        return n_points_in_cylinder - 1

    def _count_points_in_cylinder(self, point, points):
        height = np.abs(points[:, -1] - point[-1]) <= self._length
        radius = self._norm(points[:, :-1] - point[:-1]) <= self._radius

        cylinder = height & radius
        points_in_cylinder = cylinder[cylinder]
        return len(points_in_cylinder)


class FluctuationWeigher(Weigher):
    """
    A weigher that benefits data with (locally) small fluctuation.

    If a single attribute is used, we use the following procedure:

    For each data point, the weigher considers the standard deviation of the data restricted to an interval
    before and after the datestamp of the considered point.
    Given an upper bound on the standard deviation the weight is computed by a ReLu function mirrored and stretched
    such that a standard deviation of 0 gets the weight 1 and if the standard variation exceeds the upper bound, the
    weight is set to 0.

    If more than one attribute is used, the weights described above will be computed for each such attribute
    and will then be multiplied.

    Parameters
    ----------
    attributes : list of str
        The names of the numeric attributes to be considered.

    timespan : timedelta or tuple of two timedelta

        - If it is a single value, the weigher will consider the corresponding duration before the examined
        data point,
        - If it is a tuple, the weigher will consider points in the interval
        (`time` - `timespan[0]`, `time` + `timespan[1]`) where `time` is the timestamp of the currently examined data
        point.

    upper_bounds : list of int or float
        The upper bounds on the standard deviation for each considered attribute.

    See also
    ----------
    `Weigher`
    """

    def __init__(self, attributes, timespan, upper_bounds):
        super().__init__()
        if isinstance(timespan, timedelta):
            self._timespan_before = timespan
            self._timespan_after = timedelta(seconds=0)
        else:
            self._timespan_before = timespan[0]
            self._timespan_after = timespan[1]
        self._attributes = attributes
        self._upper_bounds = np.array(upper_bounds)

    def weigh(self, points):
        """
        Weighs points by the procedure described above.

        Parameters
        ----------
        points : Data
            Has to contain the key `"datetime"` and all keys contained in the `dimension` parameter during
            initialization. Fields which do not contain `float` values are ignored.

        See also
        --------
        `Weigher.weigh`
        """
        times = points["datetime"]
        dimensions, points = _set_points_from_data(
            points, self._attributes, False
        )
        upper_bounds = self._upper_bounds[
            [i for i, key in enumerate(self._attributes) if key in dimensions]
        ]

        weights = [1] * len(points)

        for curr_idx, (dt, pt) in enumerate(zip(times, points)):
            start_idx = min(
                i
                for i in range(len(points))
                if times[i] <= dt and dt - times[i] <= self._timespan_before
            )
            end_idx = max(
                i
                for i in range(len(points))
                if dt <= times[i] and times[i] - dt <= self._timespan_after
            )
            for col, ub in enumerate(upper_bounds):
                curr_pts = points[start_idx: end_idx + 1, col]
                std = np.std(curr_pts)
                if std > ub:
                    weights[curr_idx] = 0
                else:
                    weights[curr_idx] *= (ub - std) / ub

        self.set_statistics(weights)

        return weights


def _set_points_from_data(data, attributes, reduce=True):
    if isinstance(data, np.ndarray):
        if reduce:
            bsp = data[:, -1]
            data = data[:, :-1]
            return attributes, data, bsp
        return attributes, data

    if attributes is None:
        attributes = list(data.keys()) or list(data.keys)

    if reduce:
        if "BSP" in attributes:
            bsp = data["BSP"]
            attributes.remove("BSP")
        elif "SOG" in attributes:
            bsp = data["SOG"]
            attributes.remove("SOG")
        else:
            raise ValueError(
                "Data has to support one of the keys 'BSP' or 'SOG'"
            )

    if isinstance(data, dict):
        data = data_dict_to_numpy(data, attributes)

    if isinstance(data, Data):
        attributes, data = data[attributes].numerical

    if reduce:
        return attributes, data, np.asarray(bsp)
    return attributes, data


class FuzzyBool:
    """
    Class representing a fuzzy truth statement, i.e. a function with values
    between 0 and 1 (truth function).
    The truth function can be evaluated by calling the instance.
    Supports the operators `&`, `|`, `~` for the operators and, or, not.
    If `fuzzy_bool` is a `FuzzyBool` instance, then `fuzzy_bool[item]` will return a `FuzzyBool` instance
    evaluating the respective truth function on `x[item]` instead of `x` for a given input `x`.

    The easiest way to initialize a `FuzzyBool` object is by using the operators of the
    `FuzzyVariables` class, but it can also be initialized with a custom truth
    function.

    Also supports the `str` method.

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

    - `x < a`, `x <= a`, `x > a`, `x >= a`, `x == a`
            refers to the respective truth function
            (using sigmoid activation function),

    - `x(s) < a`, ... (same as above, but with sharpness `s` used),

    - `x[key] < a`, ... (same as above, but the truth function will be
            applied after getting the item referenced by `key`),

    - `x[key](s) < a`, ... (the both notations above combined),

    - `(x < a) & (x > b)`, ... ('and' concatenation),

    - `(x < a) | (x > b)`, ... ('or' concatenation),

    - `~(x < a)`, ... ('not' operation).

    Also supports the methods `repr` and `str`.

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
        """Sharpness that will be used for the next generated `FuzzyBool` instance."""
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
    `FuzzyBool`, `FuzzyVariable`, `Weigher`
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
                " `hrosailing.core.data.Data` objects, got"
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
        If set to `None`, a two-dimensional norm with scalings 1/20 (for wind speed) and 1/360 (for wind angle) is
        returned.

        Defaults to `None`.

    Returns
    ---------
    norm : callable
        The euclidean norm, scaled by coefficients corresponding to
        `constants.NORM_SCALES`.
    """
    if dimensions is None:
        dimensions = ["TWS", "TWA"]
    scales = [
        NORM_SCALES[key] if key in NORM_SCALES else 1 for key in dimensions
    ]
    return scaled_norm(euclidean_norm, scales)
