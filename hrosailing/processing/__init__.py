"""
Interfaces and classes for data processing.

Subclasses can be used with the `PolarPipeline` class
of the `hrosailing.pipeline` module.
"""

from .datahandler import (
    ArrayHandler,
    CsvFileHandler,
    DataHandler,
    NMEAFileHandler,
)
from .filter import BoundFilter, Filter, QuantileFilter
from .imputator import FillLocalImputator, RemoveOnlyImputator
from .injector import ZeroInjector
from .interpolator import (
    ArithmeticMeanInterpolator,
    IDWInterpolator,
    ImprovedIDWInterpolator,
    Interpolator,
)
from .neighbourhood import (
    Ball,
    Cuboid,
    Ellipsoid,
    Neighbourhood,
    Polytope,
    ScalingBall,
)
from .regressor import LeastSquareRegressor, ODRegressor, Regressor
from .sampler import (
    ArchimedeanSampler,
    FibonacciSampler,
    Sampler,
    UniformRandomSampler,
)
from .smoother import AffineSmoother, LazySmoother, Smoother
from .weigher import (
    AllOneWeigher,
    CylindricMeanWeigher,
    CylindricMemberWeigher,
    FluctuationWeigher,
    FuzzyVariable,
    FuzzyWeigher,
    Weigher,
)

__all__ = [
    "ArrayHandler",
    "CsvFileHandler",
    "DataHandler",
    "NMEAFileHandler",
    "ArithmeticMeanInterpolator",
    "IDWInterpolator",
    "ImprovedIDWInterpolator",
    "Interpolator",
    "Ball",
    "Cuboid",
    "Ellipsoid",
    "Neighbourhood",
    "Polytope",
    "ScalingBall",
    "LeastSquareRegressor",
    "ODRegressor",
    "Regressor",
    "ArchimedeanSampler",
    "FibonacciSampler",
    "Sampler",
    "UniformRandomSampler",
    "AffineSmoother",
    "LazySmoother",
    "Smoother",
    "AllOneWeigher",
    "CylindricMeanWeigher",
    "CylindricMemberWeigher",
    "FluctuationWeigher",
    "FuzzyVariable",
    "FuzzyWeigher",
    "Weigher",
]
