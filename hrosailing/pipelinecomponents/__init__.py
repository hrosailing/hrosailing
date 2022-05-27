"""
Components for the PolarPipeline and PipelineExtension
classes among other things
"""

from .datahandler import (
    ArrayHandler,
    CsvFileHandler,
    DataHandler,
    NMEAFileHandler,
)
from .filter import BoundFilter, Filter, QuantileFilter
from .smoother import Smoother, LazySmoother
from .influencemodel import InfluenceModel, IdentityInfluenceModel
from .interpolator import (ArithmeticMeanInterpolator, IDWInterpolator,
                           ImprovedIDWInterpolator, Interpolator)
from .neighbourhood import (Ball, Cuboid, Ellipsoid, Neighbourhood, Polytope,
                            ScalingBall)
from .regressor import LeastSquareRegressor, ODRegressor, Regressor

from .sampler import (ArchimedianSampler, FibonacciSampler, Sampler,
                      UniformRandomSampler)
from .weigher import (CylindricMeanWeigher, CylindricMemberWeigher, Weigher,
                      WeightedPoints, AllOneWeigher,
                      PastFutureFluctuationWeigher, PastFluctuationWeigher)
from .imputator import FillLocalImputator
from .injector import ZeroInjector
from .quality_assurance import QualityAssurance, MinimalQualityAssurance
