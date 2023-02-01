"""
Classes used for modular modelling of different regression methods.

Defines the `Regressor` abstract base class that can be used to create
custom regression methods.

Subclasses of `Regressor` can be used with the `CurveExtension` class
in the `hrosailing.pipeline` module.
"""


import inspect
import logging.handlers
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy.odr.odrpack import ODR, Data, Model
from scipy.optimize import curve_fit

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.handlers.TimedRotatingFileHandler(
            "hrosailing.log", when="midnight"
        )
    ],
)
logger = logging.getLogger(__name__)


class Regressor(ABC):
    """Base class for all regressor classes.


    Abstract Methods
    ----------------
    model_func

    optimal_params

    fit(self, data)
    """

    @property
    @abstractmethod
    def model_func(self):
        """
        This property should return a version of the model function used in the regression.
        """

    @property
    @abstractmethod
    def optimal_params(self):
        """This property should return a version of the optimal parameters determined
        through regression of the model function.
        """

    @abstractmethod
    def fit(self, data):
        """This method should, given data, be used to determine
        optimal parameters for the model function.

        Parameters
        ----------
        data : array_like of shape (n, 3)
            Data to which the model function will be fitted, given as
            a sequence of points consisting of wind speed, wind angle
            and boat speed.
        """


class ODRegressor(Regressor):
    """An orthogonal distance regressor based on `scipy.odr.odrpack`.

    Parameters
    ----------
    model_func : function
        The function to be fitted.

        The function signature should be `f(ws, wa, *params) -> bsp`,
        where `ws` and `wa` are `numpy.ndarrays` resp. and `params` is a
        sequence of parameters that will be fitted.

    init_values : array_like
        Initial guesses for the optimal parameters of `model_func`
        that are passed to the `scipy.odr.ODR` class.

    max_it : int, optional
        Maximum number of iterations done by `scipy.odr.ODR`.

        Defaults to `1000`.
    """

    def __init__(self, model_func: Callable, init_values, max_it=1000):
        def odr_model_func(params, x):
            ws = x[0, :]
            wa = x[1, :]
            return model_func(ws, wa, *params)

        self._func = model_func
        self._model = Model(odr_model_func)
        self._init_vals = init_values
        # self._weights_X = None
        # self._weights_y = None
        self._maxit = max_it
        self._popt = None

    @property
    def model_func(self):
        """Returns a read-only version of `self._func`."""
        return self._func

    @property
    def optimal_params(self):
        """Returns a read-only version of `self._popt`."""
        return self._popt

    def fit(self, data, _enable_logging=False):
        """Fits the model function to the given data, i.e. calculates
        the optimal parameters to minimize an objective
        function based on the given data.

        Parameters
        ----------
        data : array_like of shape (n, 3)
            Data to which the model function will be fitted, given as
            a sequence of points consisting of wind speed, wind angle
            and boat speed.

        See also
        --------
        [ODRPACK](https://docs.scipy.org/doc/external/odrpack_guide.pdf)
        """
        X, y = data[:, :2], data[:, 2]

        odr_data = Data((X[:, 0], X[:, 1]), y)
        odr = ODR(
            odr_data, self._model, beta0=self._init_vals, maxit=self._maxit
        )
        odr.set_job(fit_type=2)
        out = odr.run()

        self._popt = out.beta

        if _enable_logging:
            self._log_outcome_of_regression(out, y)

    def _log_outcome_of_regression(self, out, y):
        indep_vars = len(self._popt)
        dof = y.shape[0] - indep_vars
        chi_squared = np.sum(np.square(out.eps))
        mean = np.mean(y)
        sse = np.sum(np.square(y - mean))
        ssr = out.sum_square
        sst = ssr + sse
        logger.info(f"Sum of squared residuals: {ssr}")
        logger.info(f"Explained sum of squared residuals: {sse}")
        logger.info(f"Total sum of squared residuals: {sst}")
        logger.info(f"Sum of squared errors delta: {out.sum_square_delta}")
        logger.info(f"Sum of squared error eps: {out.sum_square_eps}")
        logger.info(f"R^2: {sse / sst}")
        logger.info(f"Degrees of freedom: {dof}")
        logger.info(
            f"R^2_corr: {sse / sst - indep_vars * (1 - sse / sst) / dof}"
        )
        logger.info(f"F_emp ={(sse / indep_vars) / (sst / dof)}")
        logger.info(f"Quasi-chi^2: {out.res_var}")
        logger.info(f"chi^2_min: {chi_squared}")
        logger.info(f"Reduced chi^2_min: {chi_squared / dof}")


class LeastSquareRegressor(Regressor):
    """A least square regressor based on `scipy.optimize.curve_fit`.

    Parameters
    ----------
    model_func : function or callable
        The function which describes the model and is to be fitted.

        The function signature should be `f(ws, wa, *params) -> bsp`,
        where `ws` and `wa` are `numpy.ndarrays` resp. and `params` is a
        sequence of parameters that will be fitted.

    init_vals : array_like, optional
        Initial guesses for the optimal parameters of `model_func`
        that are passed to `scipy.optimize.curve_fit`.

        Defaults to `None`.

    Properties
    ----------
    model_func : Callable
        Returns a read-only version of `self._func`.

    optimal_params : numpy.ndarray
        Returns a read-only version of `self._popt`.
    """

    def __init__(self, model_func: Callable, init_vals=None):
        self._func = model_func

        def fitting_func(wind, *params):
            wind = np.asarray(wind)
            ws, wa = wind[:, 0], wind[:, 1]
            return model_func(ws, wa, *params)

        sig = inspect.signature(model_func)
        args = [
            p.name
            for p in sig.parameters.values()
            if p.kind
            in [
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            ]
        ]

        if init_vals is None and len(args) < 3:
            init_vals = _determine_params(fitting_func)
        elif init_vals is None:
            init_vals = tuple(1 for _ in range(len(args)))

        self._fitting_func = fitting_func
        self._init_vals = init_vals
        self._popt = None

    @property
    def model_func(self):
        return self._func

    @property
    def optimal_params(self):
        return self._popt

    def fit(self, data, _enable_logging=False):
        """Fits the model function to the given data, i.e. calculates
        the optimal parameters to minimize the sum of the squares of
        the residuals.

        Parameters
        ----------
        data : array_like of shape (n, 3)
            Data to which the model function will be fitted, given as
            a sequence of points consisting of wind speed, wind angle
            and boat speed.

        See also
        --------
        [curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/\
        scipy.optimize.curve_fit.html)
        """
        X, y = data[:, :2], data[:, 2]

        self._popt = self._get_optimal_parameters(X, y)

        if _enable_logging:
            self._log_outcome_of_regression(X, y)

    def _get_optimal_parameters(self, X, y):
        optimal_parameters, _ = curve_fit(
            self._fitting_func, X, y, p0=self._init_vals
        )
        return optimal_parameters

    def _log_outcome_of_regression(self, X, y):
        sr = np.square(self._func(X[:, 0], X[:, 1], *self._popt) - y)
        ssr = np.sum(sr)
        mean = np.mean(y)
        sse = np.sum(np.square(y - mean))
        sst = ssr + sse
        indep_vars = len(self._popt)
        dof = y.shape[0] - indep_vars - 1
        chi_squared = np.sum(sr / y)

        logger.info(f"Model-function: {self._func}")
        logger.info(f"Optimal parameters: {self._popt}")
        logger.info(f"Max squared error: {np.max(sr)}")
        logger.info(f"Min squared error: {np.min(sr)}")
        logger.info(f"Sum of squared residuals: {ssr}")
        logger.info(f"Explained sum of squares: {sse}")
        logger.info(f"Total sum of squares: {sst}")
        logger.info(f"R^2: {sse / sst}")
        logger.info(f"Degrees of freedom: {dof}")
        logger.info(
            f"R^2_corr: {sse / sst - indep_vars * (1 - sse / sst) / dof}"
        )
        logger.info(f"F_emp ={(sse / indep_vars) / (sst / dof)}")
        logger.info(f"chi^2: {chi_squared}")
        logger.info(f"chi^2_red: {chi_squared / dof}")


def _determine_params(func):
    params = []
    while True:
        try:
            func(np.array([[0, 0]]), *params)
            break
        except (IndexError, TypeError):
            params.append(1)

    return params
