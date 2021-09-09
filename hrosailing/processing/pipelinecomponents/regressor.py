"""
Contains the baseclass for Regressors used in the CurveExtension class,
that can also be used to create custom Regressors.

Also contains two predefined and usable regressors, the ODRegressor
and the LeastSquareRegressor.
"""

# Author: Valentin F. Dannenberg / Ente


from abc import ABC, abstractmethod
import logging.handlers
from typing import Callable

import numpy as np
from scipy.odr.odrpack import Data, Model, ODR, OdrError
from scipy.optimize import curve_fit


logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    filename="hrosailing/logging/processing.log",
)

LOG_FILE = "hrosailing/logging/processing.log"

logger = logging.getLogger(__name__)
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when="midnight"
)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


class RegressorException(Exception):
    """Custom exception for errors that may appear whilst
    working with the Regressor class and subclasses
    """

    pass


class Regressor(ABC):
    """Base class for all regressor classes


    Abstract Methods
    ----------------
    model_func

    optimal_params

    set_weights(self, X_weights, y_weights)

    fit(self, data)
    """

    @property
    @abstractmethod
    def model_func(self):
        pass

    @property
    @abstractmethod
    def optimal_params(self):
        pass

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def set_weights(self, X_weights, y_weights):
        pass


class ODRegressor(Regressor):
    """An orthogonal distance regressor based on scipy.odr.odrpack

    Parameters
    ----------
    model_func : function
        The function which describes the model and is to be fitted.

        The function signature should be f(ws, wa, *params) -> bsp,
        where ws and wa are numpy.ndarrays resp. and params is a
        sequence of parameters that will be fitted

    init_values : array_like, optional
        Inital guesses for the optimal parameters of  model_func
        that are passed to the scipy.odr.ODR class

        Defaults to None

    max_it : int, optional
        Maximum number of iterations done by scipy.odr.ODR.

        Defaults to 1000
    """

    def __init__(self, model_func: Callable, init_values=None, max_it=1000):
        def odr_model_func(params, x):
            tws = x[0, :]
            twa = x[1, :]
            return model_func(tws, twa, *params)

        self._func = model_func
        self._model = Model(odr_model_func)
        self._init_vals = init_values
        self._weights_X = None
        self._weights_y = None
        self._maxit = max_it
        self._popt = None

    @property
    def model_func(self):
        return self._func

    @property
    def optimal_params(self):
        return self._popt

    def set_weights(self, X_weights, y_weights):
        pass

    def fit(self, data):
        """Fits the model function to the given data, ie calculates
        the optimal parameters to minimize an objective
        function based on the data, see also [ODRPACK](https://docs.scipy.org/doc/external/odrpack_guide.pdf)

        Parameters
        ----------
        data : array_like of shape (n, 3)
            Data to which the model function will  be fitted, given as
            a sequence of points consisting of wind speed, wind angle
            and boat speed

        Raises a RegressorException

        - if
        - if
        - if
        """
        X, y = data[:, :2], data[:, 2]

        try:
            odr_data = Data(
                (X[:, 0], X[:, 1]), y, wd=self._weights_X, we=self._weights_y
            )
            odr = ODR(
                odr_data, self._model, beta0=self._init_vals, maxit=self._maxit
            )
            odr.set_job(fit_type=2)
            out = odr.run()
        except (ValueError, OdrError) as err:
            raise RegressorException("Regression was unsuccessful") from err

        self._popt = out.beta

        logger.info(f"Modelfunction: {self._func}")
        logger.info(f"Optimal parameters: {self._popt}")

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
        logger.info(f"Quasi-χ^2: {out.res_var}")
        logger.info(f"χ^2_min: {chi_squared}")
        logger.info(f"Reduced χ^2_min: {chi_squared / dof}")


class LeastSquareRegressor(Regressor):
    """A least square regressor based on scipy.optimize.curve_fit

    Parameters
    ----------
    model_func : function or callable
        The function which describes the model and is to be fitted.

        The function signature should be f(ws, wa, *params) -> bsp,
        where ws and wa are  numpy.ndarrays resp. and params is a
        sequence of parameters that will be fitted

    init_vals : array_like ,optional
        Inital guesses for the optimal parameters of model_func
        that are passed to scipy.optimize.curve_fit

        Defaults to None
    """

    def __init__(self, model_func: Callable, init_vals=None):

        self._func = model_func
        self._init_vals = init_vals
        self._popt = None
        self._weights = None

    @property
    def model_func(self):
        return self._func

    @property
    def optimal_params(self):
        return self._popt

    def set_weights(self, X_weights, y_weights):
        pass

    def fit(self, data):
        """Fits the model function to the given data, ie calculates
        the optimal parameters to minimize the sum of the squares of
        the residuals, see also [least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)

        Parameters
        ----------
        data : array_like of shape (n, 3)
            Data to which the model function will be fitted, given as
            a sequence of points consisting of wind speed, wind angle
            and boat speed

        Raises a RegressorException if least-square minimization
        was not succesful, ie, if scipy.optimize.curve_fit
        raises a RuntimeError
        """
        X, y = data[:, :2], data[:, 2]
        X = np.ravel(X).T
        y = np.ravel(y).T

        try:
            self._popt, _ = curve_fit(
                self.model_func, X, y, p0=self._init_vals, sigma=self._weights
            )
        except RuntimeError as re:
            raise RegressorException(
                "Least-square minimization was unsuccesful"
            ) from re

        logger.info(f"Model-function: {self._func}")
        logger.info(f"Optimal parameters: {self._popt}")

        sr = np.square(self._func(X, *self._popt) - y)
        ssr = np.sum(sr)
        mean = np.mean(y)
        sse = np.sum(np.square(y - mean))
        sst = ssr + sse
        indep_vars = len(self._popt)
        dof = y.shape[0] - indep_vars - 1
        chi_squared = np.sum(sr / y)

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
        logger.info(f"χ^2: {chi_squared}")
        logger.info(f"χ^2_red: {chi_squared / dof}")
