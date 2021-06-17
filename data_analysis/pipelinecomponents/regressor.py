"""
Defines a baseclass for regressors used in the
data_analysis.processing.PolarPipeline class,
that can be used to create custom regressors for use.

Also contains various predefined and usable regressors
"""

# Author: Valentin F. Dannenberg / Ente


import numpy as np

from abc import ABC, abstractmethod
from scipy.odr.odrpack import Data, Model, ODR
from scipy.optimize import curve_fit


from exceptions import ProcessingException


class Regressor(ABC):
    """Base class for all
    regressor classes

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
    """An orthogonal distance
    regressor based on the
    scipy.odr.odrpack package

    Parameters
    ----------
    model_func : function
        The function which is to
        describes the model and
        is to be fitted.

        The function signature
        should be
        f(ws, wa, *params) -> bsp,
        where ws and wa are
        numpy.ndarrays resp. and
        params is a list of
        parameters that will be
        fitted.
    init_values : array_like, optional
        Inital guesses for the
        optimal parameters of
        model_func that are passed
        to the scipy.odr.ODR class

        Defaults to None
    max_it : int, optional
        Maximum number of iterations
        done by scipy.odr.ODR.

        Defaults to 1000

    Methods
    -------
    model_func
    optimal_params
    set_weight(self, X_weights, y_weights)
    fit(self, data)
    """

    def __init__(self, model_func, init_values=None,
                 max_it=1000):

        self._func = model_func

        def odr_model_func(params, x):
            tws = x[0, :]
            twa = x[1, :]
            return model_func(tws, twa, *params)

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
        """Fits the model
        function to the given
        data, ie calculates
        the optimal parameters
        to minimize an objective
        function based on the data,
        see also
        `ODRPACK <https://docs.scipy.org/doc/external/odrpack_guide.pdf>`_

        Parameters
        ----------
        data : array_like
            Data to which the
            model function will
            be fitted, given as
            a sequence of points
            consisting of wind speed
            wind angle and boat speed
        """
        X, y = _check_data(data)

        odr_data = Data((X[:, 0], X[:, 1]), y,
                        wd=self._weights_X,
                        we=self._weights_y)
        odr = ODR(odr_data, self._model,
                  beta0=self._init_vals,
                  maxit=self._maxit)
        odr.set_job(fit_type=2)
        out = odr.run()

        self._popt = out.beta


class LeastSquareRegressor(Regressor):
    """A least square regressor
    based on scipy.optimize.curve_fit

    Parameters
    ----------
    model_func : function or callable
        The function which is to
        describes the model and
        is to be fitted.

        The function signature
        should be
        f(ws, wa, *params) -> bsp,
        where ws and wa are
        numpy.ndarrays resp. and
        params is a list of
        parameters that will be
        fitted.

    init_vals : array_like ,optional
        Inital guesses for the
        optimal parameters of
        model_func that are passed
        to scipy.optimize.curve_fit

        Defaults to None
    """

    def __init__(self, model_func, init_vals=None):
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
        """Fits the model
        function to the given
        data, ie calculates
        the optimal parameters
        to minimize the sum
        of the squares of the
        residuals, see also
        `least squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_

        Parameters
        ----------
        data : array_like
            Data to which the
            model function will
            be fitted, given as
            a sequence of points
            consisting of wind speed
            wind angle and boat speed
        """

        X, y = _check_data(data)
        X = np.ravel(X).T
        y = np.ravel(y).T

        self._popt, _ = curve_fit(self.model_func, X, y,
                                  p0=self._init_vals,
                                  sigma=self._weights)


def _check_data(data):
    data = np.asarray(data)
    shape = data.shape
    if not data.size:
        raise ProcessingException("")
    if data.ndim != 2:
        raise ProcessingException("")
    if shape[1] != 3:
        try:
            data = data.reshape(-1, 3)
        except ValueError:
            raise ProcessingException("")

    if not np.all(np.isfinite(data)):
        raise ProcessingException("")

    return data[:, :2], data[:, 2]
