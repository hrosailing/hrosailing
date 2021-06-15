

import numpy as np
import scipy.odr.odrpack as odrpack


from abc import ABC, abstractmethod
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted


class Regressor(ABC):

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def set_weights(self, weights):
        pass

    @property
    @abstractmethod
    def model_func(self):
        pass

    @property
    @abstractmethod
    def optimal_params(self):
        pass


class ODRegressor(Regressor):

    def __init__(self, model_func, init_values,
                 X_weights=None, y_weights=None,
                 max_it=1000):

        self._func = model_func
        self._model = odrpack.Model(model_func)
        self._init_vals = init_values
        self._weights_X = X_weights
        self._weights_y = y_weights
        self._maxit = max_it
        self.X_ = None
        self.y_ = None
        self._popt = None

    def __call__(self, data):
        return self._func(
            self._popt, np.row_stack(data[:, 0], data[:, 1]))

    def set_weights(self, weights):
        self._weights_X = weights

    @property
    def model_func(self):
        return self._func

    @property
    def optimal_params(self):
        return self._popt

    def fit(self, data):
        data = np.asarray(data)
        X, y = check_X_y(data[:, :2], data[:, 2])

        odr_data = odrpack.Data((X[:, 0], X[:, 1]), y,
                                wd=self._weights_X,
                                we=self._weights_y)
        odr = odrpack.ODR(odr_data, self._model,
                          beta0=self._init_vals,
                          maxit=self._maxit)
        odr.set_job(fit_type=2)
        out = odr.run()

        self._popt = out.beta
        self.X_ = X
        self.y_ = y

    def predict(self, data):
        data = np.asarray(data)
        check_is_fitted(self, ['X_', 'y_'])
        data = check_array(data)
        return self(data)


class LeastSquareRegressor(Regressor):

    def __init__(self):
        pass

    def set_weights(self, weights):
        pass

    @property
    def model_func(self):
        return

    @property
    def optimal_params(self):
        return

    def fit(self, data):
        pass

    def predict(self, data):
        pass
