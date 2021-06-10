

import numpy as np

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class NonLinearEstimator(BaseEstimator):

    def __init__(self, o_func):
        self._o_func = o_func
        self._X = None
        self._y = None
        self._popt = None
        self._pcov = None

    @property
    def objective_func(self):
        return self._o_func

    @property
    def optimal_parameters(self):
        return self._popt

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        X, y = check_X_y(X, y)
        X = np.ravel(X).T
        y = np.ravel(y).T

        self._popt, self._pcov = curve_fit(
            self.objective_func, X, y)

        self._calculate_chi_squared(X, y, self.objective_func,
                                    self.optimal_parameters, 0.3)

        self._X = X
        self._y = y

    def predict(self, X):
        X = np.asarray(X)
        check_is_fitted(self, ['_X', '_y'])
        X = check_array(X)

        return self.objective_func(X, *self.optimal_parameters)

    def _calculate_chi_squared(self, X, y, func, popt, xerror):
        chi_squared = np.sum(((func(X, *popt) - y) / xerror) ** 2)
        reduced_chi_squared = chi_squared / (len(X) - len(popt))
        print(chi_squared)
        print(reduced_chi_squared)
