

import numpy as np
import scipy.odr.odrpack as odrpack

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted


class ODREstimator(BaseEstimator):

    def __init__(self, o_func, init_vals, weights_X=None,
                 weight_y=None, max_iter=1000):
        self._o_func = o_func
        self._model = odrpack.Model(o_func)
        self._init_vals = init_vals
        self._weights_X = weights_X
        self._weight_y = weight_y
        self._max_iter = max_iter
        self._X = None
        self._y = None
        self._dof = None
        self._popt = None
        self._pcov = None
        self._out = None
        self._beta_cor = None
        self._sum_square = None
        self._sum_square_delta = None
        self._sum_square_eps = None
        self._inv_condnum = None
        self._rel_error = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        tws = X[:, 0]
        twa = X[:, 1]

        custom_data = odrpack.Data((tws, twa), y, wd=self._weights_X,
                                   we=self._weight_y)
        custom_odr = odrpack.ODR(custom_data, self._model,
                                 beta0=self._init_vals,
                                 maxit=self._max_iter)
        custom_odr.set_job(fit_type=2)
        custom_out = custom_odr.run()

        self._popt = custom_out.beta
        self._pcov = custom_out.cov_beta
        self._sum_square = custom_out.sum_square
        self._sum_square_delta = custom_out.sum_square_delta
        self._sum_square_eps = custom_out.sum_square_eps
        self._inv_condnum = custom_out.inv_condnum
        self._rel_error = custom_out.rel_error
        self._out = custom_out

        cov = self._pcov
        cor = np.copy(self._pcov)

        for i, row in enumerate(cov):
            for j in range(len(self._popt)):
                cor_i_j = cov[i, j] / np.sqrt(cov[i, j] * cov[j, i])
                cor[i, j] = cor_i_j

        self._beta_cor = cor
        self._X = X
        self._y = y

    def predict(self, X):
        X = np.asarray(X)
        check_is_fitted(self, ['_X', '_y'])
        X = check_array(X)

        tws = X[:, 0]
        twa = X[:, 1]

        return self._o_func(self._popt, np.row_stack((tws, twa)))
