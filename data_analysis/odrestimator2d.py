

import numpy as np
import scipy.odr.odrpack as odrpack

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class ODREstimator2d(BaseEstimator):

    def __init__(self, o_func, init_vals,
                 std_x, std_y, max_iter=1000):
        self._max_iter = max_iter
        self._o_func = o_func

        def callable_o_func(x, *params):
            return self._o_func(x, *params)

        self._model = odrpack.Model(callable_o_func)
        self._init_vals = init_vals
        self._std_x = np.add(std_x, 0.0000000000001)
        self._std_y = np.add(std_y, 0.0000000000001)
        self._popt = None
        self._pcov = None
        self._sum_square = None
        self._sum_square_delta = None
        self._sum_square_eps = None
        self._inv_condnum = None
        self._rel_error = None
        self._out = None
        self._beta_cor = None
        self._X = None
        self._y = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        X, y = check_X_y(X, y)

        tws = X[:, 0]

        custom_data = odrpack.Data((tws, ), y)
        # custom_data = odrpack.RealData((tws, twa), y,
        #                                sx=self._std_x,
        #                                sy=self._std_y)
        custom_odr = odrpack.ODR(custom_data, self._model,
                                 beta0=self._init_vals,
                                 maxit=self._max_iter)

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
        cor = self._pcov.copy()

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

        return self._model.fcn(self._popt, (tws, ))
