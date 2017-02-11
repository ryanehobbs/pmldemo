import numpy as np
from models import LinearMixin
import solvers.linear as ls

class Linear(LinearMixin):

    def __init__(self, normalize=False, include_bias=False, fminfunc='linear', iterations=10, alpha=0.01):
        """
        Create a linear model class for performing regression analysis
        :param normalize:
        :param include_bias:
        """

        # call base class ctor
        super(Linear, self).__init__(normalize, include_bias, fminfunc, iterations, alpha)

    def _cost_calc(self, X, y, theta=None):
        """"""

        # reset class theta if theta is not None
        if theta is not None:
            self.theta_ = theta

        # get number of training samples
        n_samples = y.shape[0]
        # predict outcomes based on training data
        XP = self.predict(X)
        # squared error vectorized
        sumsqrd_err = np.sum(np.power(XP - y, 2))
        # get mean by dividing by count of samples and calc minimal cost
        j_cost = 1/(2 * n_samples) * sumsqrd_err
        # normalize ???
        #std = np.sqrt(np.sum(X ** 2, axis=0))
        #std[std==0] = 1
        #X /= std

        # return minimal cost calculation
        return j_cost


    def fit(self, X, y, theta=None):
        """
        Fit training data against a linear regression model
        :param X: array-like Array[n_samples, n_features] Training data
        :param y: np.ndarray Vector[n_samples] Training labels
        :param theta: array-like Vector[n_features] sample weights <-- these are actually coef weights are the ones
        :return: Integer minimized cost
        """

        if theta is not None and np.atleast_1d(theta).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        # fit intercept with theta params and bias if included
        X, y = self.fit_intercept(X, y, theta)

        # check solver type (fminfunc)
        if self.fminfunc == 'linear':
            self.theta_, self.grad_ = ls.gradient_descent(X, y, self.theta_, self.alpha, self.iterations, self._cost_calc)  # calc gradient descent


class Logistic(object):

    pass