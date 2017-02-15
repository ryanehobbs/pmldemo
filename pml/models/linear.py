import numpy as np
from models import LinearMixin
import solvers.linear as ls


class Linear(LinearMixin):

    def __init__(self, normalize=False, include_bias=True, solver='linear', iterations=10, alpha=0.01):
        """
        Create a linear model class for performing regression analysis
        :param normalize: (Default: False) Scale features in training data if they differ in order of magnitude
        :param include_bias: (Default: False) include bias column in linear model
        :param solver: Type of solver to use for linear regression calculation
        :param iterations: Number of iterations to perform on training set
        :param alpha: Learning rate to use when performing loss calculations
        """

        # call base class ctor
        super(Linear, self).__init__(normalize, include_bias, solver, iterations, alpha)

    def _cost_calc(self, X, y, theta=None):
        """
        Helper method that will calculate J(theta) cost and is helpful to evaluate that solvers such
        as gradient descent are correctly converging. Method calculates the minimized cost function.
        :param X: array-like Array[n_samples, n_features] Training data
        :param y: np.ndarray Vector[n_samples] Training labels
        :param theta: array-like Vector[n_features] coefficient parameters
        :return: Integer min->J(theta) cost
        """

        # The objective of linear regression is to minimize the cost function
        # the function J(theta) = 1/2m * sum(h_thetaX - y)^2 where the
        # h_thetaX is the linear model h_theta = theta0 + theta1 * X1

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
        #j_cost = 1/(2 * n_samples) * sumsqrd_err  # univariate
        j_cost = sumsqrd_err / (2 * n_samples)  # multivariate

        # return minimal cost calculation
        return j_cost

    def fit(self, X, y, theta=None):
        """
        Fit training data against a linear regression model
        :param X: array-like Array[n_samples, n_features] Training data
        :param y: np.ndarray Vector[n_samples] Training labels
        :param theta: array-like Vector[n_features] coefficient parameters
        :return: Integer minimized cost
        """

        if theta is not None and np.atleast_1d(theta).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        # fit intercept with theta params and bias if included
        X, y = self.fit_intercept(X, y, theta)

        # check solver type
        if self.solver == 'linear':
            self.theta_, self.grad_ = ls.gradient_descent(X, y, self.theta_, self.alpha, self.iterations, self._cost_calc)  # calc gradient descent
        elif self.solver == 'normal':
            self.theta_ = ls.linear_leastsquares(X, y)

class Logistic(object):

    pass