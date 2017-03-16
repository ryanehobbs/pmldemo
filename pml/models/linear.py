import numpy as np
from models import LinearMixin
import solvers.linear as ls
from solvers import fminfunc
import mathutils.sigmoid as sigmoid

class Linear(LinearMixin):

    __metaclass__ = "Linear"
    def __init__(self, normalize=False, solver='linear', iterations=10, alpha=0.01):
        """
        Create a linear model class for performing regression analysis
        :param normalize: (Default: False) Scale features in training data if they differ in order of magnitude
        :param include_bias: (Default: False) include bias column in linear model
        :param solver: Type of solver to use for linear regression calculation
        :param iterations: Number of iterations to perform on training set
        :param alpha: Learning rate to use when performing loss calculations
        """

        # call base class ctor
        super(Linear, self).__init__(normalize, solver, iterations, alpha)

    def _cost_calc(self, X, y, theta=None):
        """
        Helper method that will calculate J(theta) cost and is helpful to evaluate solvers such
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
        # fit intercept for linear equation this is the hypothesis
        hX = self.docalc_slope(X, theta)
        # squared error vectorized
        sumsqrd_err = np.sum(np.power(hX - y, 2))
        # # calculate the minimized objective cost function for linear regression
        j_cost = 1/(2 * n_samples) * sumsqrd_err

        # return minimal cost calculation
        return j_cost

    def docalc_slope(self, X, theta):
        """
        Calculate the slope for the linear equation. This is also
        used as the method which will calculate the hypothesis for
        a linear model
        :param X:
        :param theta:
        :return:
        """

        return np.dot(X, theta)

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

        # pre fit data with theta params and bias if included
        X, y = self._pre_fit(X, y, theta)

        # check solver type
        if self.solver == 'linear':
            self.theta_ = ls.gradient_descent(X, y, self.theta_,
                                                          alpha=self.alpha, max_iter=self.iterations,
                                                          linearclass=self)
        elif self.solver == 'normal':
            self.theta_ = ls.linear_leastsquares(X, y)

    def predict(self, X):
        """
        Predict outcomes using a linear model against training data

        :param X: array-like Array[n_samples, n_features] Training data
        :return: Array [n_samples] predicted values
        """

        if not hasattr(self, 'theta_'):
            raise RuntimeError("Instance is currently not fitted")

        X = np.array(X)

        if self.normalize:  # predict based on normalized values
            X = self.__predictN(X)
        else:
            if self.include_bias:
                # if 0-dim array we need to add bias if necessary
                if len(X.shape) != 2:  # insert col ones on axis 0
                    X = np.insert(X, 0, 1, axis=0)

        # hypothesis in linear model: h_theta(x) = theta_zero + theta_one * x_one
        return np.sum(np.dot(X, self.theta_).astype(np.float32))

class Logistic(LinearMixin):

    __metaclass__ = "Logistic"
    def __init__(self, normalize=False, solver='logistic', iterations=10, alpha=0.001, lambda_r=None):
        """
        Create a linear model class for performing regression analysis
        :param normalize: (Default: False) Scale features in training data if they differ in order of magnitude
        :param include_bias: (Default: False) include bias column in linear model
        :param solver: Type of solver to use for linear regression calculation
        :param iterations: Number of iterations to perform on training set
        :param alpha: Learning rate to use when performing loss calculations
        """

        # call base class ctor
        super(Logistic, self).__init__(normalize, solver, iterations, alpha, lambda_r)

    def _cost_calc(self, X, y, theta=None, lambda_r=None):
        """

        :param X:
        :param y:
        :param theta:
        :return:
        """

        lambda_r = lambda_r or self.lambda_r
        # The objective of linear regression is to minimize the cost function
        # the function J(theta) = 1/m * sum(-y .* log(h_thetaX) - (1 - y) .* log(1-h_thetaX))
        # where the h_thetaX is the linear model h_theta = theta0 + theta1 * X1

        # reset class theta if theta is not None
        if theta is not None:
            self.theta_ = theta

        grad = np.zeros((theta.shape[0], 1))

        # suppress RuntimeWarning: overflow encountered due to NaN
        np.seterr(divide='ignore')

        # get number of training samples
        n_samples = y.shape[0]
        # fit intercept for linear equation this is the hypothesis
        hX = self.docalc_slope(X, theta)
        # calculate the minimized objective cost function for logistic regression
        j_cost = (1/n_samples) * (np.sum(np.multiply(-y, np.log(hX)) - np.multiply((1-y), np.log(1-hX)))) + lambda_r / (2 * n_samples) * sum(np.power(theta[2:], 2))

        # colum vector blah = np.array(X[:,[0]])
        for i in range(0, n_samples):
            grad = grad + (hX[i] - y[i]) * np.array(X[i:i+1, ]).T

        # grad_reg = lambda_r / n_samples * theta[2:] TODO: << this may need to be changed back for now get entire array
        grad_reg = lambda_r / n_samples * theta[:]

        grad = (1/n_samples) * grad + grad_reg

        # if regularization parameter lambda is set
        #if lambda_r:  #FIXME: MAJOR BUG IN REGULARIZATION CODE
        #    # factor regularization into cost value
        #    j_cost_reg = np.sum((lambda_r / (2 * n_samples) * sum(np.power(theta[2:], 2))))
        #    j_cost += j_cost_reg
        #    # factor regularization into gradient
        #    grad_reg = np.sum(lambda_r / n_samples * theta[2:])
        #    grad += grad_reg

        return j_cost, grad

    def docalc_slope(self, X, theta):
        """

        :param X:
        :param theta:
        :return:
        """

        return sigmoid.sigmoid(np.dot(X, theta))

    def fit(self, X, y, theta=None):
        """

        :param X:
        :param y:
        :param theta:
        :return:
        """

        if theta is not None and np.atleast_1d(theta).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        # pre fit data with theta params and bias if included
        X, y = self._pre_fit(X, y, theta)

        # FIXME: A linear gradient descent model does not do well in predicting values
        # check solver type
        self.theta1_, self.grad_ = fminfunc(self._cost_calc, X, y,
                                              self.theta_, alpha=self.alpha,
                                              max_iter=self.iterations)
        self.theta2_ = ls.gradient_descent(X, y, self.theta_, alpha=self.alpha, max_iter=self.iterations, linearclass=self)
        print("done")

    def predict(self, X):
        """

        :param X:
        :return:
        """

        if not hasattr(self, 'theta_'):
            raise RuntimeError("Instance is currently not fitted")

        X = np.array(X)
        X.reshape((X.shape[0], 1))

        if self.include_bias:
            # if 0-dim array we need to add bias if necessary
            if len(X.shape) != 2:  # insert col ones on axis 0
                X = np.insert(X, 0, 1, axis=0)

        blah1 = np.sum(sigmoid.sigmoid(np.dot(X, self.theta1_[:X.shape[0]])).astype(np.float32))
        blah2 = np.sum(sigmoid.sigmoid(np.dot(X, self.theta2_[:X.shape[0]])).astype(np.float32))

        # hypothesis in linear model: h_theta(x) = theta_zero + theta_one * x_one
        return np.sum(sigmoid.sigmoid(np.dot(X, self.theta_[:X.shape[0]])).astype(np.float32))
