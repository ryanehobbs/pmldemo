import numpy as np
from models import LinearMixin
import solvers.linear as ls
from solvers import fminfunc
import mathutils.sigmoid as sigmoid
from functools import wraps

# suppress RuntimeWarning: overflow encountered due to NaN
np.seterr(divide='ignore', invalid='ignore')

ALPHA_MIN = 0.0001
ALPHA_MAX = 10

def fitdata(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        xargs = list(args)
        # class reference is arg0
        klass = args[0]
        theta = kwargs.get('theta', None)
        refit = kwargs.get('refit', False)

        if len(args) == 3:
            X = args[1]
            y = args[2]
        elif len(args) == 4:
            X = args[1]
            y = args[2]
            theta = args[3]
        elif len(args) < 3:
            raise Exception("Invalid arguments for pre-fitting data")

        if not klass.fitted or refit:  # call base linear class _pre_fit method
            xargs[1], xargs[2] = klass._pre_fit(X, y, theta)
            klass.X_data = xargs[1]
            klass.y_data = xargs[2]

        return func(*xargs, **kwargs)

    return wrapper

class Linear(LinearMixin):

    __metaclass__ = LinearMixin
    def __init__(self, normalize=False, solver=None, **kwargs):
        """
        Create a linear model class for performing regression analysis
        :param normalize: (Default: False) Scale features in training data if they differ in order of magnitude
        :param include_bias: (Default: False) include bias column in linear model
        :param solver: Type of solver to use for linear regression calculation
        :param iterations: Number of iterations to perform on training set
        :param alpha: Learning rate to use when performing loss calculations
        """

        # call base class ctor
        super(Linear, self).__init__(normalize, solver, **kwargs)

    def _hypothesize(self, X, theta):
        """
        Calculate the slope for the linear equation. This is also
        used as the method which will calculate the hypothesis for
        a linear model
        :param X:
        :param theta:
        :return:
        """

        hX = np.dot(X, theta)
        hX = np.reshape(hX, (hX.shape[0], -1))  # make it a nx1 dim vector
        return hX

    @fitdata
    def cost(self, X, y, theta=None):
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

        if not isinstance(theta, np.ndarray) and theta is not None:
            theta = np.array(theta, dtype='f')[:, None]

        # get number of training samples
        n_samples = y.shape[0]
        # fit intercept for linear equation this is the hypothesis
        hX = self._hypothesize(X, theta)
        # squared error vectorized
        sumsqrd_err = np.sum(np.power(hX - y, 2))
        # # calculate the minimized objective cost function for linear regression
        j_cost = 1/(2 * n_samples) * sumsqrd_err

        # return minimal cost calculation
        return j_cost

    @fitdata
    def train(self, X, y, theta=None, **kwargs):
        """
        Fit training data against a linear regression model
        :param X: array-like Array[n_samples, n_features] Training data
        :param y: np.ndarray Vector[n_samples] Training labels
        :param theta: array-like Vector[n_features] coefficient parameters
        :return: Integer minimized cost
        """

        if theta is not None and np.atleast_1d(theta).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        # Set learning rate for use by gradient descent
        alpha = kwargs.get("alpha", 0.001)
        iterations = kwargs.get("iterations", self.iterations)

        # ensure alpha (learning rate) conforms to 0.001 < alpha < 10
        if ALPHA_MIN < alpha < ALPHA_MAX:
            self.alpha = alpha
        else:
            print("Learning rate (alpha) does not fit within range 0.001 < alpha < 10 defaulting to 0.01")
            self.alpha = 0.01

        # check solver type
        if self.solver == 'linear':
            self.theta_, self.grad_ = ls.gradient_descent(X, y, self.theta_, linearclass=self, alpha=alpha, max_iter=iterations)
        elif self.solver == 'normal':
            self.theta_ = ls.linear_leastsquares(X, y)

    def predict(self, X):
        """
        Predict outcomes using a linear model against training data

        :param X: array-like Array[n_samples, n_features] Training data
        :return: Array [n_samples] predicted values
        """

        # FIXME: predict can be used by itself if self.theta is populated or allow a theta value to
        # be submitted

        if not hasattr(self, 'theta_'):
            raise RuntimeError("Instance is currently not fitted")

        X = np.array(X)

        if self.normalize:  # predict based on normalized values
            X = self.predictN(X)
        else:
            if self.include_bias:
                # if 0-dim array we need to add bias if necessary
                if len(X.shape) != 2:  # insert col ones on axis 0
                    X = np.insert(X, 0, 1, axis=0)

        # hypothesis in linear model: h_theta(x) = theta_zero + theta_one * x_one
        return np.sum(np.dot(X, self.theta_).astype(np.float32))

class Logistic(LinearMixin):

    __metaclass__ = LinearMixin

    def __init__(self, normalize=False, solver='logistic', num_of_labels=0, **kwargs):
        """
        Create a linear model class for performing regression analysis
        :param normalize: (Default: False) Scale features in training data if they differ in order of magnitude
        :param include_bias: (Default: False) include bias column in linear model
        :param solver: Type of solver to use for linear regression calculation
        :param iterations: Number of iterations to perform on training set
        :param alpha: Learning rate to use when performing loss calculations
        """

        self.multiclass = False

        if num_of_labels > 0 and isinstance(num_of_labels, int):
            self.num_of_labels = int(num_of_labels)
            self.multiclass = True

        # call base class ctor
        super(Logistic, self).__init__(normalize, solver, **kwargs)

    @fitdata
    def cost(self, X, y, theta=None, lambda_r=0, **kwargs):
        """

        :param X:
        :param y:
        :param theta:
        :return:
        """

        # The objective of linear regression is to minimize the cost function
        # the function J(theta) = 1/m * sum(-y .* log(h_thetaX) - (1 - y) .* log(1-h_thetaX))
        # where the h_thetaX is the linear model h_theta = theta0 + theta1 * X1

        if not isinstance(theta, np.ndarray) and theta is not None:
            theta = np.array(theta, dtype='f')[:, None]

        grad = np.zeros((theta.shape[0], 1))
        # get number of training samples
        n_samples = y.shape[0]
        # fit intercept for linear equation this is the hypothesis
        hX = self._hypothesize(X, theta)
        # calculate the minimized objective cost function for logistic regression
        j_cost = (1/n_samples) * np.sum(np.multiply(-y, np.log(hX)) - np.multiply((1-y), np.log(1-hX))) + \
                 (lambda_r / (2 * n_samples) * np.sum(np.power(theta[1:], 2)))

        mask = np.ones((np.size(theta), 1))
        mask[0] = 0
        grad = np.divide(1, n_samples) * np.dot(X.T, (hX - y)) + lambda_r * (theta * mask) / n_samples

        return j_cost, grad

    def _hypothesize(self, X, theta):
        """

        :param X:
        :param theta:
        :return:
        """

        hX = sigmoid.sigmoid(np.dot(X, theta))
        hX = np.reshape(hX, (hX.shape[0], -1))  # make it a nx1 dim vector
        return hX

    @fitdata
    def train(self, X, y, theta=None, lambda_r=0, **kwargs):
        """

        :param X:
        :param y:
        :param theta:
        :return:
        """

        if theta is not None and np.atleast_1d(theta).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        # Set learning rate for use by gradient descent
        alpha = kwargs.get("alpha", 0.001)
        iterations = kwargs.get("iterations", self.iterations)

        # ensure alpha (learning rate) conforms to 0.001 < alpha < 10
        if ALPHA_MIN < alpha < ALPHA_MAX:
            self.alpha = alpha
        else:
            print("Learning rate (alpha) does not fit within range 0.001 < alpha < 10 defaulting to 0.01")
            self.alpha = 0.01

        if self.multiclass:
            self.theta_ = self.one_vs_all(X, y, self.theta_, self.num_of_labels)
        else:
            # check solver type
            self.theta, self.grad_ = fminfunc(self.cost, X, y,
                                                  self.theta_, alpha=alpha,
                                                  max_iter=iterations,
                                                  lambda_r=lambda_r)
            #self.theta2_, self.grad_ = ls.gradient_descent(X, y, self.theta_, linearclass=self, alpha=alpha, max_iter=iterations)

    def predict(self, X, sum=False):
        """

        :param X:
        :return:
        """

        # FIXME: Need to handle how we insert the bias intercept better
        # FIXME: if X_data and y_data present they already have a column of 1's
        # FIXME: we need a way to check for the bias column and if it does not exist
        # FIXME: insert

        if not hasattr(self, 'theta_'):
            raise RuntimeError("Instance is currently not fitted")

        X = np.array(X)
        X.reshape((X.shape[0], -1))

        if self.include_bias:
            # if 0-dim array we need to add bias if necessary
            if len(X.shape) != 2:  # insert col ones on axis 0
                X = np.insert(X, 0, 1, axis=0)
            elif X.shape[1] < self.theta_.shape[1]:
                cols =  self.theta_.shape[1] - X.shape[1]
                X = np.insert(X, 0, cols, axis=1)

        if self.multiclass:
            return self.predictOVA(X)
        else:
            if sum:
                # hypothesis in logistic model: h_theta(x) = theta_zero + theta_one * x_one
                return np.sum(self._hypothesize(X, self.theta_[:X.shape[0]])).astype(np.float32)
                #return np.sum(sigmoid.sigmoid(np.dot(X, self.theta_[:X.shape[0]])).astype(np.float32))
            else:
                return (self._hypothesize(X, self.theta_) > sigmoid.sigmoid(0)).astype(np.int)
                #return (sigmoid.sigmoid(np.dot(X, self.theta_)) > sigmoid.sigmoid(0)).astype(np.int)