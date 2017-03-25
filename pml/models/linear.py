"""Linear models, this is a subclass"""
import mathutils.sigmoid as sigmoid
import numpy as np
import solvers.linear as ls
from functools import wraps
from models import LinearMixin
from solvers import fminfunc

# suppress RuntimeWarning: overflow encountered due to NaN
np.seterr(divide='ignore', invalid='ignore')

ALPHA_MIN = 0.0001  # learning rate minimum
ALPHA_MAX = 10  # learning rate maximum


def fitdata(func):
    """
    Decorator for ensuring that data is fit properly before processing
    :param func: Wrapped function
    :return: Fitted data
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        X = y = None
        xargs = list(args)
        # class reference is arg0
        klass = args[0]
        # get the theta and refit values from kwargs
        theta = kwargs.get('theta', None)
        refit = kwargs.get('refit', False)

        if len(args) == 3:  # only training and y label data present in args
            X = args[1]
            y = args[2]
        elif len(args) == 4:  # training, y label and theta params present
            X = args[1]
            y = args[2]
            theta = args[3]
        elif len(args) < 3:  # we need at least X and y to fit data
            raise Exception("Invalid arguments for pre-fitting data")

        # check if class is fitted if not call prefit else pass through
        if not klass.fitted or refit:  # call base linear class _pre_fit method
            xargs[1], xargs[2] = klass.pre_fit(X, y, theta)

        return func(*xargs, **kwargs)

    return wrapper


# noinspection PyTypeChecker
class Linear(LinearMixin):
    """Linear regression class model"""

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
        :param X: array-like Array[n_samples, n_features] Training data
        :param theta: array-like Vector[n_features] coefficient parameters
        :return: Linear equation slope calculation
        """

        hX = np.dot(X, theta)
        hX = np.reshape(hX, (hX.shape[0], -1))  # make it a nx1 dim vector
        return hX

    @fitdata
    def cost(self, X, y, theta=None, lambda_r=0):
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
        j_cost = 1 / (2 * n_samples) * sumsqrd_err

        # return minimal cost calculation
        return j_cost

    @fitdata
    def train(self, X, y, theta=None, **kwargs):
        """
        Fit training data against a linear regression model. Train data to make
        predictions.
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
            alpha = alpha
        else:
            print(
                "Learning rate (alpha) does not fit within range 0.001 < alpha < 10 defaulting to 0.01")
            alpha = 0.01

        # check solver type
        if self.solver == 'linear':
            self.theta_, self.grad_ = ls.gradient_descent(
                X, y, self.theta_, linearclass=self, alpha=alpha, max_iter=iterations)
        elif self.solver == 'normal':
            self.theta_ = ls.linear_leastsquares(X, y)

    def predict(self, X, sum=False):
        """
        Predict outcomes using a linear model against training data
        :param X: array-like Array[n_samples, n_features] Training data
        :param sum: True return a sum value and not a vector array. False return vector array
        :return: Array [n_samples] predicted values or integer sum value
        """

        if not self.fitted:
            raise RuntimeError("Instance is currently not fitted")

        if X is not None:
            # Always cast target param X to a numpy array
            X = np.array(X)
            X.reshape((X.shape[0], -1))
        elif self.X_data is not None and X is None:
            X = self.X_data
        else:
            raise Exception(
                "Unable to qualify training data X for predictions")

        if self.normalize:  # predict based on scaled normalized values
            X = self.predictN(X)
        else:
            if self.include_bias:
                # if 0-dim array we need to add bias if necessary
                if len(X.shape) != 2:  # insert col ones on axis 0
                    X = np.insert(X, 0, 1, axis=0)

        if sum:
            # hypothesis in linear model: h_theta(x) = theta_zero + theta_one *
            # x_one as sum
            return np.sum(self._hypothesize(X, self.theta_).astype(np.float32))
        else:
            # hypothesis in linear model: h_theta(x) = theta_zero + theta_one *
            # x_one as vector
            return (self._hypothesize(X, self.theta_).astype(np.float32))


# noinspection PyTypeChecker
class Logistic(LinearMixin):
    """Logistic regression class model"""

    __metaclass__ = LinearMixin

    def __init__(
            self,
            normalize=False,
            solver='logistic',
            num_of_labels=0,
            **kwargs):
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

        :param X: array-like Array[n_samples, n_features] Training data
        :param y: np.ndarray Vector[n_samples] Training labels
        :param theta: array-like Vector[n_features] coefficient parameters
        :param lambda_r: integer Regularization parameter to reduce over-fitting
        :return: Tuple (int, array) cost value and array containing cost history
        """

        # The objective of linear regression is to minimize the cost function
        # the function J(theta) = 1/m * sum(-y .* log(h_thetaX) - (1 - y) .* log(1-h_thetaX))
        # where the h_thetaX is the linear model h_theta = theta0 + theta1 * X1

        if not isinstance(theta, np.ndarray) and theta is not None:
            theta = np.array(theta, dtype='f')[:, None]

        # get number of training samples
        n_samples = y.shape[0]
        # fit intercept for linear equation this is the hypothesis
        hX = self._hypothesize(X, theta)
        # calculate the minimized objective cost function for logistic
        # regression
        first_term = (1 / n_samples) * np.sum(np.multiply(-y, np.log(hX)) - np.multiply((1 - y), np.log(1 - hX)))
        second_term = (lambda_r / (2 * n_samples) *np.sum(np.power(theta[1:], 2)))
        j_cost = first_term + second_term

        # create a vector mask of zeros
        mask = np.ones((np.size(theta), 1))
        mask[0] = 0
        # calculate cost gradient
        grad = np.divide(1, n_samples) * np.dot(X.T, (hX - y)) + \
            lambda_r * (theta * mask) / n_samples

        return j_cost, grad

    def _hypothesize(self, X, theta):
        """
        Calculate the slope for the linear equation. This is also
        used as the method which will calculate the hypothesis for
        a linear model
        :param X: array-like Array[n_samples, n_features] Training data
        :param theta: array-like Vector[n_features] coefficient parameters
        :return: Linear equation slope calculation
        """

        hX = sigmoid.sigmoid(np.dot(X, theta))
        hX = np.reshape(hX, (hX.shape[0], -1))  # make it a nx1 dim vector
        return hX

    @fitdata
    def train(self, X, y, theta=None, lambda_r=0, **kwargs):
        """
        Fit training data against a linear regression model. Train data to make
        predictions.
        :param X: array-like Array[n_samples, n_features] Training data
        :param y: np.ndarray Vector[n_samples] Training labels
        :param theta: array-like Vector[n_features] coefficient parameters
        :param lambda_r: integer Regularization parameter to reduce over-fitting
        :param fmin: True use Newton-Gauss minimization function
        :return: Integer minimized cost
        """

        if theta is not None and np.atleast_1d(theta).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        # Set learning rate for use by gradient descent
        alpha = kwargs.get("alpha", 0.001)
        iterations = kwargs.get("iterations", self.iterations)
        fmin = kwargs.get("fmin", True)

        # ensure alpha (learning rate) conforms to 0.001 < alpha < 10
        if ALPHA_MIN < alpha < ALPHA_MAX:
            alpha = alpha
        else:
            print(
                "Learning rate (alpha) does not fit within range 0.001 < alpha < 10 defaulting to 0.01")
            alpha = 0.01

        if self.multiclass:
            self.theta_ = self.one_vs_all(
                X, y, self.theta_, self.num_of_labels)
        else:  # check solver type
            if fmin:  # default choice
                self.theta, self.grad_ = fminfunc(self.cost, X, y,
                                                  self.theta_, alpha=alpha,
                                                  max_iter=iterations,
                                                  lambda_r=lambda_r)
            else:  # this can be used but should not for logistic
                self.theta_, self.grad_ = ls.gradient_descent(
                    X, y, self.theta_, linearclass=self, alpha=alpha, max_iter=iterations)

    def predict(self, X=None, sum=False):
        """
        Predict outcomes using a linear model against training data
        :param X: array-like Array[n_samples, n_features] Training data
        :param sum: True return a sum value and not a vector array. False return vector array
        :return: Array [n_samples] predicted values or integer sum value
        """

        if not self.fitted:
            raise RuntimeError("Instance is currently not fitted")

        if X is not None:
            # Always cast target param X to a numpy array
            X = np.array(X)
            X.reshape((X.shape[0], -1))
        elif self.X_data is not None and X is None:
            X = self.X_data
        else:
            raise Exception(
                "Unable to qualify training data X for predictions")

        if self.include_bias:
            # if 0-dim array we need to add bias if necessary
            if len(X.shape) != 2:  # insert col ones on axis 0
                X = np.insert(X, 0, 1, axis=0)
            elif X.shape[1] < self.theta_.shape[1]:
                cols = self.theta_.shape[1] - X.shape[1]
                X = np.insert(X, 0, cols, axis=1)

        if self.multiclass:  # use ova prediction
            return self.predictOVA(X)
        else:
            if sum:
                # hypothesis in linear model: h_theta(x) = theta_zero +
                # theta_one * x_one as sum
                return np.sum(
                    self._hypothesize(
                        X, self.theta_[
                            :X.shape[0]])).astype(
                    np.float32)
            else:
                # hypothesis in linear model: h_theta(x) = theta_zero +
                # theta_one * x_one as vector
                return (
                    self._hypothesize(
                        X, self.theta_) > sigmoid.sigmoid(0)).astype(
                    np.int)
