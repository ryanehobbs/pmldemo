import numpy as np
import numbers
import preprocessing
import six
from abc import ABCMeta, abstractmethod

ALPHA_MIN = 0.0001
ALPHA_MAX = 10

class Linear(six.with_metaclass(ABCMeta)):
    """
    Base class for linear regression models.Implemented as a MixIn
    """
    __metaclass__ = ABCMeta

    def __init__(self, normalize=False, solver="linear", iterations=10, alpha=0.01, lambda_r=None, multi_class=False):
        """
        Base class for linear model regression calculations
        :param normalize: (Default: False) Scale features in training data if they differ in order of magnitude
        :param include_bias: (Default: False) include bias column in linear model
        :param solver: Type of solver to use for linear regression calculation
        :param iterations: Number of iterations to perform on training set
        :param alpha: Learning rate to use when performing loss calculations
        """

        # normalize only if using gradient descent always default to False
        self.normalize = normalize if solver != 'normal' else False
        self.include_bias = True  # TODO: refactor the args
        self.solver = solver
        self.iterations = iterations
        self.lambda_r = lambda_r
        self.multi_class = multi_class

        # ensure alpha (learning rate) conforms to 0.001 < alpha < 10
        if ALPHA_MIN < alpha < ALPHA_MAX:
            self.alpha = alpha
        else:
            print("Learning rate (alpha) does not fit within range 0.001 < alpha < 10 defaulting to 0.01")
            self.alpha = 0.01

    @abstractmethod
    def fit(self, X, y):
        """Abstract fitting method must be implemented in subclass"""
        pass

    @abstractmethod
    def docalc_slope(self, X, theta):
        """Abstract slope calculation method must be implemented in subclass"""
        pass

    def calculate_slope(self, X, theta):
        """
        Perform slope calculation by multiplying the nx1 vector (theta params) with the matrix X.
        Used primarily for peforming both cost calculations and gradient descent.  Method employs
        a GoF Template pattern which allows the linear subclass types to override.
        https://en.m.wikipedia.org/wiki/Template_method_pattern
        :param X: array-like Array[n_samples, n_features] Training data
        :param theta: array-like Vector[n_features]  coefficient parameters
        :return: Linear equation slope calculation
        """

        return self.docalc_slope(X, theta)

    def _pre_fit(self, X, y, theta=None):
        """
        Center data in linear model to zero along axis 0. If theta
        (sampled weights) are 0 then the weighted means of X and y is
        zero.
        :param X: array-like Array[n_samples, n_features] Training data
        :param y: np.ndarray Vector[n_samples] Training labels
        :param theta: array-like Vector[n_features]  coefficient parameters
        :return: Tuple (X, y) containing fitted intercept
        """

        # extract the array shape of row samples and column features
        n_samples, n_features = X.shape

        # check theta params and ensure correct data type
        if isinstance(theta, numbers.Number):
            theta = None
        elif not isinstance(theta, np.ndarray) and theta is not None:
            theta = np.array(theta, dtype='f')

        # initialize parameters (theta) using supplied or set to zero
        if theta is None:  # initialize weights to zero
            if self.include_bias:
                self.theta_ = np.zeros((n_features + 1, 1))
            else:
                self.theta_ = np.zeros((n_features, 1))
        else:
            self.theta_ = theta  # use values passed in

        if self.normalize:
            self.mu_ = np.mean(X, axis=0)
            self.sigma_ = np.std(X, axis=0)
            tf_mu = X - np.kron(np.ones((X.shape[0], 1)), self.mu_)
            tf_std = np.kron(np.ones((X.shape[0], 1)), self.sigma_)
            # divide feature values by their respective standard deviations
            X = np.divide(tf_mu, tf_std)

        # if including bias (intercept) set first column to ones
        if self.include_bias:
            self.bias_ = np.ones((n_samples, 1))
            X = np.insert(X, 0, self.bias_.T, axis=1)

        # FIXME: Maybe use sparse instead of numpy.matrix?
        X = np.matrix(X)
        return X, y

    @abstractmethod
    def predict(self, X):
        """"""
        pass

    def __predictN(self, X):
        """
        Method that will perform Normalized prediction.  This method assumes you are passing in
        an array that matches the cound of feature parameters to test against.
        :param X:
        :return:
        """

        if len(X.shape) != 2:
            X_sub = X-self.mu_  # 0-dim array (n,)
        else:
            X_sub = X[:,1:]-self.mu_   # n x 1 dim array (n,1)
        # divide mean by std_dev
        X = np.divide(X_sub, self.sigma_)

        if self.include_bias: # if bias in dataset
            if len(X.shape) != 2:  # insert col ones on axis 0
                X = np.insert(X, 0, 1, axis=0)
            else:  # insert col ones on axis 1
                X = np.insert(X, 0, 1, axis=1)

        return X

class LinearMixin(Linear):
    """Abstract mixin class for use by Linear Regression models"""
    __metaclass__ = ABCMeta




