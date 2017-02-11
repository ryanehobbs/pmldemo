import numpy as np
import numbers
import six
from abc import ABCMeta, abstractmethod

class LinearMixin(six.with_metaclass(ABCMeta)):
    """
    Base class for linear regression models.Implemented as a MixIn
    """

    def __init__(self, normalize=False, include_bias=False, fminfunc="linear", iterations=10, alpha=0.01):
        """

        :param normalize:
        :param include_bias:
        """

        self.normalize = normalize
        self.include_bias = include_bias
        self.fminfunc = fminfunc
        self.iterations = iterations
        self.alpha = alpha

    @abstractmethod
    def fit(self, X, y):
        """Abstract fitting method must be implemented in subclass"""
        pass

    def fit_intercept(self, X, y, theta=None):
        """
        Center data in linear model to zero along axis 0. If theta
        (sampled weights) are 0 then the weighted means of X and y is
        zero.
        :param X: array-like Array[n_samples, n_features] Training data
        :param y: np.ndarray Vector[n_samples] Training labels
        :param theta: array-like Vector[n_features] sample weights <-- these are actually coef weights are the ones
        :return: Tuple (X, y) containing fitted intercept
        """

        # extract the array shape of row samples and column features
        n_samples, n_features = X.shape

        # check theta params and ensure correct data type
        if isinstance(theta, numbers.Number):
            theta = None
        elif not isinstance(theta, np.ndarray) and theta is not None:
            theta = np.array(theta, dtype='f')

        # initialize weights (theta) using supplied or set to zero
        if theta is None:  # initialize weights to zero
            self.theta_ = np.zeros((n_features + 1, 1))
        else:
            self.theta_ = theta  # use values passed in

        # if including bias set first column to ones
        if self.include_bias:
            self.coef_ = np.ones((n_samples, 1))
            X = np.insert(X, 0, self.coef_.T, axis=1)

        return X, y

    def predict(self, X):
        """
        Predict outcomes using a linear model against training data

        :param X: array-like Array[n_samples, n_features] Training data
        :return: Array [n_samples] predicted values
        """

        if not hasattr(self, 'theta_'):
            raise RuntimeError("Instance is currently not fitted")

        # hypothesis in linear model: h_theta(x) = theta_zero + theta_one * x_one
        return np.dot(X, self.theta_)

