import numpy as np
import numbers
import six
from abc import ABCMeta, abstractmethod

ALPHA_MIN = 0.001
ALPHA_MAX = 10

class LinearMixin(six.with_metaclass(ABCMeta)):
    """
    Base class for linear regression models.Implemented as a MixIn
    """

    def __init__(self, normalize=False, include_bias=True, solver="linear", iterations=10, alpha=0.01):
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
        self.include_bias = include_bias
        self.solver = solver
        self.iterations = iterations

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

    def fit_intercept(self, X, y, theta=None):
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

        # if including bias set first column to ones
        if self.include_bias:
            self.bias_ = np.ones((n_samples, 1))
            X = np.insert(X, 0, self.bias_.T, axis=1)
            #self.mu_ = np.insert(self.mu_, 0, 1, axis=0)
            #self.sigma_ = np.insert(self.sigma_, 0, 1, axis=0)

        return X, y

    def predict(self, X):
        """
        Predict outcomes using a linear model against training data

        :param X: array-like Array[n_samples, n_features] Training data
        :return: Array [n_samples] predicted values
        """

        theta_length = self.theta_.shape[0]

        if not hasattr(self, 'theta_'):
            raise RuntimeError("Instance is currently not fitted")

        X = np.array(X)

        if self.normalize:  # predict based on mu and sigma values
            X = self.__predictN(X)
        else:
            if self.include_bias:
                if len(X.shape) != 2:
                    # insert col ones again
                    X = np.insert(X, 0, 1, axis=0)

        # hypothesis in linear model: h_theta(x) = theta_zero + theta_one * x_one
        return np.dot(X, self.theta_)

    def __predictN(self, X):
        """
        Normalized prediction
        :param X:
        :return:
        """

        #if X.shape[0] == 47:
        #    return np.divide((X[0:,]-self.mu_), self.sigma_)
        #else:  #<-- This works when costcalc is not used AND mu/sigma do not have ones added but this will break when using
        #       # cost calc in formula
        #    blah = np.divide((X[1:]-self.mu_), self.sigma_)
        #    blah2 = np.insert(blah, 0, 1, axis=0)  # <-- janky and hacky
        #    return blah2

        if len(X.shape) != 2:  # 0-dim array
            X_sub = X-self.mu_
        else:  # n x 1 dim array
            X_sub = X[:,1:]-self.mu_
        X_div = np.divide(X_sub, self.sigma_)

        if self.include_bias:
            if len(X.shape) != 2:
                # insert col ones again
                X = np.insert(X_div, 0, 1, axis=0)
            else:
                X = np.insert(X_div, 0, 1, axis=1)

        return X