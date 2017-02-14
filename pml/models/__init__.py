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

    def __init__(self, normalize=False, include_bias=False, solver="linear", iterations=10, alpha=0.01):
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

        # initialize weights (theta) using supplied or set to zero
        if theta is None:  # initialize weights to zero
            self.theta_ = np.zeros((n_features + 1, 1))
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

        return X, y

    def predict(self, X):
        """
        Predict outcomes using a linear model against training data

        :param X: array-like Array[n_samples, n_features] Training data
        :return: Array [n_samples] predicted values
        """

        # FIXME: THIS DOES NOT WORK WITH JUST PASSING IN TWO VALUES FOR PREDICTIONS NEEDS FIXING

        if not hasattr(self, 'theta_'):
            raise RuntimeError("Instance is currently not fitted")

        if type(X) not in (list, np.ndarray, np.matrix):
            X = [X]

        if isinstance(X, list):
            X = np.array(X) # FIXME: maybe convert the other way around ndarry -> list ??

        if self.normalize:
            return self.__predictN(X)
        else:
            # hypothesis in linear model: h_theta(x) = theta_zero + theta_one * x_one
            return np.dot(X, self.theta_)

    def __predictN(self, X):
        """
        Normalized prediction
        :param X:
        :return:
        """

        # FIXME: THIS DOES NOT WORK WITH JUST PASSING IN TWO VALUES FOR PREDICTIONS NEEDS FIXING
        X = np.array(X)

        # get length of mu/sigma and ensure X length equals if not the bias may have been passed in to X
        if X.shape[0] != self.mu_.shape[0]:
            raise Exception("To many prediction values pass in")

        return np.dot(np.divide((X-self.mu_), self.sigma_), self.theta_)

        #blah2 = np.divide((X[1:]-self.mu_), self.sigma_)
        #blah2 = np.insert(blah2, 0, 1, axis=0)  # <-- janky and hacky
        #blah3 = np.dot(blah2, self.theta_)
        print("done")





