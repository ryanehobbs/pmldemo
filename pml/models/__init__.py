"""Base model class"""

import numbers
import numpy as np
import six
from abc import ABCMeta, abstractmethod
from solvers import fmincg


class LinearBase(six.with_metaclass(ABCMeta)):
    """
    Base class for linear regression models.Implemented as a MixIn
    """
    __metaclass__ = ABCMeta

    fitted_ = False  # True target and labels fitted with intercept
    X_ = []  # store target data in class instance
    y_ = []  # store y label data in class instance

    def __init__(self, normalize=False, solver=None, **kwargs):
        """
        Base class for linear model regression calculations
        :param normalize: (Default: False) Scale features in training data if they differ in order of magnitude
        :param solver: Type of solver to use for linear regression calculation
        :param include_bias: (Default: False) include bias column in linear model
        :param iterations: Number of iterations to perform on training set
        """

        # solver must be defined else raise exception
        if not solver:
            raise Exception(
                "Solver must be defined, solver parameter set to {}".format(solver))

        # normalize only if using gradient descent always default to False
        self.normalize = normalize if solver != 'normal' else False
        # set the solver type logistic or linear
        self.solver = solver
        # True or False include bias column (1's) y-intercept
        self.include_bias = kwargs.get("include_bias", True)
        # Set max iterations to execute min cost routine
        self.iterations = kwargs.get("max_iter", 10)

    def __str__(self):
        """
        Return string name representation of model (Linear, Logistic, etc)
        :return: String containing model name ("Linear", "Logistic", etc)
        """

        return self.solver

    @property
    def fitted(self):
        """
        Property returns True if target data has been fitted properly with a column of 1's which
        is the intercept term. Fitting also normalized the target data.
        :return: True intercept term fitted into target data. False intercept term not fitted
        """

        return self.fitted_

    @property
    def X_data(self):
        """
        Property returns the target data matrix or vector
        :return: array-like Array[n_samples, n_features] Training data
        """

        return self.X_

    @X_data.setter
    def X_data(self, value):
        """
        Property sets the class target data. Useful if you want to store the fitted target data in the class
        :param value: array-like Array[n_samples, n_features] Training data
        :return: Nothing
        """

        self.X_ = value

    @property
    def y_data(self):
        """
        Property return the y label data vector
        :return: np.ndarray Vector[n_samples] Training labels
        """

        return self.y_

    @y_data.setter
    def y_data(self, value):
        """
        Property sets the class y label data. Useful if you want to store the fitted y label data in the class
        :param value: np.ndarray Vector[n_samples] Training labels
        :return: Nothing
        """

        self.y_ = value

    @abstractmethod
    def predict(self, X):
        """Abstract predict method must be implemented in subclass"""
        pass

    @abstractmethod
    def cost(self, X, y, theta=None, lambda_r=0):
        """Abstract cost calculation method must be implemented in subclass"""
        pass

    @abstractmethod
    def train(self, X, y):
        """Abstract fitting method must be implemented in subclass"""
        pass

    @abstractmethod
    def _hypothesize(self, X, theta):
        """Abstract slope calculation method must be implemented in subclass"""
        pass

    def calculate_hypothesis(self, X, theta):
        """
        Perform slope calculation by multiplying the nx1 vector (theta params) with the matrix X.
        Used primarily for performing both cost calculations and gradient descent.
        :param X: array-like Array[n_samples, n_features] Training data
        :param theta: array-like Vector[n_features]  coefficient parameters
        :return: Linear equation slope calculation
        """

        # Method employs a GoF Template pattern which allows the
        # linear subclass types to override.
        # https://en.m.wikipedia.org/wiki/Template_method_pattern

        # refer to subclass to see actual implementation
        return self._hypothesize(X, theta)

    def calculate_cost(self, X, y, theta):
        """
        Perform cost minimization between training data and labels with weights.  This
        is an objective function that will return the minimized cost based upon
        a weighted prediction and the actual class label.
        :param X: array-like Array[n_samples, n_features] Training data
        :param y: np.ndarray Vector[n_samples] Training labels
        :param theta: array-like Vector[n_features]  coefficient parameters
        :return: Tuple (int, array) cost value and array containing cost history
        """

        # Method employs a GoF Template pattern which allows the
        # linear subclass types to override.
        # https://en.m.wikipedia.org/wiki/Template_method_pattern

        # refer to subclass to see actual implementation
        return self.cost(X, y, theta)

    def pre_fit(self, X, y, theta=None):
        """
        Center data in linear model to zero along axis 0. If theta
        (sampled weights) are 0 then the weighted means of X and y is
        zero. If normalization is requested, the data will be normalized.
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
        if theta is None:
            if self.include_bias:
                # initialize weights to zero include bias
                self.theta_ = np.zeros((n_features + 1, 1))
            else:
                # initialize weights to zero no bias
                self.theta_ = np.zeros((n_features, 1))
        else:
            self.theta_ = theta  # use values passed in

        if self.normalize:  # scale data
            self.mu_ = np.mean(X, axis=0)  # calc mean of training data
            self.sigma_ = np.std(X, axis=0)  # calc stddev of training data
            # compute Kronecker product of X and Mu
            tf_mu = X - np.kron(np.ones((X.shape[0], 1)), self.mu_)
            # compute Kronecker product of X and sigma
            tf_std = np.kron(np.ones((X.shape[0], 1)), self.sigma_)
            # divide feature values by their respective standard deviations
            X = np.divide(tf_mu, tf_std)

        # if including bias (intercept) set first column to ones
        if self.include_bias:
            self.bias_ = np.ones((n_samples, 1))
            X = np.insert(X, 0, self.bias_.T, axis=1)

        self.fitted_ = True  # we are now fitted
        self.X_ = X  # set class training data var
        self.y_ = y  # set class training label data var

        return X, y


# noinspection PyTypeChecker
class LinearMixin(LinearBase):
    """Abstract mixin class for use by Linear Regression models"""
    __metaclass__ = ABCMeta

    def __str__(self):
        """
        Return string name representation of model (Linear, Logistic, etc)
        :return: String containing model name ("Linear", "Logistic", etc)
        """

        return self.solver

    def __repr__(self):
        """
        Return string name representation of model (Linear, Logistic, etc)
        :return: String containing model name ("Linear", "Logistic", etc)
        """

        return self.solver

    def predictN(self, X):
        """
        Method that will perform Normalized prediction.  This method assumes you are passing in
        an array that matches the count of feature parameters to test against.
        :param X: array-like Array[n_samples, n_features] Training data
        :return:
        """

        if len(X.shape) != 2:
            X_sub = X - self.mu_  # 0-dim array (n,)
        else:
            X_sub = X[:, 1:] - self.mu_   # n x 1 dim array (n,1)

        # divide mean by std_dev
        X = np.divide(X_sub, self.sigma_)

        if self.include_bias:  # if bias in dataset
            if len(X.shape) != 2:  # insert col ones on axis 0
                X = np.insert(X, 0, 1, axis=0)
            else:  # insert col ones on axis 1
                # noinspection PyTypeChecker
                X = np.insert(X, 0, 1, axis=1)

        return X

    def predictOVA(self, X):
        """
        Predict one vs. all. Return a vector of predictions for each example in the matrix X
        :param X: array-like Array[n_samples, n_features] Training data
        :return: np.ndarray Vector[predictions] prediction values
        """

        hX = self._hypothesize(X, self.theta_.T)
        # return the indice(index) of array that contains the larget value
        indice_array = np.argmax(hX, axis=1)
        # make this a n x 1 dimensional array
        indice_array = indice_array[:, None]

        return indice_array

    def one_vs_all(self, X, y, initial_theta, num_of_labels, **kwargs):
        """
        Support multi-class training. Trains multiple logistic regression classifiers.
        Uses one vs. all (OvA) or called one vs. rest OvR. https://en.wikipedia.org/wiki/Multiclass_classification
        Trains multiple logistic regression classifiers and returns all the classifiers
        in a matrix ova_theta, where the i-th row of ova_theta corresponds to the classifier for label i
        :param X: array-like Array[n_samples, n_features] Training data
        :param y: np.ndarray Vector[n_samples] Training labels
        :param initial_theta: rray-like Vector[n_features] coefficient parameters
        :param num_of_labels: Number of labels to train
        :return: array-like Array[classifiers] trained data
        """

        if initial_theta is not None and np.atleast_1d(initial_theta).ndim < 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        # Set learning rate for use by gradient descent
        alpha = kwargs.get("alpha", 0.001)
        # set how many iterations to cycle if defined else use class set value
        iterations = kwargs.get("iterations", self.iterations)

        # get size of training data
        n = np.size(X, axis=1)
        # create ova_theta ndarray m x n based where
        ova_theta = np.zeros((num_of_labels, n))
        # iterate over labels to train on
        for i in range(0, num_of_labels):
            theta, _, _ = fmincg(self.cost, X, (y == i),
                                 initial_theta=initial_theta,
                                 alpha=alpha,
                                 max_iter=iterations)
            # set ova_theta i'th element to trained classifier labels
            ova_theta[i, :] = theta.T

        return ova_theta
