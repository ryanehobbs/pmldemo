import numpy as np
import numbers
import six
from mathutils import sigmoid
from solvers import fmincg
from abc import ABCMeta, abstractmethod



class LinearBase(six.with_metaclass(ABCMeta)):
    """
    Base class for linear regression models.Implemented as a MixIn
    """
    __metaclass__ = ABCMeta

    fitted = False
    _X = []
    _y = []
    def __init__(self, normalize=False, solver=None, **kwargs):
        """
        Base class for linear model regression calculations
        :param normalize: (Default: False) Scale features in training data if they differ in order of magnitude
        :param include_bias: (Default: False) include bias column in linear model
        :param solver: Type of solver to use for linear regression calculation
        :param iterations: Number of iterations to perform on training set
        :param alpha: Learning rate to use when performing loss calculations
        """

        # solver must be defined else raise exception
        if not solver:
            raise Exception("Solver must be defined, solver parameter set to {}".format(solver))

        # normalize only if using gradient descent always default to False
        self.normalize = normalize if solver != 'normal' else False
        # set the solver type logistic or linear
        self.solver = solver
        # True or False include bias column (1's) y-intercept
        self.include_bias = kwargs.get("include_bias", True)
        # Set max iterations to execute min cost routine
        self.iterations = kwargs.get("max_iter", 10)

    def __str__(self):

        return self.solver

    @property
    def X_data(self):

        return self._X

    @X_data.setter
    def X_data(self, value):

        self._X = value

    @property
    def y_data(self):

        return self._y

    @y_data.setter
    def y_data(self, value):

        self._y = value

    @abstractmethod
    def predict(self, X):
        """"""
        pass

    @abstractmethod
    def cost(self, X, y, theta=None, lambda_r=0):
        """"""
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
        Used primarily for peforming both cost calculations and gradient descent.
        :param X: array-like Array[n_samples, n_features] Training data
        :param theta: array-like Vector[n_features]  coefficient parameters
        :return: Linear equation slope calculation
        """

        # Method employs a GoF Template pattern which allows the
        # linear subclass types to override.
        # https://en.m.wikipedia.org/wiki/Template_method_pattern

        return self._hypothesize(X, theta)

    def calculate_cost(self, X, y, theta):
        """
        Perform cost minimization between training data and labels with weights.  This
        is an objective function that will return the minimized cost based upon
        a weighted prediction and the actual class label.
        :param X:
        :param y:
        :param theta:
        :return:
        """

        # Method employs a GoF Template pattern which allows the
        # linear subclass types to override.
        # https://en.m.wikipedia.org/wiki/Template_method_pattern

        return self.cost(X, y, theta)

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

        self.fitted = True

        return X, y

class LinearMixin(LinearBase):
    """Abstract mixin class for use by Linear Regression models"""
    __metaclass__ = ABCMeta

    def __str__(self):

        return self.solver

    def __repr__(self):

        return self.solver

    def predictN(self, X):
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

    def predictOVA(self, X):
        """

        :param X:
        :return:
        """

        hX = self._hypothesize(X, self.theta_.T)
        # return the indice(index) of array that contains the larget value
        indice_array = np.argmax(hX, 1)
        # make this a n x 1 dimensional array
        indice_array = indice_array[:, None]
        # return an array of indices (rows)
        #r, c = np.indices((indice_array.size, 1))
        # we add 1 because y labels are 1 - 10, but python array indexes are 0 - 9
        #np.add.at(indice_array, r, 1)

        return indice_array

    def one_vs_all(self, X, y, initial_theta, num_of_labels, **kwargs):
        """
        Support multi-class training. Trains multiple logistic regression classifiers.
        Uses one vs. all (OvA) or called one vs. rest OvR.
        https://en.wikipedia.org/wiki/Multiclass_classification
        :param X:
        :param y:
        :param num_of_labels:
        :return:
        """

        if initial_theta is not None and np.atleast_1d(initial_theta).ndim < 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        # Set learning rate for use by gradient descent
        alpha = kwargs.get("alpha", 0.001)
        iterations = kwargs.get("iterations", self.iterations)

        n = np.size(X, axis=1)

        ova_theta = np.zeros((num_of_labels, n))

        for i in range(0, num_of_labels):
            y_idx = i + 1
            theta, _, _ = fmincg(self.cost, X, (y == i),
                                     initial_theta=initial_theta,
                                     alpha=alpha,
                                     max_iter=iterations)
            ova_theta[i, :] = theta.T

        return ova_theta







