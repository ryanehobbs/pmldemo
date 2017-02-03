import numpy as np
from graphs.plotter import Plotter
import matplotlib.pyplot as plt

class Classifier(object):

    eta_ = 0.01
    n_iter_ = 0
    w_ = []
    cost_ = []
    fitted_ = False

    def __init__(self, eta=0.01, n_iter=10):
        """
        Initialize Perceptron class object
        :param eta: float Learning rate (between 0.0 and 1.0)
        :param n_iter: int Passes over the training dataset.
        """
        self.eta_ = eta
        self.n_iter_ = n_iter

    def fit(self, X, y, n_iter=None, eta=None):
        """

        :param X:
        :param y:
        :param n_iter:
        :param eta:
        :return:
        """

        if n_iter:
            self.n_iter_ = n_iter
        if eta:
            self.eta_ = eta

    def predict(self, X):
        """
        Predict class label
        :param X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        :return: Return class label after unit step
        """

        # threshold, if input >= 0.0 choose 1 else -1 label
        value = np.where(self.activation(X) >= 0.0, 1, -1)
        return value

    def activation(self, X):
        """
        Calculate input values
        :param X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        :return: Sum of input values
        """

        # perform matrix multiplication
        # W0X0 + (W1X1 + WmXm)
        value = np.dot(X, self.w_[1:]) + self.w_[0]
        return value

    def linear(self, X, y, theta):
        """
        Method will perform a linear regression calculation. This
        objective returns a minimized cost function given a linear
        hypothesis model. This model assumes only one variable in
        the linear regression model
        :param X:
        :param y:
        :param theta:
        :return:
        """

        # hypothesis h(x): h_theta(x) = theta^Tx = theta_zero + (theta_one * (x_one))
        # cost function: J(theta) = 1/2m * sum(h_theta(x^i) - y^i)^2

        # X is a m x n matrix
        # y is a n x 1 vector
        # theta is a n x 1 vector

        if not isinstance(theta, list) or theta is None:
            theta = np.zeros((2, 1))  # set initial theta values to a 1 x n vector
        else:
            theta = np.array(theta, dtype='f')
            theta = theta[0:, None]

        if X.shape[1] != 1:  # can only have 2 cols for single variable linear regression
            raise Exception("Dimension mismatch (mx2) != ({}). Training data matrix size can only"
                            "have a minimum of 1 columns.".format(X.shape))

        if theta.shape != (2, 1):
            raise Exception("Dimension mismatch (2x1) != ({}). Model parameters vector size must "
                            "be a 2 x 1 vector.".format(theta.shape))

        # get length of training samples in vector y
        m = y.shape[0]

        # create n x 1 vector of 1's
        x_vector = np.ones((1, m))

        #if len(y.shape) <= 1:  # if there are no cols defined define at least n x 1
        #    # create n x 1 vector of target data
        #    y = y[0:, None]

        # create a matrix based on x/y vectors
        X = np.matrix(np.insert(X, 0, x_vector, axis=1))
        # solve h(x) and create n x 1 prediction vector
        predictions = np.dot(X, theta)
        # compute cost function J(theta) calculate mean squared error (MSE) (x - y)^2 how close do we fit
        return 1/(2*m) * np.sum(np.power(predictions - y, 2))

    def multi_linear(self, X, y, theta, theta_sz=1):
        """

        :param X:
        :param y:
        :param theta:
        :return:
        """


        if not isinstance(theta, list) or theta is None:
            if theta_sz in (0, None):
                raise RuntimeError("Initial theta array size must be > 0 and not null")
            theta = np.zeros(int(theta_sz), 1) # set initial theta values to a 1 x n vector
        else:
            theta = np.array(theta, dtype='f')
            theta = theta[0:, None]

        # get length of training samples in vector y
        m = y.shape[0]
        # create n x 1 vector of 1's
        x_vector = np.ones((1, m))
        # create a matrix based ??
        X = np.matrix(np.insert(X, 0, x_vector, axis=1))
        # solve h(x) and create n x 1 prediction vector
        predictions = np.dot(X, theta)
        # compute cost function J(theta) calculate mean squared error (MSE) (x - y)^2 how close do we fit
        return np.sum(np.power(predictions - y, 2)) / (2*m)

    def logistic(self):
        pass
    def multi_logistic(self):
        pass
    def standardize(self):
        pass
    def normalize(self):
        pass

    @staticmethod
    def graph(X, **properties):

        graph = Plotter()
        graph.binary_scatterplot(X, **properties)
        plt.show()

    @property
    def alpha(self, eta):

        self.eta_ = eta

    @property
    def epochs(self, n_iter):

        self.n_iter_ = n_iter