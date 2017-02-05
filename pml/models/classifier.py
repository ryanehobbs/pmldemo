import numpy as np
from graphs.plotter import Plotter
import matplotlib.pyplot as plt
from functools import wraps

def sigmoid(z):
    """
    Calculate sigmoid function A sigmoid function
    is a mathematical function having an "S"
    shaped curve (sigmoid curve). It is capped
    by -1 and 1 and crosses 0.5.  Typically used
    as an activation function
    :param z: z can be a matrix, vector or scalar
    :return:
    """

    # create n x 1 vector of zeros
    g = np.zeros(len(z))
    # calculate sigmoid
    g = np.divide(1, 1 + np.exp(np.power(z, 2)))
    return g

def linear_param(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        # matrix should be first element in args
        X = args[1]
        y = args[2]
        xargs = list(args)

        if isinstance(X, np.matrix):
            X = np.array(X)
        size = y.shape[0]
        # check if we already inserted theta zero
        theta_set = len(set(X[:, 0]))
        if theta_set != 1:
            # create n x 1 vector of 1's
            x_vector = np.ones((1, size))
            # create a matrix based on x/y vectors
            X = np.matrix(np.insert(X, 0, x_vector, axis=1))
            # insert back into args
            xargs[1] = X
        # call wrapped method
        return func(*xargs, **kwargs)
    return wrapper

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

    def predict(self, parameters, theta_params, resolution):
        """

        :param parameters:
        :param theta_params:
        :param resolution:
        :return:
        """

        value = 0
        if not isinstance(parameters, np.ndarray):
            parameters = np.array(parameters, dtype='f')
            parameters = parameters[0:, None]

        product_vector = (parameters*theta_params)*resolution

        for i in range(0, len(product_vector)):
            value = abs(product_vector[i]) - value

        return value[0]

    @linear_param
    def linear_costcalc(self, X, y, theta=[0,0]):
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

        if not isinstance(theta, np.ndarray):
            theta = np.array(theta, dtype='f')
            theta = theta[0:, None]

        # get length of training samples in vector y
        m = y.shape[0]
        # solve h(x) and create n x 1 prediction vector
        predictions = np.dot(X, theta)
        # compute cost function J(theta) calculate mean squared error (MSE) (x - y)^2 how close do we fit
        #return 1/(2*m) * np.sum(np.power(predictions - y, 2))
        return np.sum(np.power(predictions - y, 2)) / (2*m)

    @linear_param
    def gradient_descent(self, X, y, theta=[0,0], alpha=0.01, iterations=0):
        """

        :param X:
        :param y:
        :param theta:
        :param alpha:
        :param iterations:
        :return:
        """

        if not isinstance(theta, np.ndarray):
            theta = np.array(theta, dtype='f')
            theta = theta[0:, None]


        m = y.shape[0]
        # create history matrix
        j_history = np.zeros((iterations, 1))

        for i in range(0, iterations):
            # FIXME: this may be possible to vectorize this implementation using numpy
            theta -= (alpha/m) * (X.T * (np.dot(X, theta) - y))
            j_history[i] = self.linear_costcalc(X, y, theta)

        theta = np.nan_to_num(theta)
        j_history = np.nan_to_num(j_history)

        return (X, theta, j_history)

    @linear_param
    def logistic_costcalc(self, X, y, theta=[0,0], regularization=False):

        # X is a m x n Matrix
        # theta is a n x 1 vector
        # result is n x 1 vector for h

        if not isinstance(theta, np.ndarray):
            theta = np.array(theta, dtype='f')
            theta = theta[0:, None]

        # get length of training samples in vector y
        m = y.shape[0]
        predictions = sigmoid(np.dot(X, theta))
        # // J = (1/m) * sum(-y .* log(h) - (1-y) .* log(1-h));
        J = (1/m) * np.sum(np.multiply(-y, np.log(predictions)) - np.multiply(1-y), np.log(1-predictions))

        return J

    def linear_regression(self, X, y, theta=[0,0], **properties):
        """

        :param X:
        :param y:
        :param properties:
        :return:
        """

        Xm, theta, j_history = self.gradient_descent(X, y, theta, alpha=0.01, iterations=1500)
        #Classifier.graph_scatter(X, y)
        #Classifier.graph_line(X[:,0], np.dot(Xm, theta))
        #plt.show()

        return (theta, j_history)

    @staticmethod
    def graph_scatter(X, y, **properties):

        graph = Plotter()
        graph.scatterplot(X, y)

    @staticmethod
    def graph_line(X, y, **properties):

        graph = Plotter()
        graph.lineplot(X, y)

    @property
    def alpha(self, eta):

        self.eta_ = eta

    @property
    def epochs(self, n_iter):

        self.n_iter_ = n_iter