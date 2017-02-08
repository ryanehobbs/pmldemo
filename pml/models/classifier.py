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

    # calculate sigmoid curve g(z) = 1/(1+e^-z)
    g = np.divide(1, 1 + np.exp(-z))
    return g

def linear_param(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        # matrix should be first element in args
        X = args[1]
        if len(args) >= 4:
            theta = kwargs.get('theta') or args[3]
        else:
            theta = kwargs.get('theta')
        xargs = list(args)

        if isinstance(X, np.matrix):
            X = np.array(X)
        # check if we already inserted theta zero
        theta_set = len(set(X[:, 0]))
        if theta_set != 1:
            # get diminesions of matrix
            m, n = X.shape
            # insert n x 1 vector of 1's into matrix X
            X = np.matrix(np.insert(X, 0, np.ones((1, m)), axis=1))
            xargs[1] = X
            if not theta:
                # initialize fitting params (initial theta params)
                initial_theta = np.zeros((n+1, 1))
                kwargs['theta'] = initial_theta
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
        j_cost = 1/(2*m) * np.sum(np.power(predictions - y, 2))
        return j_cost
        #return np.sum(np.power(predictions - y, 2)) / (2*m)

    @linear_param
    def gradient_descent(self, X, y, theta=[0,0], alpha=0.01, iterations=0):
        """
        Gradient descent can be used in both Linear univariate and multivariate
        calculations

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

        # suppress RuntimeWarning: overflow encountered due to NaN
        np.seterr(over='ignore')

        for i in range(0, iterations):
            theta = np.nan_to_num(theta)  # set NaN to valid number 0
            theta -= (alpha/m) * (X.T * (np.dot(X, theta) - y))
            j_history[i] = self.linear_costcalc(X, y, theta)

        # set NaN to valid number 0
        j_history = np.nan_to_num(j_history)

        return (X, theta, j_history)

    @linear_param
    def logistic_costcalc(self, X, y, theta=[0,0], lambdaR=None):
        """

        :param X:
        :param y:
        :param theta:
        :param lambdaR: Apply regularization (usually 1.0)
        :return:
        """

        # X is a m x n Matrix
        # theta is a n x 1 vector
        # result is n x 1 vector for h

        if not isinstance(theta, np.ndarray):
            theta = np.array(theta, dtype='f')
            theta = theta[0:, None]

        # get length of training samples in vector y
        m = y.shape[0]
        # calculate the hypothesis (activation - sigmoid)
        predictions = sigmoid(np.dot(X, theta))
        j_cost = (1/m) * np.sum(np.multiply(-y, np.log(predictions)) - np.multiply((1-y), np.log(1-predictions)))
        if lambdaR:
            reg_term = (lambdaR/(2*m)) * np.sum(np.power(theta[2:], 2))
            j_cost = j_cost + reg_term
        # return minJ which is the minimized cost calculation
        return j_cost

    def linear_regression(self, X, y, theta=[0,0], **properties):
        """

        :param X:
        :param y:
        :param properties:
        :return:
        """

        Xm, theta, j_history = self.gradient_descent(X, y, theta, alpha=0.01, iterations=1500)

        return (theta, j_history)

    def logistic_regression(self, X, y, theta=[0,0], **properties):
        pass