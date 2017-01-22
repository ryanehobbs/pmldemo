import numpy as np


class Classifier(object):

    eta = None
    n_iter = None
    w_ = None
    errors = None

    def __init__(self, eta=0.01, n_iter=10):
        """
        Initialize Perceptron class object
        :param eta: float Learning rate (between 0.0 and 1.0)
        :param n_iter: int Passes over the training dataset.
        """
        self.eta = eta
        self.n_iter = n_iter

    def predict(self, X):
        """
        Predict class label
        :param X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        :return: Return class label after unit step
        """

        # threshold, if input >= 0.0 choose 1 else -1 label
        value = np.where(self.net_input(X) >= 0.0, 1, -1)
        return value

    def net_input(self, X):
        """
        Calculate input values
        :param X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        :return: Sum of input values
        """

        # perform matrice multiplication
        # W0X0 + (W1X1 + WmXm)
        value = np.dot(X, self.w_[1:]) + self.w_[0]
        return value

    def activation(self, X):
        """

        :param X:
        :return:
        """

        return self.net_input(X)