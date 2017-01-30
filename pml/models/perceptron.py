"""Perceptron classifier"""

import models.classifier as classifier
import numpy as np

class Perceptron(classifier.Classifier):
    """Perceptron classifier.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.01, n_iter=10):
        """
        Initialize Perceptron class object
        :param eta: float Learning rate (between 0.0 and 1.0)
        :param n_iter: int Passes over the training dataset.
        """

        super(Perceptron, self).__init__(eta, n_iter)

    def fit(self, X, y, n_iter=None, eta=None):
        """
        Fit the training data
        :param X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        :param y: array-like, shape = [n_samples]
            Target values.
        :return: Perceptron object
        """

        super(Perceptron, self).fit(X, y, n_iter=n_iter, eta=eta)
        # initialize weights to zeros, Shape is a tuple (m#, feature#)
        self.w_ = np.zeros(1 + X.shape[1])  # [0., 0., 0.]
        self.cost_ = []

        for _ in range(self.n_iter):
            errors = 0
            # xi = sample array, target = classification 1 or -1
            for xi, target in zip(X, y):
                # delta<w> = learn-rate(target class label - predicted class label)
                update = self.eta * (target - self.predict(xi))
                # multiply delta<w> by values in target array and update weights
                self.w_[1:] += update * xi
                # update bias first weight based on delta<w>
                self.w_[0] += update
                # if not 0 increment error count
                errors += int(update != 0.0)
            # record convergence errors for each epoch (iteration) this will establish the decision boundary
            self.cost_.append(errors)

        #
        self.fitted = True
        return self