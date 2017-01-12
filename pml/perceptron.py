import numpy as np

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """

        # initialize weights to zeros, Shape is a tuple (m#, feature#)
        self.w_ = np.zeros(1 + X.shape[1])  # [0., 0., 0.]
        self.errors_ = []

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
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""

        # perform matrice multiplication
        # W0X0 + (W1X1 + WmXm)
        value = np.dot(X, self.w_[1:]) + self.w_[0]
        return value

    def predict(self, X):
        """Return class label after unit step"""

        # threshold, if input >= 0.0 choose 1 else -1 label
        value = np.where(self.net_input(X) >= 0.0, 1, -1)
        return value