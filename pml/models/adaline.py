import models.classifier as classifier
import numpy as np

class AdalineGD(classifier.Classifier):
    """Adaptive Linear Neuron classifier"""

    def __init__(self, eta=0.01, n_iter=50):
        """

        :param eta:
        :param n_iter:
        """

        super(AdalineGD, self).__init__(eta, n_iter)

    def fit(self, X, y, n_iter=None, eta=None):
        """
        Fit the training data
        :param X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        :param y: array-like, shape = [n_samples]
            Target values.
        :return: AdalineGC object
        """

        super(AdalineGD, self).fit(X, y, n_iter=n_iter, eta=eta)
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta_ * X.T.dot(errors)
            self.w_[0] += self.eta_ * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        #
        self.fitted_ = True
        return self