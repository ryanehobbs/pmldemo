import neuron
import numpy as np

class AdalineGD(neuron.Neuron):
    """Adaptive Linear Neuron classifier"""

    def __init__(self, eta=0.01, n_iter=50):
        """

        :param eta:
        :param n_iter:
        """

        super(AdalineGD, self).__init__(eta, n_iter)

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum / 2.0
            return self.cost_.append(cost)