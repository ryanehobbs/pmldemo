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