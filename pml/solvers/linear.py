import numpy as np
import numbers

def gradient_descent(X, y, theta=None, alpha=0.01, iterations=1, costfunc=None):
    """
    Perform gradient descent calculation against training data.
    Gradient descent is a first-order iterative optimization algorithm.
    To find a local minimum of a function using gradient descent,
    one takes steps proportional to the negative of the gradient
    (or of the approximate gradient) of the function at the current point.
    :param X: array-like Array[n_samples, n_features] Training data
    :param y: np.ndarray Vector[n_samples] Training labels
    :param theta: array-like Vector[n_features] sample weights <-- these are actually coef weights are the ones
    :param alpha: float learning rate parameter. alpha is equal to 1 / C.
    :param iterations: integer number of iterations for the solver.
    :param costfunc: Optional function pointer that solver can call to calculate min->cost
    :return: Tuple (theta, grad) theta contains the minimized cost coeffecient, if cost function
    parameter set, grad will contain the historical costs associated with gradient descent calculation
    """

    # extract the array shape of row samples and column features
    n_samples, n_features = X.shape

    # check theta params and ensure correct data type
    if isinstance(theta, numbers.Number):
        theta = None
    elif not isinstance(theta, np.ndarray) and theta is not None:
        theta = np.array(theta, dtype='f')

    # initialize weights (theta) using supplied or set to zero
    if theta is None:  # initialize weights to zero
        theta = np.zeros((n_features + 1, 1))
    else:
        theta = theta  # use values passed in

    # FIXME: Maybe use sparse instead of numpy.matrix?
    X = np.matrix(X)

    if costfunc:
        # create history matrix to store cost values
        grad = np.zeros((iterations, 1))
    # suppress RuntimeWarning: overflow encountered due to NaN
    np.seterr(over='ignore')

    # loop through all iteration samples
    for i in range(0, iterations):
        theta = np.nan_to_num(theta)  # set NaN to valid number 0
        # calculate gradient descent
        theta -= (alpha/n_samples) * (X.T * (np.dot(X, theta) - y))
        if grad is not None:  # used to check and validate learning rate
            grad[i] = costfunc(X, y, theta)
            grad = np.nan_to_num(grad)  # set NaN to valid number 0

    return theta, grad
