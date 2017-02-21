import numpy as np
import numbers
from . import doglegm
from . import cholupdate

def linear_leastsquares(X, y):
    """
    Use a closed-form solution to calculate linear regression against the
    training data. This forumula does not require any feature scaling. It
    is a very quick way to calculate the minimized costs for a linear regression.
    Note: Should only be applied to small data sets. Larger data sets should use
    gradient descent. The normal equation is a very computationally expensive
    calculation for the processor.  It is also an approximation.
    :param X:
    :param y:
    :return:
    """

    # linear least squares (Normal Equation) θ = (X.T * X)^−1 * X.T ⃗y .
    # ensure our training data X type is Matrix before performing calculation
    X = np.matrix(X)
    # calculate minimize cost paramateres using (Moore-Penrose) pseudo-inverse
    theta = np.linalg.pinv(X.T * X) * X.T * y

    return theta

def gradient_descent(X, y, theta=None, alpha=0.01, max_iter=1, costfunc=None):
    """
    Perform gradient descent calculation against training data.
    Gradient descent is a first-order iterative optimization algorithm.
    To find a local minimum of a function using gradient descent,
    one takes steps proportional to the negative of the gradient
    (or of the approximate gradient) of the function at the current point.
    :param X: array-like Array[n_samples, n_features] Training data
    :param y: np.ndarray Vector[n_samples] Training labels
    :param theta: array-like Vector[n_features] coefficient parameters
    :param alpha: float learning rate parameter. alpha is equal to 1 / C.
    :param max_iter: integer number of iterations for the solver.
    :param costfunc: (Optional) function pointer that solver can call to calculate min->cost
    :return: Tuple (theta, grad) theta contains the minimized cost coeffecient, if cost function
    parameter set, grad will contain the historical costs associated with gradient descent calculation
    """

    # This implementation uses a method known as "batch" gradient descent
    # where theta is updated on each iteration instead of at the very end.
    # With each "step" of gradient descent the parameters "theta" move
    # closer to being optimized to achieve lowest cost.  The formula is
    # expressed as thetaJ = thetaJ - alpha(1/m) * sum((h_thetaX - y)*(X))
    # h_thetaX is the linear model h_theta = theta0 + theta1 * X1

    cost_gradient = None
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
        cost_gradient = np.zeros((max_iter, 1))
    # suppress RuntimeWarning: overflow encountered due to NaN
    np.seterr(over='ignore')

    # loop through all iteration samples
    for i in range(0, max_iter):
        # calculate gradient descent
        theta -= (alpha/n_samples) * (X.T * (np.dot(X, theta) - y))
        if cost_gradient is not None:  # used to check and validate learning rate
            cost_gradient[i] = costfunc(X, y, theta)
    theta = np.nan_to_num(theta)  # set NaN to valid number 0
    cost_gradient = np.nan_to_num(cost_gradient)  # set NaN to valid number 0

    return theta, cost_gradient
