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

# create a new function that called fminfunc
# fminfunc will take a pointer to a cost function/method with X,y and t)
# it will have **kwargs that set the iteration, and alpha and any they option
# so it would be called as fminfunc(*pointer_to_costfunc, X, y, initial_theta, **kwargs)

def fminfunc(costfunc, X, y, theta, **kwargs):
    """

    :param costfunc:
    :param X:
    :param y:
    :param theta:
    :param kwargs:
    :return:
    """


    max_iter = kwargs.get('max_iter', 400)

    nargout = 2
    has_grad = True
    factor = 0.1
    x = theta[:]
    xsz = x.shape
    tolf = 1e-7
    tolx = 1e-7
    n = len(x)
    nfev = 0
    info = 0
    # initial evaluation reshape
    fval = costfunc(X, y, theta)[0]
    grad = np.zeros((n, 1))
    dg = np.ones((n, 1))

    if theta.dtype == np.float32:
        macheps = np.spacing(np.single(1))
    elif theta.dtype == np.float64:
        macheps = np.spacing(np.double(1))

    nsuciter = 0
    lastratio = 0

    for i in range(0, max_iter):
        grad0 = grad
        fval, grad = costfunc(X, y, theta)
        grad = grad[:]
        nfev += 1

        if i == 0:
            hesr = np.matrix(np.identity(n))
        else:
            # Use the damped BFGS formula.
            y2 = grad - grad0
            sBs = np.sum(np.power(w, 2))
            Bs = hesr.T * w
            sy = np.sum(np.dot(y2.T, s))
            theta2 = 0.8 / max(1 - sy / sBs, 0.8)
            r = theta2 * y2 + (1 - theta2) * Bs
            hesr = cholupdate(hesr, r / np.sum(np.sqrt(s.T * r)), "+");
            hesr = cholupdate(hesr, Bs / np.sqrt(sBs), "-");
            # FIXME: Possibly trap exception if downdate is not successful
            #if info:
                #hesr = np.matrix(np.identity(n))
            # FIXME: End

        if i == 0:
            xn = np.linalg.norm(np.multiply(dg, x))
            delta = factor * max(xn, 1)

        if (np.linalg.norm(grad) <= tolf * n* xn):
            info = 1;
            break

        suc = False
        decfac = 0.5

        # inner loop
        while(not suc and i < max_iter):
            s = -doglegm(hesr, grad, dg, delta)
            sn = np.linalg.norm(np.multiply(dg, s))
            if i == 0:
                delta = min(delta, sn)

            fval1 = costfunc(X, y, x+s)[0]

            if fval1 < fval:
                # Scaled actual reduction.
                actred = (fval - fval1) / (abs(fval1) + abs(fval))
            else:
                actred = -1

            w = hesr * s
            # Scaled predicted reduction, and ratio.
            # FIXME: sum of squares in some books is np.sum(w ** 2)
            t = 1/2 * np.sum(np.power(w, 2)) + np.sum(np.dot(grad.T, s))
            if t < 0:
                prered = -t/(abs (fval) + abs (fval + t))
                ratio = actred / prered
            else:
                prered = 0
                ratio = 0

            ## Update delta.
            if (ratio < min (max (0.1, 0.8 * lastratio), 0.9)):
                delta *= decfac
                decfac = np.power(decfac, 1.4142)
                if (delta <= 10 * macheps * xn):
                    ## Trust region became uselessly small.
                    info = -3
                    break
            else:
                lastratio = ratio
                decfac = 0.5
                if (abs (1 - ratio) <= 0.1):
                    delta = 1.4142 * sn
                elif (ratio >= 0.5):
                    delta = max (delta, 1.4142 * sn)

            if (ratio >= 1e-4):
                ## Successful iteration.
                x += s
                xn = np.linalg.norm(np.multiply(dg, x))
                fval = fval1
                nsuciter += 1
                suc = True

    ## When info != 1, recalculate the gradient and Hessian using the final x.
    if (nargout > 4 and (info == -1 or info == 2 or info == 3)):
        grad0 = grad;
        if has_grad:
            [fval, grad] = costfunc(X, y, theta)
            grad = grad[:]
        if nargout > 5:
            ## Use the damped BFGS formula.
            y2 = grad - grad0
            sBs = np.sum(np.power(w, 2))
            Bs = hesr.T * w
            sy = y2.T * s
            theta2 = 0.8 / max (1 - sy / sBs, 0.8)  # FIXME: should we relabel theta?
            r = theta2 * y2 + (1-theta2) * Bs
            hesr = cholupdate(hesr, r / np.sum(np.sqrt(s.T * r)), "+");
            hesr = cholupdate(hesr, Bs / np.sqrt(sBs), "-");
            # FIXME: Possibly trap exception if downdate is not successful
            #if info:
            #hesr = np.matrix(np.identity(n))
            # FIXME: End

        ## Return the gradient in the same shape as x
        grad = np.reshape(grad, xsz)

    ## Restore original shapes.
    x = np.reshape(x, xsz)

    if nargout > 5:
        hess = hesr.T * hesr;

    print('done')

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
