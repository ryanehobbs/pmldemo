import math
import numpy as np


def _sumsq(X):
    """
    Helper function to calculate sum of squares of a vector or matrix
    :param X: array-like Array[values]
    :return: Single value of a sum of squares calculation against a vector or matrix
    """

    return np.sum(np.power(X, 2))

def fminfunc(costfunc, X, y, initial_theta, **kwargs):
    """

    :param costfunc:
    :param X:
    :param y:
    :param initial_theta:
    :param kwargs:
    :return:
    """

    # key references
    # https://www.cs.nyu.edu/overton/papers/pdffiles/bfgs_exactLS.pdf
    # http://dinamico2.unibg.it/icsp2013/doc/ps/albAALI-On%20a%20Limited-Memory%20Damped-BFGS%20Method%20for%20Large%20Scale%20Optimization.pdf
    # http://sepwww.stanford.edu/data/media/public/docs/sep117/antoine1/paper_html/node5.html
    # https://www.researchgate.net/publication/277405097_14Jota_AGM_lbfgs#pfc
    # https://en.wikipedia.org/wiki/Limited-memory_BFGS
    # https://en.wikipedia.org/wiki/Symmetric_rank-one
    # https://en.wikipedia.org/wiki/Line_search
    # https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm

    _FACTOR = 0.1  # static constant used for initial delta calc at iteration 0
    _SQRT_2 = 1.4142

    max_iter = kwargs.get('max_iter', 400)  # max iterations to process
    max_evals = kwargs.get('max_evals', math.inf)  # max function evaluations
    grad_obj = kwargs.get('grad_obj', False)  # solve the gradient objective equation

    tolf = 1e-7  # tolerance for the objective function value
    tolx = 1e-7  # termination tolerance for the unknown variables
    n_funcval = 0  # count of function evaluations used with max_evals
    n_iter = 0  # count of overall convergence loop iteration
    lastratio = 0  # variable to store last ratio used in iteration
    info = 0  # variable that takes on result information

    # create a copy of initial theta to be used in calculations
    theta_cpy = initial_theta[:]
    # get row,col shape tuple of initial theta
    theta_sz = initial_theta.shape
    # get the len of elements in initial theta
    theta_len = len(initial_theta)
    # create n x 1 vector for Double-Dogleg Optimization (DBLDOG)
    dbldog = np.ones((theta_len, 1))
    # create n x 1 vector for gradient calculations this is the initial iter0 copy
    grad = np.zeros((theta_len, 1))

    # Distance between 1 and the nearest floating point number. For 32 or 64 bit
    if initial_theta.dtype == np.float32:
        macheps = np.spacing(np.single(1))
    elif initial_theta.dtype == np.float64:
        macheps = np.spacing(np.double(1))

    # initialize cost
    cost_outer, _ = costfunc(X, y, initial_theta)

    # outer loop of convergence
    while(n_iter < max_iter and n_funcval < max_evals and not info):

        success = False  # TODO: add comments here
        descfact = 0.5  # TODO: add comments here

        # set previous gradient on each iteration
        grad_prev = grad

        # call cost function and set gradient value
        cost_outer, grad = costfunc(X, y, initial_theta)
        grad = grad[:]
        n_funcval += 1

        # initial loop create identify matrix for hessian based in size
        # of initial theta. Normalize initial theta values and get initial
        # delta based on constant _FACTOR and the max value of theta or 1
        if n_iter == 0:
            hesr = np.matrix(np.identity(theta_len))
            theta_norm = np.linalg.norm(dbldog * theta_cpy)
            delta = _FACTOR * max(theta_norm, 1)
        else:  # FIXME: possibly wrap this section in separate method / function
            # Use the damped BFGS formula.
            grad_delta = grad - grad_prev  # calculate gradient increment
            sBs = _sumsq(w)   # positive deﬁniteness of BGFS
            Bs = hesr.T * w  #  hess approximation
            sy = np.sum(np.dot(grad_delta.T, s))  # part of hess factor
            theta = 0.8 / max(1 - sy / sBs, 0.8)  # calculate inverse Hessian
            r = theta * grad_delta + (1 - theta) * Bs  # update inverse Hessian
            hesr = cholupdate(hesr, r / np.sum(np.sqrt(s.T * r)), "+")  # Update Cholesky decomposition for pos triangle
            hesr = cholupdate(hesr, Bs / np.sqrt(sBs), "-")  # Update Cholesky decomposition for neg triangle

        if (np.linalg.norm(grad) <= tolf * theta_len * theta_norm):
            # Converged to a solution point.  Relative gradient error is less than
            # specified by tolf (default 1e-7)
            info = 1

        # inner loop of convergence
        while(not success and n_iter <= max_iter and n_funcval < max_evals and not info):

            # use the double dog leg optimization s^k = alpha_s1^k = alpha_s2^k,
            # where s1 = ascent search direction, s2 = quasi-Newton search direction
            # subtract because we are stepping down
            s = -doglegm(hesr, grad, dbldog, delta)
            s_norm = np.linalg.norm(dbldog * s)

            # initial loop get the delta based on minimum between
            # current delta and normalized dbldog optimization
            if n_iter == 0:
                delta = min(delta, s_norm)

            # call costfunction and evaluate cost based on initial cost in outer loop
            cost_inner = costfunc(X, y, theta_cpy + s)[0]

            if cost_inner < cost_outer:  # scaled actual reduction (average)
                actred = (cost_outer - cost_inner) / (abs(cost_inner) + abs(cost_outer))
            else:
                actred = -1

            # Scaled predicted reduction, and ratio.
            w = hesr * s  # coefficient vector between hesr and dogleg optimization
            t = 1/2 * _sumsq(w) + np.sum(np.dot(grad.T, s))
            if t < 0:  # predict reduction step to take
                prered = -t/(abs(cost_outer) + abs(cost_outer + t))
                ratio = actred / prered
            else:
                ratio = 0

            # update delta values for gradient descent
            ratio_min = min(max(0.1, 0.8 * lastratio), 0.9)
            if (ratio < ratio_min):
                delta *= descfact  # multiply delta by current descent factor
                descfact = np.power(descfact, _SQRT_2)  # new descent factor is current^sqrt(2)
                trust_reg = 10 * macheps * theta_norm  # calc trust region based on line spacing and normalized theta
                if (delta <= trust_reg):  # The trust region radius became excessively small. (diverged)
                    info = -3
                    break  # break execution from inner loop
            else:
                lastratio = ratio  # store current ratio
                descfact = 0.5  # reset descfact back to default 0.5
                if (abs(1 - ratio) <= 0.1):
                    delta = _SQRT_2 * s_norm
                elif (ratio >= 0.5):
                    delta = max(delta, _SQRT_2 * s_norm)

            # Successful iteration.
            if (ratio >= 1e-4):   # found minimum set initial theta values for the next descent
                theta_cpy += s
                theta_norm = np.linalg.norm(dbldog * theta_cpy)  # calculate 2-norm from vector product
                cost_outer = cost_inner
                success = True

            n_iter += 1  # increment iteration count

    # Restore original shapes.
    theta_cpy = np.reshape(theta_cpy, theta_sz)
    return theta_cpy, cost_outer

def doglegm(r, g, d, delta):
    """
    The double-dogleg optimization method combines the ideas of the quasi-Newton
    and trust region methods. In each iteration, the double-dogleg algorithm computes
    the step as the linear combination of the steepest descent or ascent search direction
    and a quasi-Newton search direction.

    Solve the double dogleg trust-region minimization problem:
    Minimize 1/2 * norm(r * x)^2 subject to the constraint norm(d.T * x) <= delta,
    x being a convex combination of the gauss-newton and scaled gradient.

    :param r: Diagonal scaling matrix
    :param g: Gradient scaling matrix
    :param dg:
    :param delta:
    :return:
    """

    # https://github.com/dkogan/libdogleg/blob/master/dogleg.c
    # http://support.sas.com/documentation/cdl/en/etsug/60372/HTML/default/viewer.htm#etsug_nlomet_sect006.htm

    # solve matrix for s2 direction quasi-Newton search direction
    b = np.linalg.solve(r.T, g)
    # solve matrix for s1 direction descent/ascent search direction
    x = np.linalg.solve(r, b)
    # calc initial trust region size
    xn = np.linalg.norm(np.multiply(d, x))

    # begin by getting gauss newton direction
    if (xn > delta):  # gauss newton is too big, get scaled gradient.
        s_dir = np.divide(g, d)
        norm_grad = np.linalg.norm(s_dir)
        if (norm_grad > 0):  # normalize and rescale
            # calculate the newton s direction
            s_dir = np.divide((s_dir / norm_grad), d)
            # get line minimizer in s direction
            s_mindiv = np.linalg.norm(r * s_dir)
            s_minline = (norm_grad / s_mindiv) / s_mindiv
            if (s_minline < delta):  # get the dogleg path minimizer
                # normalize trust region delta
                dxn = delta / xn
                # get min trust region delta
                snmd = s_minline / delta
                # normalize vector
                bn = np.linalg.norm(b)
                # calculate the step selection by dividing by gauss-newton
                t = (bn / norm_grad) * (bn / xn) * snmd
                t -= dxn * np.power(snmd, 2) - np.sqrt(np.power((t-dxn), 2) + (1 - np.power(dxn, 2)) * (1 - np.power(snmd, 2)))
                alpha = dxn * (1 - np.power(snmd, 2)) / t
            else:
                alpha = 0  # no step selection delta of trust region greater than line minimizer in s direction
        else:
            # calculate the step selection by dividing scaled gradient
            alpha = delta / norm_grad
            s_minline = 0
        # form convex combination
        x = alpha * x + ((1 - alpha) * min(s_minline, delta)) * s_dir

    return x

# https://en.wikipedia.org/wiki/Cholesky_decomposition
def cholupdate(R, x, sign):
    """
    Update a Cholesky factorization. This will perform a Rank 1
    update if sign is positive or a Rank 1 downdate if sign is negative.

    Update the upper right triangle portion of the Cholesky factorized
    matrix.

    :param R: Original Cholesky factorization matrix
    :param x: Column vector of appropriate length to add or subtract
    :param sign: + implies (R + x * x') - implies (R - x * x')
    :return: Updated Cholesky factorized matrix
    """

    # get length of vector X
    p = np.size(x)
    # iterate up to length of vector X
    for k in range(p):
        if sign == '+':  # rank 1 update
            inner = np.sum(np.power(R[k, k], 2) + (np.sum(np.power(x[k], 2))))
            r = np.sqrt(inner)
        elif sign == '-':  # rank 2 downdate
            inner = np.sum(np.power(R[k, k], 2) - (np.sum(np.power(x[k], 2))))
            r = np.sqrt(inner)

        # calculate outer product and update R matrix
        c = np.divide(r, R[k, k])
        s = np.sum(np.divide(x[k], R[k, k]))
        R[k, k] = r

        # based on sign update upper right triangle of matrix in place
        if sign == '+':  # rank 1 update
            R[k, k + 1:p] = (R[k, k + 1:p] + s * x[k + 1:p].T)/c
        elif sign == '-':  # rank 2 downdate
            R[k, k + 1:p] = (R[k, k + 1:p] - s * x[k + 1:p].T)/c

        # update vector x in place
        x[k + 1:p] = c * x[k + 1:p] - s * R[k, k + 1:p].T

    return R
