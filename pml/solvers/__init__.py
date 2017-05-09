import math
import numpy as np
from pml import DeferredExec

def _sumsq(X):
    """
    Helper function to calculate sum of squares of a vector or matrix
    :param X: array-like Array[values]
    :return: Single value of a sum of squares calculation against a vector or matrix
    """

    return np.sum(np.power(X, 2))

def fmincg(costfunc, X, y, initial_theta, **kwargs):
    """
    Minimize a differentiable multivariate function. Implementation
    from Carl Edward Rasmussen (2006-09-08).
    :param costfunc:
    :param X:
    :param y:
    :param initial_theta:
    :param kwargs:
    :return:
    """

    _RHO = 0.01    # Wolfe-Powell condition minimum allowed fraction of the expected slope
    _SIG = 0.5     # Wolfe-Powell condition maximum allowed absolute ratio between previous and new slopes
    _INT = 0.1     # don't reevaluate within 0.1 of the limit of the current bracket
    _EXT = 3.0     # extrapolate maximum 3 times the current bracket
    _MAX = 20      # max 20 function evaluations per line search
    _RATIO = 100   # maximum allowed slope ratio

    max_iter = kwargs.get('max_iter', 400)  # max iterations to process
    max_evals = kwargs.get('max_evals', math.inf)  # max function evaluations
    grad_obj = kwargs.get('grad_obj', False)  # solve the gradient objective equation
    lambda_r = kwargs.get('lambda_r', 0)  #

    if np.max(np.size(max_iter)) == 1:
        red = 1
    elif np.max(np.size(max_iter)) == 2:
        red = 2

    # count epoch
    i = 0 + (max_iter < 0)
    # line search failed flag
    ls_failed = 0
    # convert bool type in y ndarray to int values
    if y is not None:
        y = y.astype(int)
    # call into cost function to set up initial values
    # ->>> cost, grad = costfunc(X, y, initial_theta)
    cost, grad = DeferredExec.call_func(costfunc, X, y, initial_theta)
    # cost array
    costX = []
    # search direction is steepest
    search_direction = -grad
    # slope
    slope = np.dot(-search_direction.T, search_direction)
    # get initial step
    initial_step = red / (1 - slope)

    # copy and set current values suffix0
    cost1 = cost
    grad1 = grad
    slope1 = slope
    initial_step1 = initial_step
    theta = initial_theta if initial_theta is not None else X

    #for i in range(0, np.abs(max_iter)):
    while i < np.abs(max_iter):

        # set current values suffix0 X0 = X; f0 = f1; df0 = df1;
        cost0 = cost1
        grad0 = grad1
        theta0 = theta

        # increment count
        i = i + (max_iter > 0)

        # start the line search
        theta = theta + initial_step1 * search_direction
        # call into cost function
        # ->>>  cost2, grad2 = costfunc(X, y, theta)
        cost2, grad2 = DeferredExec.call_func(costfunc, X, y, theta)
        # calculate new slope
        slope2 = np.dot(grad2.T, search_direction)
        # count epoch
        i = i + (max_iter < 0)
        # initialize point 3 equal to point 1
        cost3 = cost1
        slope3 = slope1
        initial_step3 = -initial_step1

        if max_iter > 0:
            M = _MAX
        else:
            M = min(_MAX, -max_iter-i)
        # initialize quantities
        success = 0
        limit = -1

        while True:  # begin long running process
            while ((cost2 > cost1 + initial_step1 * _RHO * slope1) or (slope2 > -_SIG * slope1)) and (M > 0):
                limit = initial_step1  # tighten the bracket
                if cost2 > cost1:  # quadratic fit
                    offset = (0.5 * slope3 * initial_step3 * initial_step3) / (slope3 * initial_step3 + cost2 - cost3)
                    initial_step2 = initial_step3 - offset
                else:  # cubic fit
                    A = 6 * (cost2 - cost3) / initial_step3 + 3 * (slope2 + slope3)
                    B = 3 * (cost3 - cost2) - initial_step3 * (slope3 + 2 * slope2)
                    initial_step2 = (np.sqrt(B * B - A * slope2 * initial_step3 * initial_step3) - B)/A
                if np.isnan(initial_step2) or np.isinf(initial_step2):  # bisect if numerical problem
                    initial_step2 = initial_step3 / 2

                initial_step2 = max(min(initial_step2, _INT * initial_step3), (1 - _INT) * initial_step3)  # do not accept too close to limits
                initial_step1 = initial_step1 + initial_step2  # update the step
                theta += initial_step2 * search_direction
                # ->>>  cost2, grad2 = costfunc(X, y, theta)
                cost2, grad2 = DeferredExec.call_func(costfunc, X, y, theta)
                M -= 1
                i = i + (max_iter < 0)  # count epoch
                slope2 = np.dot(grad2.T, search_direction)
                initial_step3 = initial_step3 - initial_step2  # z3 is now relative to the location of z2

            if cost2 > cost1 + initial_step1 + _RHO * slope1 or slope2 > -_SIG * slope1:
                break  # this is a failure
            elif slope2 > _SIG * slope1:
                success = 1
                break  # success
            elif M == 0:
                break  # this is a failure

            # cubic extrapolation
            A = 6 * (cost2 - cost3) / initial_step3 + 3 * (slope2 + slope3)
            B = 3 * (cost3 - cost2) - initial_step3 * (slope3 + 2 * slope2)
            initial_step2 = -slope2 * initial_step3 * initial_step3 / (B + np.sqrt(B * B - A * slope2 * initial_step3 * initial_step3))

            if not np.isreal(initial_step2) or np.isnan(initial_step2) or np.isinf(initial_step2) or initial_step2 < 0:  # num prob or wrong sign?
                if limit < -0.5:  # if we have no upper limit
                    initial_step2 = initial_step1 * (_EXT - 1)  # the extrapolate the maximum amount
                else:
                    initial_step2 = (limit - initial_step1) / 2  # otherwise bisect
            elif limit > -0.5 and (initial_step2 + initial_step1 > limit):  # extraplation beyond max?
                initial_step2 = (limit - initial_step1) / 2  # bisect
            elif limit < -0.5 and (initial_step2 + initial_step1 > initial_step1 * _EXT):  # extrapolation beyond limit
                initial_step2 = initial_step1 * (_EXT - 1.0)  # set to extrapolation limit
            elif initial_step2 < -initial_step3 * _INT:
                initial_step2 = -initial_step3 * _INT
            elif limit > -0.5 and (initial_step2 < (limit - initial_step1) * (1.0 - _INT)):  # too close to limit?
                initial_step2 = (limit - initial_step1) * (1.0 - _INT)

            # set point 3 equal to point 2
            cost3 = cost2
            slope3 = slope2
            initial_step3 = -initial_step2

            # update estimates
            initial_step1 += initial_step2
            theta += initial_step2 * search_direction
            # ->>>  cost2, grad2 = costfunc(X, y, theta)
            cost2, grad2 = DeferredExec.call_func(costfunc, X, y, theta)
            M -= 1
            i = i + (max_iter < 0)  # count epoch
            slope2 = np.dot(grad2.T, search_direction)
            #  end of line search

        if success:  # line search succeeded
            cost1 = cost2
            costX.append(cost1)
            #print("Iteration %d | Cost: %4.6E" % (i, cost1), end='\n')
            # Polack-Ribiere direction
            search_direction = (np.dot(grad2.T, grad2) - np.dot(grad1.T, grad2)) / \
                               (np.dot(grad1.T, grad1)) * search_direction - grad2
            grad1, grad2 = grad2, grad1  # swap derivatives
            slope2 = np.dot(grad1.T, search_direction)
            if slope2 > 0:
                search_direction = -grad1
                slope2 = np.dot(-search_direction.T, search_direction)
            initial_step1 = initial_step1 * min(_RATIO, slope1/(slope2-np.finfo(np.double).eps))  # slope ratio but max RATIO
            slope1 = slope2
            ls_failed = 0  # this line search did not fail
        else:  # restore point from before failed line search X = X0; f1 = f0; df1 = df0
            theta = theta0
            cost1 = cost0
            grad1 = grad0
            if ls_failed or i > abs(max_iter):  # line search failed twice in a row
                break  # or we ran out of time, so we give up
            grad1, grad2 = grad2, grad1  # swap derivatives
            search_direction = -grad1  # try steepest
            slope1 = np.dot(-search_direction.T, search_direction)
            initial_step1 = 1/(1-slope1)
            ls_failed = 1  # this line search failed

    # convert to numpy ndarray
    costX = np.array(costX)

    return theta, costX, i

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
    lambda_r = kwargs.get('lambda_r', 0)  # get the regularization parameter

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
    grad = np.zeros(theta_len)

    # Distance between 1 and the nearest floating point number. For 32 or 64 bit
    if initial_theta.dtype == np.float32:
        macheps = np.spacing(np.single(1))
    elif initial_theta.dtype == np.float64:
        macheps = np.spacing(np.double(1))

    # initialize cost
    cost_outer, _ = costfunc(X, y, initial_theta, lambda_r=lambda_r)

    # outer loop of convergence
    while(n_iter < max_iter and n_funcval < max_evals and not info):

        success = False  # TODO: add comments here
        descfact = 0.5  # TODO: add comments here

        # set previous gradient on each iteration
        grad_prev = grad

        # call cost function and set gradient value
        cost_outer, grad = costfunc(X, y, theta_cpy, lambda_r=lambda_r)
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

            # call cost function and evaluate cost based on initial cost in outer loop
            cost_inner = costfunc(X, y, theta_cpy + s, lambda_r=lambda_r)[0]

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
