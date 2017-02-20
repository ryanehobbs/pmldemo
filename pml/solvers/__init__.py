import numpy as np

def doglegm(r, g,d, delta):

    b = np.linalg.solve(r.T, g)
    x = np.linalg.solve(r, b)
    xn = np.linalg.norm(np.multiply(d, x))

    if (xn > delta):
        # GN too big must scale
        s = np.divide(g, d)
        sn = np.linalg.norm(s)
        if (sn > 0):
            # normalize and rescale
            s = np.divide((s / sn), d)
            # get line minimzer in s direction
            tn = np.linalg.norm(r*s)
            snm = (sn / tn) / tn
            if (snm < delta):
                # get the dogleg path minimizer
                bn = np.linalg.norm(b)
                dxn = delta / xn
                snmd = snm / delta
                t = (bn / sn) * (bn / xn) * snmd
                t -= dxn * np.power(snmd, 2) - np.sqrt(np.power((t-dxn), 2) + (1 - np.power(dxn, 2)) * (1 - np.power(snmd, 2)))
                alpha = dxn * (1 - np.power(snmd, 2)) / t
            else:
                alpha = 0
        else:
            alpha = delta / xn
            snm = 0;
        # form convex combination
        x = alpha * x + ((1 - alpha) * min(snm, delta)) * s
    return x

# https://en.wikipedia.org/wiki/Cholesky_decomposition
def cholupdate(R,x,sign):
    p = np.size(x)

    #x = x.T

    for k in range(p):
        if sign == '+':
            inner = np.sum(np.power(R[k,k], 2) + (np.sum(np.power(x[k], 2))))
            r = np.sqrt(inner)
        elif sign == '-':
            inner = np.sum(np.power(R[k,k], 2) - (np.sum(np.power(x[k], 2))))
            r = np.sqrt(inner)
        #c = r/R[k,k]
        c = np.divide(r, R[k,k])
        #s = np.sum(x[k]/R[k,k])
        s = np.sum(np.divide(x[k], R[k,k]))
        R[k,k] = r
        if sign == '+':
            R[k,k+1:p] = (R[k,k+1:p] + s * x[k+1:p].T)/c
        elif sign == '-':
            R[k,k+1:p] = (R[k,k+1:p] - s * x[k+1:p].T)/c
        x[k+1:p]= c * x[k+1:p] - s * R[k, k+1:p].T
    return R
