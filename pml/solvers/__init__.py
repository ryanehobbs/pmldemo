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
