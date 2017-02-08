import numpy as np

DEGREE=6

def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features
    maps two input features to quadratic features
    used in regularization calculations.

    Returns a new feature array with more features,
    comprising of  X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Inputs X1, X2 must be the same size
    :param X1: n x n+1 dimensional array
    :param X2: n x n+1 dimensional array
    :return: Feature array
    """

    ones = np.ones((len(X1), 1))
    for i in range(1, DEGREE+1):
        for j in range(0, i+1):
            calc = np.multiply(np.power(X1, (i-j)), np.power(X2, j))
            ones = np.insert(ones, ones.shape[1],  calc, axis=1)
    # index and pull all rows 2nd column
    #blah = ones[:,1]
    #print(np.array_str(ones, precision=2, suppress_small=True))
    return ones
