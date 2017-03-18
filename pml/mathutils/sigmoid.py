import numpy as np
import math

def sigmoid(z):
    """
    Calculate sigmoid function A sigmoid function
    is a mathematical function having an "S"
    shaped curve (sigmoid curve). It is capped
    by -1 and 1 and crosses 0.5.  Typically used
    as an activation function
    :param z: z can be a matrix, vector or scalar
    :return: Value between 0 and 1
    """

    # calculate sigmoid curve g(z) = 1/(1+e^-z)
    return np.divide(1, 1 + np.exp(-z))