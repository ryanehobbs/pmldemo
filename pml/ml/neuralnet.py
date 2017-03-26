import models.linear
import numpy as np

class NeuralNetwork(models.linear.Logistic):
    """"""

    def __init__(self, num_of_labels):

        super(NeuralNetwork, self).__init__(num_of_labels=num_of_labels)

    def load(self, filepath):
        """
        If you have pre-initialized neural network parameters (trained),
        you can load it using this method.
        :param filepath: File path of matlab data file
        :return:
        """

        pass

    def predict(self, X, data_layers):
        """
        Outputs the predicted values of X using the trained data contained in the data layers array
        :param X: array-like Array[n_samples, n_features] Training data
        :param data_layers: Array containing a list of theta parameters for the entire neural network.
                            includes input, hidden and output layers.
        :return:
        """

        n_samples = np.size(X, 1)
        p = np.zeros(np.size(X, 1), 1)
        layer_count = len(data_layers)

        # get input layer X = [np.ones(n_samples, 1) X] (a1)

        # iterate over the data layers (a2...)
        for layer in data_layers:
            # calc hypothesis sigmoid(X * Theta1')
            # cur_layer [[np.ones(n_samples, 1)], hypothesis];
            # yield prediction
            pass