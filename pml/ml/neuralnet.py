import data.loader as data
import models.linear
import numpy as np
from pml import DataTypes
from functools import wraps


class NeuralNetwork(models.linear.Logistic):
    """"""

    def __init__(self, num_of_labels, input_layer_size, hidden_layer_size, data_source=None, **kwargs):

        if data_source:
            data_type = kwargs.get('data_type', 'matlab')
            self.param_names, data = self.load(data_source=data_source, data_type=data_type)

        # unroll data
        self.data = self._unroll(data)
        self.input_size = input_layer_size
        self.hidden_size = hidden_layer_size

        super(NeuralNetwork, self).__init__(num_of_labels=num_of_labels)

    def load(self, data_source, data_type='matlab'):
        """
        If you have pre-initialized neural network parameters (trained),
        you can load it using this method.
        :param data_source:
        :param data_type:
        :return:
        """

        if isinstance(data_type, DataTypes) or data_type.lower() == 'matlab':
            return data.load_matdata(file_name=data_source, keys_only=True)

    def cost(self, X, y, **kwargs):
        """

        :param nn_params:
        :param lambda_r:
        :param kwargs:
        :return:
        """

        lambda_r = kwargs.get('lambda_r', 0)

        # perform feedforward calculations
        self.ff(X, y, self.data)

        # unroll (temp test)
        #single_vector = self._unroll(nn_params)
        # theta neural net params should be "rolled" (converted back into the weight matrices)
        #nn_params = self._roll(single_vector, self.input_size, self.hidden_size, self.param_names)
        # when returning theta should be "unrolled" vector of the partial derivatives of the neural network.
        pass

    def ff(self, X, y, nn_params):
        """

        :return:
        """

        # re constitute original theta params (weights)
        nn_params = self._roll(nn_params, self.input_size, self.hidden_size, self.param_names)
        layers = len(nn_params)

        # input layer (a1)
        a1 = X  # column of ones bias already added when fitted
        z2 = a1 * nn_params['Theta1']

        # TODO: add looping layer for count of layers to process all hidden layers
        # TODO: For now (testing) just process manually and improve later


        # FF -> Process layer 2 (hidden)
        #a2 = sigmoid(z2);
        # add bias to second layer a2
        #a2 = [ones(size(a2, 1), 1) a2];

        pass

    def bp(self):
        """

        :return:
        """

        pass

    def predict(self):
        """

        :return:
        """

        pass

    def _roll(self, single_vector, input_size, hidden_size, data_keys):
        """
        Reshape nn_params back into their parameters, the weight matrices
        for our neural network

        :return:
        """

        nn_params = {}
        shape = hidden_size * (input_size + 1)
        index = 0
        cur_idx = 0
        length = len(data_keys)

        # FIXME:  Need to be able to iterate over single vector and reconstitute the Thetas
        # iterate over unrolled nn_params and reshape them back to weight parameters to be used for processing
        for key in data_keys:
            cur_idx += 1
            if cur_idx == length:  # reached the end
                nn_params[key] = np.reshape(single_vector[index:], (self.num_of_labels, hidden_size + 1))
            else:  # iterate and put together data FIXME: This may not work need to eval with greater than two data_keys
                nn_params[key] = np.reshape(single_vector[index:shape], (-1, input_size + 1))  # -1 may have to be changed to a dynamic value
                index = (hidden_size * (input_size + 1))

        return nn_params

    def _unroll(self, nn_params):
        """

        :return:
        """

        theta_params = np.array([])

        # iterate over nn_params and unroll them
        # FIXME: this may not be efficient ravel may be a method to use
        for param in nn_params:
            theta_params = np.append(theta_params, param[1])

        return np.reshape(theta_params, (theta_params.shape[0], -1))