import data.loader as data
import models.linear
import numpy as np
from mathutils.sigmoid import sigmoid, sigmoid_gradient
from pml import DataTypes
from random import uniform
from types import GeneratorType

def init_weights(inputL, outputL):

    # return a random initialized weights
    return np.zeros((outputL, uniform(0, 1) + inputL))

def numerical_gradient(cost_func, theta):

    # TODO: add a deferred call back for the cost function parameter

    tolg = 1e-4  # tolerance for the gradient check

    n_grad = np.zeros(np.size(theta))
    p_grad = np.zeros(np.size(theta))

    for p in range(0, theta.size):
        #
        p_grad[p] = tolg
        loss1 = cost_func()


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

    def gradient(self):

        # use this to calculate the gradients
        pass

    @models.linear.fitdata
    def cost(self, X, y, theta=None, **kwargs):
        """

        :param nn_params:
        :param lambda_r:
        :param kwargs:
        :return:
        """

        # get number of training samples
        n_samples = y.shape[0]
        # retrieve the nn_params which are the weight samples
        nn_params = kwargs.get("nn_params", None)
        # retrieve the hyper parameter lambda_r for regularization calculation
        lambda_r = kwargs.get("lambda_r", 0)

        if theta:  # if calling base class get theta param if does not exist raise exception
            if theta is not None and np.atleast_1d(theta).ndim > 1:
                raise ValueError("Sample weights must be 1D array or scalar")
            else:
                super(NeuralNetwork, self).cost(X, y, theta, **kwargs)

        def computereg():

            for nn_theta in nn_params.values():
                value = np.sum(np.power(nn_theta[1:], 2))
                yield value

        # re constitute original theta params (weights)
        nn_params = self._roll(nn_params, self.input_size, self.hidden_size, self.param_names)
        # perform feed forward calculations
        y_label, hX = self.ff(X, y, nn_params)
        # calculate the minimized objective cost function for logistic regression
        first_term = (1 / n_samples) * np.sum(np.sum(np.multiply((-1 * y_label), np.log(hX)) - np.multiply((1 - y_label), np.log(1 - hX))))
        # iterate over nn_params removing their bias units before calculating regularized cost
        second_term = (lambda_r / (2 * n_samples) * sum(computereg()))
        # combine terms
        j_cost = first_term + second_term
        # perform back propagation calculations
        theta_gradients = self.bp(X, y, nn_params, lambda_r)
        # Unroll gradients
        grad = self._unroll(theta_gradients)

        return j_cost, grad

    @models.linear.fitdata
    def train(self, X, y, **kwargs):
        """

        :param X:
        :param y:
        :param kwargs:
        :return:
        """

        lambda_r = kwargs.get('lambda_r', 0)
        alpha = kwargs.get("alpha", 0.001)
        iterations = kwargs.get("iterations", self.iterations)

        # ensure alpha (learning rate) conforms to 0.001 < alpha < 10
        if models.linear.ALPHA_MIN < alpha < models.linear.ALPHA_MAX:
            alpha = alpha
        else:
            print(
                "Learning rate (alpha) does not fit within range 0.001 < alpha < 10 defaulting to 0.01")
            alpha = 0.01

        return None

    def ff(self, X, y, nn_params):
        """

        :return:
        """

        # TODO: add looping layer for count of layers to process all hidden layers
        # TODO: For now (testing) just process manually and improve later

        # get size of training data
        n_samples = np.size(X, axis=0)

        if not isinstance(nn_params, dict):  # needs to be rolled
            # re constitute original theta params (weights)
            nn_params = self._roll(nn_params, self.input_size, self.hidden_size, self.param_names)
        layers = len(nn_params)

        # input layer (a1)
        a1 = X  # column of ones bias already added when fitted
        z2 = np.dot(a1, nn_params['Theta1'].transpose())
        # FF -> Process layer 2 (hidden)
        a2 = sigmoid(z2)
        # add bias to second layer a2
        bias_ = np.ones((np.size(a2, 0), 1))
        a2 = np.insert(a2, 0, bias_.T, axis=1)
        z3 = np.dot(a2, nn_params['Theta2'].transpose())
        # FF -> Process layer 3 (output)
        a3 = sigmoid(z3)

        # recode y labels
        y_label = self._recode_labels(y, n_samples, self.num_of_labels)

        return (y_label, a3)  # FIXME: this will be replaced with a deterministic value calculated via loop

    def bp(self, X, y, nn_params, lambda_r):
        """

        :return:
        """

        # TODO: add looping layer for count of layers to process all hidden layers
        # TODO: For now (testing) just process manually and improve later

        # get size of training data
        n_samples = np.size(X, axis=0)
        if not isinstance(nn_params, dict):  # needs to be rolled
            # re constitute original theta params (weights)
            nn_params = self._roll(nn_params, self.input_size, self.hidden_size, self.param_names)
        # store ndarray of theta gradients
        theta_gradients = {}
        layers = len(nn_params)

        # create new gradient arrays
        for name, theta in nn_params.items():
            theta_gradients[name] = np.zeros((np.size(theta, axis=0), np.size(theta, axis=1)))

        for i in range(0, n_samples):
            # perform a feed forward calculation set input layers a^1 to t-th training x^t
            a1 = np.reshape(X[i,:].transpose(), (X.shape[1], 1))
            z2 = np.dot(nn_params['Theta1'], a1)  # FIXME: current fixed values will need to be refactored
            # add bias to second layer a2
            bias_ = np.ones((1, 1))
            # compute sigmoid of z3 and add bias row
            a2 = np.vstack((bias_, np.reshape(sigmoid(z2), (z2.shape[0], 1))))
            # output layer 3 this is odd why .T is not needed
            z3 = np.dot(nn_params['Theta2'], a2)  # FIXME: current fixed values will need to be refactored
            # compute sigmoid of z3
            a3 = sigmoid(z3)

            # perform back propagation for each output k set 1 or 0 based on coded labels y
            # calculate delta k3
            # k3 = a3 - ([1:num_labels]==y(t))';
            k3 = a3 - ((np.arange(0, self.num_of_labels) == (y[i] - 1)).astype(int)).reshape(1,-1, order='F').transpose()
            # add bias to second layer a2
            bias_ = np.ones((1, 1))
            # calc delta of hidden layer theta_two.T * k3 .* sigmoid(z2)
            k2_grad = np.vstack((bias_, np.reshape(sigmoid_gradient(z2), (z2.shape[0], 1))))
            k2 = np.multiply(np.dot(nn_params['Theta2'].T, k3), k2_grad)  # FIXME: current fixed values will need to be refactored
            # accumulate the gradients for all deltas
            k2 = k2[1:]
            # calculate running sum of gradients for weight parameter  # FIXME: current fixed values will need to be refactored
            theta_gradients['Theta1'] = theta_gradients['Theta1'] + np.multiply(k2, a1.T)
            # calculate running sum of gradients for weight parameter
            theta_gradients['Theta2'] = theta_gradients['Theta2'] + np.multiply(k3, a2.T)

        # calculate gradients and create new gradient arrays
        for name, theta in nn_params.items():
            # Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
            theta_term = np.hstack((np.zeros((np.size(theta, axis=0), 1)), theta[:,1:]))
            theta_gradients[name] = np.multiply((1/n_samples), theta_gradients[name]) + np.multiply((lambda_r/n_samples), theta_term)

        return theta_gradients

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
        if isinstance(nn_params, GeneratorType):
            for param in nn_params:
                theta_params = np.append(theta_params, param[1])
        elif isinstance(nn_params, dict):
            for param in nn_params.values():
                theta_params = np.append(theta_params, param)

        return np.reshape(theta_params, (theta_params.shape[0], -1))

    def _recode_labels(self, y, sample_size, num_of_labels):

        # create a y label vector for recoding
        y_label = np.zeros((sample_size, num_of_labels))

        # iterate over sample sizes recoding label either 0 or 1
        for x in range(0, sample_size):
            y_label[x, (y[x]-1)] = 1

        return y_label