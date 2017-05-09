import collections
import data.loader as data
import models.linear
import numpy as np
from copy import copy
from mathutils.sigmoid import sigmoid, sigmoid_gradient
from pml import DataTypes, DeferredExec
from random import uniform
from solvers import fmincg
from types import GeneratorType


def init_weights(inputL, outputL):
    """
    Initialize random weight values based on sample inputs and outputs
    :param inputL:
    :param outputL:
    :return:
    """

    # return a random initialized weights
    return np.zeros((outputL, uniform(0, 1) + inputL))

def numerical_gradient(cost_func, theta):
    """

    :param cost_func:
    :param theta:
    :return:
    """

    # TODO: add a deferred call back for the cost function parameter

    tolg = 1e-4  # tolerance for the gradient check

    n_grad = np.zeros(np.size(theta))
    p_grad = np.zeros(np.size(theta))

    for p in range(0, theta.size):
        #
        p_grad[p] = tolg
        loss1 = cost_func()


class NeuralNetwork(models.linear.Logistic):
    """NeuralNetwork class """

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
        """

        :return:
        """

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
        nn_params = theta if theta is not None else kwargs.get("nn_params", None)
        # retrieve the hyper parameter lambda_r for regularization calculation
        lambda_r = kwargs.get("lambda_r", 0)

        #if theta:  # if calling base class get theta param if does not exist raise exception
        #if theta and not np.atleast_1d(theta).ndim < 1:
        #    raise ValueError("Sample weights must be 1D array or scalar")
        #elif theta and np.atleast_1d(theta).ndim > 1:
        #    super(NeuralNetwork, self).cost(X, y, theta, **kwargs)

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

        # Call fmincg passing in delagate that will be used in processing
        #cost_func = DeferredExec(self.cost, X, y, nn_params=self.data, lambda_r=0)
        fmincg(self.cost, X, y, self.data)



        return None

    def ff(self, X, y, nn_params):
        """
        Execute feed forward propagation on neural network model. Returning recoded
        labeled data and neural network output layer
        :return:
        """

        # get size of training data
        n_samples = np.size(X, axis=0)
        # perform feed forward propagation
        output_layer, _ , _ = self._ff_calc(X, nn_params, transpose=True)
        # recode y labels
        y_label = self._recode_labels(y, n_samples, self.num_of_labels)

        return y_label, output_layer

    def _ff_calc(self, X, nn_params, transpose=False, vstack=False):
        """
        Perform feed forward calculation across all layers of a neural
        network.  Uses a sigmoid function to calculate z_calc

        :param X:
        :param nn_params:
        :return:
        """

        if not isinstance(nn_params, dict):  # needs to be rolled
            # re constitute original theta params (weights)
            nn_params = self._roll(nn_params, self.input_size, self.hidden_size, self.param_names)

        last_key = list(nn_params.keys())[-1]
        output_layer = X
        prev_z_calc = None
        prev_layer = None
        # process each layer of the feed forward neural network
        for key in nn_params.keys():
            if transpose:  # calc dot product of input layer and nn_param key
                z_calc = np.dot(output_layer, nn_params[key].transpose())
            else:
                z_calc = np.dot(nn_params[key], output_layer)
            # sigmoid activation of z_calc
            input_layer = sigmoid(z_calc)
            if key != last_key:   # check if last layer, if not add bias and continue.
                if vstack:  # insert bias veritcally
                    input_layer = models.linear.LinearMixin.insert_bias(input_layer, vstack=True)
                else:  # insert bias horizontally
                    input_layer = models.linear.LinearMixin.insert_bias(input_layer, axis=1)
                prev_layer = input_layer
                prev_z_calc = z_calc  # store previous layers product calc
            # move to next layer
            output_layer = input_layer

        return output_layer, prev_z_calc, prev_layer

    def _bp_calc(self, X, y, nn_params, transpose=False, vstack=False, iterable=False):

        # get size of training data
        n_samples = np.size(X, axis=0)

        # store ndarray of theta gradients
        theta_gradients = {}

        # create new gradient arrays
        for name, theta in nn_params.items():
            theta_gradients[name] = np.zeros((np.size(theta, axis=0), np.size(theta, axis=1)))

        # common back propagation for each output k set 1 or 0 based on coded labels y
        for i in range(0, n_samples):
            # reset for every sample
            nn_params_cpy = copy(nn_params)
            last_key = list(nn_params.keys())[-1]
            prev_layer = None
            output_layer = None
            z_calc = None
            # for every sample get sample data
            a1 = np.reshape(X[i, :].transpose(), (X.shape[1], 1))
            if iterable:
                for name, theta_gradient in self.__bp_calciter__(a1, y, nn_params_cpy, i, transpose, vstack):
                    theta_gradients[name] = theta_gradients[name] + theta_gradient
            else:
                for key in nn_params_cpy.keys():
                    if not key == last_key:  # only ff if not last key
                        output_layer, z_calc, prev_layer = self._ff_calc(a1, nn_params, vstack=vstack)
                    # calculate delta k3
                    if transpose:
                        k_delta = output_layer - ((np.arange(0, self.num_of_labels) ==
                                               (y[i] - 1)).astype(int)).reshape(1, -1, order='F').transpose()
                    else:
                        k_delta = output_layer - ((np.arange(0, self.num_of_labels) ==
                                                   (y[i] - 1)).astype(int)).reshape(1, -1, order='F')
                    if not key == last_key:
                        # calc delta of hidden layer theta_two.T * k3 .* sigmoid(z2)
                        k_grad = models.linear.LinearMixin.insert_bias(sigmoid_gradient(z_calc), vstack=vstack)
                        # accumulate the gradients for all deltas
                        k2 = np.multiply(np.dot(nn_params[last_key].T, k_delta), k_grad)[1:]
                        # calculate running sum of gradients for weight parameter
                        theta_gradients[key] = theta_gradients[key] + np.multiply(k2, a1.T)
                    else:
                        # calculate running sum of gradients for weight parameter
                        theta_gradients[last_key] = theta_gradients[last_key] + np.multiply(k_delta, prev_layer.T)
                        # delete this layers last key
                        nn_params_cpy.popitem(last=True)
                        # set new last key
                        last_key = list(nn_params_cpy.keys())[-1]

        return theta_gradients

    def __bp_calciter__(self, X, y, nn_params, label_index, transpose=False, vstack=False):

        last_key = list(nn_params.keys())[-1]
        output_layer, z_calc, prev_layer = self._ff_calc(X, nn_params, vstack=vstack)

        # common back propagation for each output k set 1 or 0 based on coded labels y
        for key in nn_params.keys():
            # calculate delta k3
            if transpose:
                k_delta = output_layer - ((np.arange(0, self.num_of_labels) ==
                                           (y[label_index] - 1)).astype(int)).reshape(1, -1, order='F').transpose()
            else:
                k_delta = output_layer - ((np.arange(0, self.num_of_labels) ==
                                           (y[label_index] - 1)).astype(int)).reshape(1, -1, order='F')
            if not key == last_key:
                # calc delta of hidden layer theta_two.T * k3 .* sigmoid(z2)
                k_grad = models.linear.LinearMixin.insert_bias(sigmoid_gradient(z_calc), vstack=vstack)
                # accumulate the gradients for all deltas
                k2 = np.multiply(np.dot(nn_params[last_key].T, k_delta), k_grad)[1:]
                # calculate running sum of gradients for weight parameter
                result = np.multiply(k2, X.T)
            else:
                # calculate running sum of gradients for weight parameter
                result = np.multiply(k_delta, prev_layer.T)
                # delete this layers last key
                nn_params.popitem(last=True)
                # set new last key
                last_key = list(nn_params.keys())[-1]

            yield key, result


    def bp(self, X, y, nn_params, lambda_r):
        """
        Execute back propagation on neural network model. Returning the weight gradients for
        the neural network parameters
        :return:
        """

        # get size of training data
        n_samples = np.size(X, axis=0)

        if not isinstance(nn_params, dict):  # needs to be rolled
            # re constitute original theta params (weights)
            nn_params = self._roll(nn_params, self.input_size, self.hidden_size, self.param_names)

        #
        theta_gradients = self._bp_calc(X, y, nn_params, transpose=True, vstack=True, iterable=True)

        # calculate gradients and create new gradient arrays
        for name, theta in nn_params.items():
            # Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
            theta_term = np.hstack((np.zeros((np.size(theta, axis=0), 1)), theta[:,1:]))
            theta_gradients[name] = np.multiply((1/n_samples),
                                                theta_gradients[name]) + np.multiply((lambda_r/n_samples), theta_term)

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

        nn_params = collections.OrderedDict()
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
        Reshape nn_params into a single vector
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
        """
        Recode labeled data to either a 0 or 1 and returning a
        vector of the recoded values
        :param y:
        :param sample_size:
        :param num_of_labels:
        :return:
        """

        # create a y label vector for recoding
        y_label = np.zeros((sample_size, num_of_labels))

        # iterate over sample sizes recoding label either 0 or 1
        for x in range(0, sample_size):
            y_label[x, (y[x]-1)] = 1

        return y_label