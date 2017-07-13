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
        # store ndarray of theta gradients
        theta_gradients = {}

        # create new gradient arrays
        for name, theta in nn_params.items():
            theta_gradients[name] = np.zeros((np.size(theta, axis=0), np.size(theta, axis=1)))

        for i in range(0, n_samples):
            # reset for every sample
            nn_params_cpy = copy(nn_params)
            last_key = list(nn_params.keys())[-1]
            prev_layer = None
            output_layer = None
            z_calc = None
            # for every sample get sample data
            a1 = np.reshape(X[i, :].transpose(), (X.shape[1], 1))
            for key in nn_params_cpy.keys():
                if not key == last_key:  # only ff if not last key
                    output_layer, z_calc, prev_layer = self.__ff_calc(a1, nn_params, vstack=True)
                # common back propagation for each output k set 1 or 0 based on coded labels y
                # calculate delta k3
                k_delta = output_layer - ((np.arange(0, self.num_of_labels) ==
                                           (y[i] - 1)).astype(int)).reshape(1, -1, order='F').transpose()
                if not key == last_key:
                    # calc delta of hidden layer theta_two.T * k3 .* sigmoid(z2)
                    k_grad = models.linear.LinearMixin.insert_bias(sigmoid_gradient(z_calc), vstack=True)
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

        # calculate gradients and create new gradient arrays
        for name, theta in nn_params.items():
            # Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
            theta_term = np.hstack((np.zeros((np.size(theta, axis=0), 1)), theta[:,1:]))
            theta_gradients[name] = np.multiply((1/n_samples),
                                                theta_gradients[name]) + np.multiply((lambda_r/n_samples), theta_term)

        return theta_gradients