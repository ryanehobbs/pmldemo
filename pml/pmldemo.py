"""Main"""

import data.loader as data
import preprocessing
import models.linear as linear
import ml.neuralnet as nn
import pml


def demo_preprocessing():

    X, y = data.load_data(source='samples/sample_data4.txt', columns=3, features=[1, 2])
    preprocessor = preprocessing.PolyFeatures(degrees=2)
    preprocessor.mapfeatures(X)
    print("done")

def demo_newmultilinear():

    X, y = data.load_data(source='samples/sample_data2.txt', columns=3, features=[1,2])
    linear_model = linear.Linear(solver='linear', normalize=True, max_iter=1500)
    linear_model.train(X, y, alpha=0.01)
    price1 = linear_model.predict([1650, 3])
    print("A 1650 sq. foot home with 3 bedrooms, we predict its price  to be ${}".format(price1))
    price2 = linear_model.predict([4000, 4])
    print("A 4000 sq. foot home with 4 bedrooms, we predict its price  to be ${}".format(price2))


def demo_newlinear():

    X, y = data.load_data(source='samples/sample_data.txt', columns=2, features=1)
    linear_model = linear.Linear(solver='linear', normalize=False, max_iter=1500)
    linear_model.train(X, y, alpha=0.01)
    price1 = linear_model.predict([3.5])*10000
    print("For a population of 35,000 people, we predict a profit of ${}".format(price1))
    price2 = linear_model.predict([7])*10000
    print("For a population of 70,000 people, we predict a profit of ${}".format(price2))

def demo_newlogistic():

    X, y = data.load_data(source='samples/sample_data3.txt', columns=3, features=[1,2])
    logistic_model = linear.Logistic(solver='logistic', normalize=False, max_iter=400)
    logistic_model.train(X, y)
    result = logistic_model.predict([45, 85], sum=True)
    print("For a student with scores 45 and 85, we predict an admission probability of {0:.2f}%".format(result*100))

def demo_logisticreg():

    import numpy as np
    X, y = data.load_data(source='samples/sample_data4.txt', columns=3, features=[1,2])
    logistic_model = linear.Logistic(solver='logistic', normalize=False, max_iter=1500, lambda_r=1)
    # map polynomial features
    preprocessor = preprocessing.PolyFeatures(degrees=6)
    X = preprocessor.mapfeatures(X)
    logistic_model.train(X, y, lambda_r=1, iterations=400)
    pred = logistic_model.predict()
    print('Train Accuracy: {}'.format(np.mean(np.double(pred == y)) * 100))

def demo_multiclass():

    import numpy as np
    X, y = data.load_matdata(file_name='samples/ex3data1.mat')
    logistic_model = linear.Logistic(solver='logistic', normalize=False, num_of_labels=10, max_iter=50, lambda_r=3)

    y = preprocessing.PolyFeatures.zeroalign(y) # <---- original data mapped based on 1-based index not 0 based
    logistic_model.train(X, y, lambda_r=0.1)

    pred = logistic_model.predict(X)
    print('Training Set Accuracy: {}%'.format(np.mean(np.double(pred == y)) * 100))

    # FIXME: Need to have a common way of calling into cost calc and fit and do the right thing
    # should this be class vars or should they be defined at the methods themselves
    theta_t = [-2, -1, 1, 2]
    X_t = np.arange(1,16).reshape(5,-1, order='F')/10
    y_t = np.array([[1],[0],[1],[0],[1]])
    J, grad = logistic_model.cost(X_t, y_t, theta_t, lambda_r=3, refit=True)
    print('Cost: {}'.format(J))
    print('Expected cost: 2.534819')
    print('Gradients: {}'.format(grad))
    print('Expected gradients: {}'.format('0.146561\n -0.548558\n 0.724722\n 1.398003\n'))

def demo_neuralnetwork():

    X, y = data.load_matdata(file_name='samples/ex4data1.mat')
    neuralnet = nn.NeuralNetwork(num_of_labels=10, input_layer_size=400, hidden_layer_size=25, data_source='samples/ex4weights.mat', data_type=pml.DataTypes.MATLAB)
    neuralnet.cost(X, y, nn_params=neuralnet.data)

    print("done")

if __name__ == '__main__':
    """Main function for console application"""

    # plot classification data
    #demo_preprocessing()
    #demo_newmultilinear()
    #demo_newlinear()
    #demo_newlogistic()
    #demo_logisticreg()
    #demo_multiclass()
    demo_neuralnetwork()


