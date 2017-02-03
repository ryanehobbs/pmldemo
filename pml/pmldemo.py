"""Main"""

import models.perceptron as perceptron
import models.adaline as adaline
import data.loader as data

def demo_linear():

    X, y = data.load_data(source='samples/sample_data.txt', columns=2, features=1)
    classifier = perceptron.Perceptron(eta=0.1, n_iter=10)
    cost = classifier.linear(X, y, None)

    print('done cost={}'.format(cost))

def demo_multilinear():

    X, y = data.load_data(source='samples/sample_data2.txt', columns=3, features=[1,2])
    classifier = perceptron.Perceptron(eta=0.1, n_iter=10)
    cost = classifier.multi_linear(X, y, [0,0,0])

    print('done cost={}'.format(cost))

def demo_perceptron():
    """

    :return:
    """

    # create object class
    classifier = perceptron.Perceptron(eta=0.1, n_iter=10)
    X, y = data.load_data(source='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                             rows=100,
                             columns=1,
                             features=[0, 1],
                             label='Iris-setosa')
    cost = classifier.linear(X, y, None)
    print('done')

def demo_adaline():

    # create object class
    classifier = adaline.AdalineGD(eta=0.01, n_iter=10)
    X, y = data.load_data(source='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                                          rows=100,
                                          columns=4,
                                          features=[0, 2],
                                          label='Iris-setosa')
    classifier.graph(X, xtitle='setosa', ytitle='versicolor', xlabel='sepal length [cm]', ylabel='petal length [cm]')

if __name__ == '__main__':
    """Main function for console application"""

    # plot classification data
    #demo_perceptron()
    #demo_adaline()
    demo_linear()
    demo_multilinear()

