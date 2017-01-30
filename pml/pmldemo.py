"""Main"""
import models.perceptron as perceptron
import models.adaline as adaline
from graphs.binary import BinaryClassificationGraph

def demo_perceptron():
    """

    :return:
    """

    # create object class
    classifier = perceptron.Perceptron(eta=0.1, n_iter=10)
    X, y = BinaryClassificationGraph.load_data(source='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                             rows=100,
                             columns=4,
                             features=[0, 2],
                             label='Iris-setosa')
    classifier.graph(X, y, xtitle='setosa', ytitle='versicolor', xlabel='sepal length [cm]', ylabel='petal length [cm]')

def demo_adaline():

    # create object class
    classifier = adaline.AdalineGD(eta=0.01, n_iter=10)
    X, y = BinaryClassificationGraph.load_data(source='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                                          rows=100,
                                          columns=4,
                                          features=[0, 2],
                                          label='Iris-setosa')
    classifier.graph(X, y, xtitle='setosa', ytitle='versicolor', xlabel='sepal length [cm]', ylabel='petal length [cm]')

if __name__ == '__main__':
    """Main function for console application"""

    # plot classification data
    demo_perceptron()
    demo_adaline()

