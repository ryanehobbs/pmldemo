"""Main"""

import perceptron
import plotter

def demo_perceptron():
    """

    :return:
    """

    # create object class
    classifier = perceptron.Perceptron(eta=0.1, n_iter=10)
    X, y = plotter.load_binary_classification_data(url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                                                   rows=100,
                                                   columns=4,
                                                   features=[0, 2],
                                                   classifier_label='Iris-setosa')

    # plot binary classification data
    plotter.plot_binary_chart(X, y, xtitle='setosa', ytitle='versicolor',
                              xlabel='sepal length [cm]', ylabel='petal length [cm]')
    # plot error chart
    plotter.plot_error_chart(X, y, classifier, xlabel='Epochs', ylabel='Number of misclassifications')
    # plot decision boundary chart
    plotter.plot_decision_regions_chart(X, y, classifier)


if __name__ == '__main__':
    """Main function for console application"""

    # plot classification data
    demo_perceptron()

