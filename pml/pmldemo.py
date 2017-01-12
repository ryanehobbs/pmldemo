from matplotlib import use
from perceptron import Perceptron
use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(url_loc, header=None):

    return pd.read_csv(url_loc, header=header)

def plot_data():

    # read in the data set into a data frame
    df = load_data(url_loc='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

    # set two classes setosa and versicolor get first 100 rows and 4 columns of features
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values

    # plot the data 1st 50 is setosa and
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('/tmp/classes.png', dpi=300)
    plt.show()

    return X, y

def plot_classifications(X, y):

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.savefig('/tmp/error.png', dpi=300)
    plt.show()

if __name__ == '__main__':

    # plot the data
    X, y = plot_data()
    plot_classifications(X, y)

