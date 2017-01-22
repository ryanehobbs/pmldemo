from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_error_chart(X, y, classifier, resolution=0.02, **plot_properties):
    """
    Plot error chart
    :param X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
    :param y: array-like, shape = [n_samples]
             Target values.
    :param classifier:
    :param resolution:
    :param plot_properties:
    :return: Save or display plot using matplotlib
    """

    xlabel = plot_properties.get('xlabel', 'xlabel')
    ylabel = plot_properties.get('ylabel', 'ylabel')
    title = plot_properties.get('title', 'title')
    marker = plot_properties.get('marker', 'o')
    color = plot_properties.get('color', 'red')

    classifier.fit(X, y)
    plt.plot(range(1, len(classifier.errors_) + 1), classifier.errors_, marker=marker, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show(block=True)

def plot_cost_chart(X, y, classifier, resolution=0.02, **plot_properties):

    xlabel = plot_properties.get('xlabel', 'xlabel')
    ylabel = plot_properties.get('ylabel', 'ylabel')
    title = plot_properties.get('title', 'title')
    marker = plot_properties.get('marker', 'o')
    color = plot_properties.get('color', 'red')


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

def plot_binary_data(X, y, classifier_label, **plot_properties):
    """
    Plot the classification data
    :param df:
    :return: Save or display plot using matplotlib
    """

    xlabel = plot_properties.get('xlabel', 'xlabel')
    ylabel = plot_properties.get('ylabel', 'ylabel')
    xtitle = plot_properties.get('xtitle', 'xtitle')
    ytitle = plot_properties.get('ytitle', 'ytitle')
    xmarker = plot_properties.get('xmarker', 'o')
    ymarker = plot_properties.get('ymarker', 'x')
    xcolor = plot_properties.get('xcolor', 'red')
    ycolor = plot_properties.get('ycolor', 'blue')

    y = np.where(y == classifier_label, -1, 1)
    range_end, features = X.shape
    range_start = range_end/features


    # plot the data 1st 50 is setosa and
    plt.scatter(X[:range_start, 0], X[:range_start, 1], color=xcolor, marker=xmarker, label=xtitle)
    plt.scatter(X[range_start:range_end, 0], X[range_start:range_end, 1], color=ycolor, marker=ymarker, label=ytitle)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    return y

def load_data(url, header=None, **properties):
    """
    Use pandas to load CSV data
    :param url_loc: URL of data source
    :param header: Additional headers (optional)
    :return: Pandas DataFrame object
    """

    rows = properties.get('rows', 0)
    columns = properties.get('columns', 0)
    features = properties.get('features', [])

    data_frame = pd.read_csv(url, header=header)

    # set two classes setosa and versicolor get first 100 rows and 4 columns of features
    y = data_frame.iloc[0:rows, columns].values
    # extract sepal length and petal length
    X = data_frame.iloc[0:rows, features].values

    return X, y