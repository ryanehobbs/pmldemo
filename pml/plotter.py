from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.scimath
import pandas as pd


def show_subplots(plots_array):

    ncols=len(plots_array)

    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(8, 4))

    ax[0] = plots_array[0]
    ax[1] = plots_array[1]
    ax[2] = plots_array[2]

    plt.tight_layout()
    plt.show()

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
    plt.show()
    return plt

def plot_cost_chart(X, y, classifier, resolution=0.02, np_mathfunc=None, **plot_properties):

    xlabel = plot_properties.get('xlabel', 'xlabel')
    ylabel = plot_properties.get('ylabel', 'ylabel')
    title = plot_properties.get('title', 'title')
    marker = plot_properties.get('marker', 'o')
    color = plot_properties.get('color', 'red')

    classifier.fit(X, y)


    if np_mathfunc and np_mathfunc in numpy.lib.scimath.__all__:
        # get numpy math operation method
        math_op = getattr(numpy, np_mathfunc)
        plt.plot(range(1, len(classifier.cost_) + 1), math_op(classifier.cost_), marker=marker)
    else:
        plt.plot(range(1, len(classifier.cost_) + 1), classifier.cost_, marker=marker)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    return plt

def plot_decision_regions_chart(X, y, classifier, resolution=0.02):

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

    plt.show()
    return plt

def plot_binary_chart(X, y, **plot_properties):
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

    range_end, features = X.shape
    range_start = int(range_end/features)

    plt.scatter(X[:range_start, 0], X[:range_start, 1], color=xcolor, marker=xmarker, label=xtitle)
    plt.scatter(X[range_start:range_end, 0], X[range_start:range_end, 1], color=ycolor, marker=ymarker, label=ytitle)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.show()
    return plt

def load_binary_classification_data(url, classifier_label, **properties):
    """
    Use pandas to load CSV data
    :param url_loc: URL of data source
    :param header: Additional headers (optional)
    :return: Pandas DataFrame object
    """

    rows = properties.get('rows', 0)
    columns = properties.get('columns', 0)
    features = properties.get('features', [])

    data_frame = pd.read_csv(url)

    # set two classes setosa and versicolor get first 100 rows and 4 columns of features
    y = data_frame.iloc[0:rows, columns].values
    # extract sepal length and petal length
    X = data_frame.iloc[0:rows, features].values

    y = np.where(y == classifier_label, -1, 1)

    return X, y