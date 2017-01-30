from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPOCHS = 'n_iter'
ALPHA = 'eta'

class Plotter(object):

    @classmethod
    def load_data(cls, *args, **kwargs):
        """
        Use pandas to load CSV data.
        Return tuple containing Matrix X and vector y
        :param args:
        :param kwargs:
        :return:
        """

        if 0 <= 2 < len(args):
            source = args[0]  # Source URL or file
            label = args[1]  # classifier label (y)
        else:
            source = kwargs.get('source')
            label = kwargs.get('label')

        rows = kwargs.get('rows', 0)
        columns = kwargs.get('columns', 0)
        features = kwargs.get('features', [])

        data_frame = pd.read_csv(source)

        # set two classes setosa and versicolor get first 100 rows and 4 columns of features
        y = data_frame.iloc[0:rows, columns].values
        # extract sepal length and petal length
        X = data_frame.iloc[0:rows, features].values

        y = np.where(y == label, -1, 1)

        return X, y

    def binary_scatterplot(self, matrix, **properties):
        """

        :param matrix:
        :param label:
        :param properties:
        :return:
        """

        xlabel = properties.get('xlabel', 'xlabel')
        ylabel = properties.get('ylabel', 'ylabel')
        xtitle = properties.get('xtitle', 'xtitle')
        ytitle = properties.get('ytitle', 'ytitle')
        xmarker = properties.get('xmarker', 'o')
        ymarker = properties.get('ymarker', 'x')
        xcolor = properties.get('xcolor', 'red')
        ycolor = properties.get('ycolor', 'blue')

        range_end, features = matrix.shape
        range_start = int(range_end/features)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
        ax.scatter(matrix[:range_start, 0], matrix[:range_start, 1],
                    color=xcolor, marker=xmarker, label=xtitle)
        ax.scatter(matrix[range_start:range_end, 0], matrix[range_start:range_end, 1],
                    color=ycolor, marker=ymarker, label=ytitle)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper left')


    def decision_regions(self, X, y, classifier, resolution=0.02, **properties):

        refit = properties.get('refit', False)
        if (classifier.fitted and refit) or (not classifier.fitted):
            classifier.fit(X, y)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        meshX = np.arange(x1_min, x1_max, resolution)
        meshY = np.arange(x2_min, x2_max, resolution)
        xx1, xx2 = np.meshgrid(meshX, meshY)

        # flatten
        flatX = xx1.ravel()
        flatY = xx2.ravel()
        flattend_array = np.array([flatX, flatY]).T

        Z = classifier.predict(flattend_array)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=cl)


