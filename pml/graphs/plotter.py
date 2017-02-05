import matplotlib.pyplot as plt
import random as ran
from mpl_toolkits.mplot3d import Axes3D

EPOCHS = 'n_iter'
ALPHA = 'eta'

class Plotter(object):

    def scatterplot(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """

        cmhot = plt.cm.get_cmap("hot")
        l = plt.scatter(X, y, c=y, cmap=cmhot)
        plt.colorbar(l)

    def lineplot(self, X, y):

        plt.cm.get_cmap("hot")
        plt.plot(X,y)

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

