import matplotlib.pyplot as plt

EPOCHS = 'n_iter'
ALPHA = 'eta'

class Plotter(object):

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

