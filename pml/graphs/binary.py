#from models.classifier import Classifier
from graphs.plotter import Plotter, ALPHA, EPOCHS
import matplotlib.pyplot as plt
import numpy as np

class BinaryClassificationGraph(Plotter):

    def __init__(self):

        super(BinaryClassificationGraph, self).__init__()

    # kw_costs = {classifier:[epochs, learning rate], ...}
    def cost_old(self, X, y, classifier, model_properties=None, **properties):
        """

        :param X:
        :param y:
        :param classifiers:
        :param properties:
        :return:
        """

        if not isinstance(model_properties, dict):
            model_properties = {EPOCHS:0, ALPHA:[]}

        xlabel = properties.get('xlabel', 'xlabel')
        ylabel = properties.get('ylabel', 'ylabel')
        title = properties.get('title', 'title')
        marker = properties.get('marker', 'o')
        color = properties.get('color', 'red')
        refit = properties.get('refit', False)

        if model_properties[EPOCHS] in (None, [], "", 0):
            model_properties[EPOCHS] = classifier.n_iter
        if model_properties[ALPHA] in (None, [], "", 0):
            model_properties[ALPHA] = [classifier.eta]

        eta = model_properties[ALPHA]
        n_iter = model_properties[EPOCHS]
        alpha_count = len(model_properties[ALPHA])

        if alpha_count > 1:
            fig, ax = plt.subplots(nrows=1, ncols=alpha_count, figsize=(8, 4))
            for ax_idx in range(0, alpha_count):
                classifier.fit(X, y, n_iter=n_iter, eta=eta[ax_idx]) # always refit
                ax[ax_idx].plot(range(1, len(classifier.cost_) + 1), np.log10(classifier.cost_), marker=marker)
                ax[ax_idx].set_xlabel(xlabel)
                ax[ax_idx].set_ylabel(ylabel)
                ax[ax_idx].set_title(title)
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
            if (classifier.fitted and refit) or (not classifier.fitted):
                classifier.fit(X, y, n_iter=n_iter, eta=eta[0])  # fit only when necessary
            ax.plot(range(1, len(classifier.cost_) + 1), classifier.cost_, marker=marker, color=color)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)

        return ax


