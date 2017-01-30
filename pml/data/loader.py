import numpy as np
import pandas as pd

def load_data(*args, **kwargs):
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