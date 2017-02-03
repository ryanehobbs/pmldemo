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

    if 0 <= 1 < len(args):
        source = args[0]  # Source URL or file
    else:
        source = kwargs.get('source')
        columns = kwargs.get('columns', None)
        features = kwargs.get('features', None)
        headers = kwargs.get('headers', None)

    data_frame = pd.read_csv(source, header=headers)
    # get dimensions of loaded data
    df_rows, df_columns = data_frame.shape

    # get start and end col ranges
    if not isinstance(columns, list) and isinstance(columns, int):
        if columns == 0: # always set to 1 if 0
            columns = 1
        # set start and end column to isolate just that colume
        col_start, col_end = abs((df_columns-columns) - 1), columns
    elif isinstance(columns, list) and (len(columns) <= 2 or len(columns) > 0):
        if len(columns) == 1:  # default remaining columns if no end specified
            columns.append(df_columns)
        col_start, col_end = columns
        # if start is 0 keep it 0
        col_start = 0 if col_start == 0 else abs(((df_columns-col_start)-1) - 1)
    elif columns is None:
        col_start, col_end = (0, df_columns)
    else:
        raise Exception("Invalid value for columns parameter")

    if col_end > df_columns:
        raise Exception('Column count exceeds data column count')

    if not isinstance(features, list) and isinstance(features, int):
        # set start and end column to isolate just that colume
        feature_start, feature_end = abs((df_columns-features) - 1), df_columns-1
    elif isinstance(features, list):
        if len(features) == 1:  # default remaining columns if no end specified
            features.append(df_columns-1)
        feature_start, feature_end = features
        feature_start, feature_end = abs(((df_columns-feature_start)-1) - 1), feature_end
    elif features is None:
        feature_start, feature_end = (0, df_columns-1)
    else:
        raise Exception("Invalid value for features parameter")


    y = data_frame.iloc[0:df_rows, col_start:col_end].values
    X = data_frame.iloc[0:df_rows, feature_start:feature_end].values

    #y = np.where(y == label, -1, 1)

    return X, y