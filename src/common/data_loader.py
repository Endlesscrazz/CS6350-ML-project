import pandas as pd
import numpy as np

def load_data(filename, label_column='label'):
    """
    Loads data from a csv file

    Parameters:
    filename(str): Path for the csv file
    label_column(str): Name of column containing labels
    """

    df = pd.read_csv(filename)
    
    # Check if label_column exists in the data
    if label_column is not None and label_column in df.columns:
        X = df.drop(columns=label_column).values
        y = df[label_column].values
        return X, y
    else:
        return df.values

# print("training data shape:",X_train.shape)
# print("test data shape:",X_test.shape)