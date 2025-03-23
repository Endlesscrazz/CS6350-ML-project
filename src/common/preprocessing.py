import numpy as np

def log_transform(X):

    return np.log1p(X)

def standardize_train(X):

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X-mean)/std, mean, std

def standardize_test(X, mean, std):
    return (X - mean)/std
