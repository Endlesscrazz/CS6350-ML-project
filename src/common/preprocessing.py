import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, TruncatedSVD

def log_transform(X):

    return np.log1p(X)

def standardize_train(X):

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X-mean)/std, mean, std

def standardize_test(X, mean, std):
    return (X - mean)/std

def remove_low_variance_features(X_train, X_test, threshold=1e-4):
    """
    Function to remove features below certain threshold
    """

    selector = VarianceThreshold(threshold=threshold)
    X_train_reduced = selector.fit_transform(X_train)
    X_test_reduced = selector.transform(X_test)

    return X_train_reduced, X_test_reduced, selector

def apply_pca(X_train, X_test, n_components=50):
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca, pca

def apply_truncated_svd(X_train, X_test, n_components=50):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_svd = svd.fit_transform(X_train)
    X_test_svd = svd.transform(X_test)

    return X_train_svd, X_test_svd, svd