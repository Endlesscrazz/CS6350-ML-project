import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA

def log_transform(X):

    return np.log1p(X)

preprocessing_pipeline = Pipeline([
    ('log', FunctionTransformer(log_transform)),
    ('scaler', StandardScaler()),
    ('var_thresh', VarianceThreshold(threshold=1e-4)),
    ('pca', PCA(n_components=50, random_state=42)),
    
    #('kcpa', KernelPCA(n_components=50, kernel='rbf', gamma=0.1, random_state=42))

])