import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

def log_transform(X):
    return np.log1p(X)

# Pipeline with log transformation and StandardScaler.
pipeline_log_standard = Pipeline([
    ('log', FunctionTransformer(log_transform)),
    ('scaler', StandardScaler()),
    ('var_thresh', VarianceThreshold(threshold=1e-4)),
    ('pca', PCA(n_components=50, random_state=42)),
])

# Pipeline with log transformation and MinMaxScaler.
pipeline_log_minmax = Pipeline([
    ('log', FunctionTransformer(log_transform)),
    ('scaler', MinMaxScaler()),
    ('var_thresh', VarianceThreshold(threshold=1e-4)),
    ('pca', PCA(n_components=50, random_state=42)),
])

# Pipeline with no log transformation and RobustScaler.
# pipeline_no_log = Pipeline([
#     # No log transformation step.
#     ('scaler', RobustScaler()),
#     ('var_thresh', VarianceThreshold(threshold=1e-4)),
#     ('pca', PCA(n_components=50, random_state=42)),
# ])

pipeline_scaled_only = Pipeline([
  ('scaler', StandardScaler()),
  ('var_thresh', VarianceThreshold(threshold=1e-4)),
])

# Dictionary to hold all pipeline options.
preprocessing_pipelines = {
    "log_standard": pipeline_log_standard,
    "log_minmax": pipeline_log_minmax,
    "scaled_only": pipeline_scaled_only
}

# For backward compatibility, aliasing the default pipeline.
preprocessing_pipeline = pipeline_log_standard
