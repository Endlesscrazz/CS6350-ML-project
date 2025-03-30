import numpy as np

class DecisionStump:
    """
    A simple decision stump used as a weak learner in AdaBoost.
    It finds the best feature and threshold that minimizes the weighted error.
    """
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.aplha = None

    def train(self, X, y, sample_weights):
        m,n = X.shape
        min_error = float('inf')
        for feature_i in range(n):
            thresholds = np.unique(X[:, feature_i])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(m)
                    predictions[X[:, feature_i] * polarity < threshold * polarity] = -1
                    error = np.sum(sample_weights[y != predictions])
                    if error < min_error:
                        min_error = error
                        self.feature_index = feature_i
                        self.threshold = threshold
                        self.polarity = polarity
        return min_error

    def predict(self, X):
        m = X.shape[0]
        predictions = np.ones(m)
        predictions[X[:, self.feature_index] * self.polarity < self.threshold * self.polarity] = -1
        return predictions
    
class AdaBoostModel:
    """
    AdaBoost ensemble using decision stumps as weak learners.
    """
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def train(self, X, y):
        m, _ = X.shape
        sample_weights = np.full(m, 1.0 / m)
        self.stumps = []
        self.alphas = []
        for _ in range(self.n_estimators):
            stump = DecisionStump()
            error = stump.train(X, y, sample_weights)
            # Avoid division by zero
            if error == 0:
                error = 1e-10
            alpha = 0.5 * np.log((1 - error) / error)
            stump.alpha = alpha

            predictions = stump.predict(X)
            # Update sample weights: increase for misclassified samples
            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)

            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        stump_preds = np.array([stump.alpha * stump.predict(X) for stump in self.stumps])
        agg_preds = np.sum(stump_preds, axis=0)
        return np.where(agg_preds >= 0, 1, -1)