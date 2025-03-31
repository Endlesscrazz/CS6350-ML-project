import numpy as np

# class DecisionStump:
#     """
#     A simple decision stump used as a weak learner in AdaBoost.
#     It finds the best feature and threshold that minimizes the weighted error.
#     """
#     def __init__(self):
#         self.feature_index = None
#         self.threshold = None
#         self.polarity = 1
#         self.alpha = None  # Fixed typo from aplha to alpha

#     def train(self, X, y, sample_weights, n_thresholds=10):
#         m, n = X.shape
#         min_error = float('inf')
#         for feature_idx in range(n):
#             X_column = X[:, feature_idx]
#             thresholds = np.unique(X_column)
#             # too many unique thresholds, sample a few using percentiles.
#             if len(thresholds) > n_thresholds:
#                 thresholds = np.percentile(X_column, np.linspace(0, 100, n_thresholds))
#             for threshold in thresholds:
#                 for polarity in [1, -1]:
#                     preds = np.where(X_column * polarity < threshold * polarity, -1, 1)
#                     error = np.sum(sample_weights * (preds != y))
#                     if error < min_error:
#                         min_error = error
#                         self.feature_index = feature_idx
#                         self.threshold = threshold
#                         self.polarity = polarity
#         return min_error

#     def predict(self, X):
#         X_column = X[:, self.feature_index]
#         return np.where(X_column * self.polarity < self.threshold * self.polarity, -1, 1)
    
def weighted_majority(y, sample_weights):
    """Compute the weighted majority vote for a set of labels."""
    pos_weight = np.sum(sample_weights[y == 1])
    neg_weight = np.sum(sample_weights[y == -1])
    return 1 if pos_weight >= neg_weight else -1

class DecisionTreeNode:
    def __init__(self, is_leaf=False, prediction=None, feature_index=None,
                 threshold=None, polarity=1, left=None, right=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature_index = feature_index
        self.threshold = threshold
        self.polarity = polarity
        self.left = left
        self.right = right

class DecisionTreeWeakLearner:
    """
    A shallow decision tree weak learner that supports a maximum depth.
    If max_depth=1, it behaves as a decision stump.
    """
    def __init__(self, max_depth=2, n_thresholds=10):
        self.max_depth = max_depth
        self.n_thresholds = n_thresholds
        self.tree = None

    def train(self, X, y, sample_weights):
        self.tree = self._build_tree(X, y, sample_weights, depth=0)
        
        preds = self.predict(X)
        return np.sum(sample_weights * (preds != y))

    def _build_tree(self, X, y, sample_weights, depth):
        # Base case: if homogeneous labels or reached maximum depth, return a leaf.
        if np.all(y == y[0]) or depth == self.max_depth:
            leaf_prediction = weighted_majority(y, sample_weights)
            return DecisionTreeNode(is_leaf=True, prediction=leaf_prediction)
        
        m, n = X.shape
        best_error = float('inf')
        best_feature = None
        best_threshold = None
        best_polarity = 1
        best_left_indices = None
        best_right_indices = None

        # Iterate over features.
        for feature_idx in range(n):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            # If too many unique thresholds, sample a few using percentiles.
            if len(thresholds) > self.n_thresholds:
                thresholds = np.percentile(X_column, np.linspace(0, 100, self.n_thresholds))
            for threshold in thresholds:
                for polarity in [1, -1]:
                    preds = np.where(X_column * polarity < threshold * polarity, -1, 1)
                    error = np.sum(sample_weights * (preds != y))
                    if error < best_error:
                        best_error = error
                        best_feature = feature_idx
                        best_threshold = threshold
                        best_polarity = polarity
                        best_left_indices = np.where(X_column * polarity < threshold * polarity)[0]
                        best_right_indices = np.where(X_column * polarity >= threshold * polarity)[0]

        # If no split found (should not happen), return a leaf.
        if best_threshold is None or best_left_indices is None or len(best_left_indices) == 0 or len(best_right_indices) == 0:
            leaf_prediction = weighted_majority(y, sample_weights)
            return DecisionTreeNode(is_leaf=True, prediction=leaf_prediction)

        left_node = self._build_tree(X[best_left_indices], y[best_left_indices],
                                     sample_weights[best_left_indices], depth + 1)
        right_node = self._build_tree(X[best_right_indices], y[best_right_indices],
                                      sample_weights[best_right_indices], depth + 1)
        return DecisionTreeNode(is_leaf=False,
                                feature_index=best_feature,
                                threshold=best_threshold,
                                polarity=best_polarity,
                                left=left_node,
                                right=right_node)

    def _predict_node(self, node, x):
        if node.is_leaf:
            return node.prediction
        if x[node.feature_index] * node.polarity < node.threshold * node.polarity:
            return self._predict_node(node.left, x)
        else:
            return self._predict_node(node.right, x)

    def predict(self, X):
        m = X.shape[0]
        preds = np.array([self._predict_node(self.tree, X[i]) for i in range(m)])
        return preds

class AdaBoostModel:
    """
    AdaBoost ensemble using decision stumps as weak learners.
    """
    def __init__(self, n_estimators=50, n_thresholds=10, weak_learner_depth=2):
        self.n_estimators = n_estimators
        self.n_thresholds = n_thresholds
        self.weak_learner_depth = weak_learner_depth
        self.stumps = []
        self.alphas = []
        self.verbose = False

    # def _initialize_sample_weights(self, y):
    #     m = len(y)
    #     if self.init_weights_method == 'balanced':
    #         # Compute weights inversely proportional to class frequencies.
    #         unique, counts = np.unique(y, return_counts=True)
    #         class_weights = {cls: 1.0 / count for cls, count in zip(unique, counts)}
    #         sample_weights = np.array([class_weights[val] for val in y])
    #         # Normalize the weights.
    #         sample_weights = sample_weights / np.sum(sample_weights)
    #     else:  # default is uniform
    #         sample_weights = np.full(m, 1.0 / m)
    #     return sample_weights

    def train(self, X, y):
        m, _ = X.shape
        sample_weights = np.full(m, 1.0 / m)
        self.stumps = []
        self.alphas = []
        for i in range(self.n_estimators):

            learner = DecisionTreeWeakLearner(max_depth=self.weak_learner_depth, n_thresholds=self.n_thresholds)
            #stump = DecisionStump()
            error = learner.train(X, y, sample_weights)
            # Avoid division by zero by setting a minimum error
            if error == 0:
                error = 1e-10
            alpha = 0.5 * np.log((1 - error) / error)
            learner.alpha = alpha

            preds = learner.predict(X)
            # Update sample weights: increase for misclassified samples
            sample_weights *= np.exp(-alpha * y * preds)
            sample_weights /= np.sum(sample_weights)

            self.stumps.append(learner)
            self.alphas.append(alpha)

            if self.verbose and (i % 10 == 0 or i == self.n_estimators - 1):
                current_pred = self.predict(X)
                acc = np.mean(current_pred == y)
                print(f"Round {i+1}/{self.n_estimators} | Acc: {acc:.4f} | Error: {error:.4f} | Alpha: {alpha:.4f}")

    def predict(self, X):
        stump_preds = np.array([stump.alpha * stump.predict(X) for stump in self.stumps])
        agg_preds = np.sum(stump_preds, axis=0)
        return np.where(agg_preds >= 0, 1, -1)

    def predict_submission(self, X):
        """
        Returns predictions in {0,1} for submission purposes.
        """
        raw_preds = self.predict(X)
        # Convert predictions: -1 becomes 0, 1 stays 1.
        return np.where(raw_preds == -1, 0, raw_preds)