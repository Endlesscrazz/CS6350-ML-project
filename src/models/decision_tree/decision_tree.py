import numpy as np

def gini_impurity(y):
    """
    Computes the gini impurity for array of binary labels.
    """
    n = len(y)
    if n==0:
        return 0
    p1 = np.mean(y)
    p0 = 1 - p1

    return 1 - (p1**2 + p0**2)

def best_split(X,y):
    """
    Function to find the best feature index and threshold to split the data basd on the
    calculated gini impurity
    """

    n_samples, n_features = X.shape
    current_impurity = gini_impurity(y)
    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature_index in range(n_features):
        feature_values = X[:, feature_index]
        unique_values = np.unique(feature_values)

        for threshold in unique_values:
            left_mask = feature_values <= threshold
            right_mask = feature_values > threshold

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_impurity = gini_impurity(y[left_mask])
            right_impurity = gini_impurity(y[right_mask])

            weighted_impurity = (np.sum(left_mask) / n_samples) * left_impurity \
                                + (np.sum(right_mask) / n_samples) * right_impurity
            
            gain = current_impurity - weighted_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        """
        A node in the decision tree

        Parameters:
        feature_index (int): Index of the feature used for splitting.
        threshold (float): Threshold value for the split.
        left (DecisionTreeNode): Left child node.
        right (DecisionTreeNode): Right child node.
        value (int): Predicted label if this is a leaf node
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # If not None, this node is a leaf.

    def is_leaf(self):
        return self.value is not None
    
def build_tree(X, y, max_depth=5, min_samples_split=2, depth=0):

    """
    Recursively builds the decision tree.
    
    Parameters:
    X (np.array): 2D array of features.
    y (np.array): 1D array of labels.
    max_depth (int): Maximum depth of the tree.
    min_samples_split (int): Minimum number of samples required to split.
    depth (int): Current depth in the tree.
    
    Returns:
    DecisionTreeNode: The root node of the built tree.
    """
    n_samples = y.shape[0]

    if n_samples < min_samples_split or depth >= max_depth or np.unique(y).size == 1:
        majority_label = 1 if np.sum(y) > n_samples/2 else 0
        return DecisionTreeNode(value=majority_label)
    
    feature_index, threshold, gain = best_split(X, y)

    if gain == 0:
        majority_label = 1 if np.sum(y) > n_samples/2 else 0
        return DecisionTreeNode(value=majority_label)

    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold

    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]

    left_child = build_tree(X_left, y_left, max_depth, min_samples_split, depth+1)
    right_child = build_tree(X_right, y_right, max_depth, min_samples_split, depth+1)

    return DecisionTreeNode(feature_index, threshold, left_child, right_child)

def predict_sample(node, sample):
    """
    Function to predict the label of a single sample by traversing tree
    """

    if node.is_leaf():
        return node.value
    if sample[node.feature_index ] <= node.threshold:
        return predict_sample(node.left, sample)
    else:
        return predict_sample(node.right, sample)
    
def predict(tree, X):
    """
    Function to predict the labels for all examples in X.
    """

    return np.array([predict_sample(tree, sample)for sample in X])

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def train(self, X, y):

        self.tree = build_tree(X, y, max_depth=self.max_depth, min_samples_split=self.min_samples_split)
    
    def predict(self, X):

        return predict(self.tree, X)