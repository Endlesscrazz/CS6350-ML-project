from models.decision_tree.decision_tree import DecisionTree

def build_model_decision_tree(X_train, y_train, hyperparams):

    max_depth = hyperparams.get('max_depth', 10)
    min_sample_splits = hyperparams.get('min_sample_splits', 5)

    model = DecisionTree(max_depth=max_depth, min_samples_split=min_sample_splits)
    model.train(X_train, y_train)

    return model