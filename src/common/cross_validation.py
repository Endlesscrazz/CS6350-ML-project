import numpy as np
from models.decision_tree import build_tree, predict
from common.evaluation import compute_metrics

def k_fold_splits(X, y, k=5, seed=42):
    """
    Function to split data into k folds
    """

    np.random.seed(seed)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    fold_size = len(y) // k
    folds = []

    for i in range(k):
        if i < k-1:
            fold = indices[i * fold_size: (i+1) * fold_size]
        else:
            fold = indices[i * fold_size :]
        
        folds.append(fold)
    
    return folds

def stratified_k_fold_splits(y, k=5, seed=42):
    """
    Function to split data indices into stratified k folds.
    Each fold will have approximately the same proportion of classes as y.
    """
    np.random.seed(seed)
    folds = [[] for _ in range(k)]

    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        np.random.shuffle(label_indices)
        splits = np.array_split(label_indices, k)

        for i in range(k):
            folds[i].extend(splits[i])

    folds = [np.array(fold) for fold in folds]
    return folds

def grid_search_cv_generic(X, y, model_builder, hyperparam_grid, k=5, label_conversion=lambda labels: labels, stratified=True):

    best_f1 = -np.inf
    best_params = None

    for params in hyperparam_grid:
            
            avg_precision, avg_recall, avg_f1_score = cross_validations_generic(X, y, model_builder, params, k=k, 
                                                                                label_conversion=label_conversion,stratified=True)
            print(f"Parameters: {params} => F1: {avg_f1_score:.3f}")

            if avg_f1_score > best_f1:
                best_f1 = avg_f1_score
                best_params = params

    return best_params, best_f1

def cross_validations_generic(X, y, model_builder, hyperparams, k=5, label_conversion=lambda labels: labels, stratified=True):
    """
    Function to perform k-fold cross validation
    """

    if stratified:
        folds = stratified_k_fold_splits(y, k=k)
    else:
        folds = k_fold_splits(X, y, k=k)

    precision_list = []
    recall_list = []
    f1_score_list = []

    for i in range(k):

        val_indices = np.array(folds[i]).astype(np.intp)
        train_indices = np.concatenate([folds[j] for j in range(k) if j != i]).astype(np.intp)

        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X[val_indices]
        y_val_fold = y[val_indices]

        model = model_builder(X_train_fold, y_train_fold, hyperparams)
        val_preds = np.array(model.predict(X_val_fold))

        y_val_conv = label_conversion(y_val_fold)
        preds_conv = label_conversion(val_preds)

        #print(f"Fold {i}: Unique y_val_conv: {np.unique(y_val_conv)}, Unique preds_conv: {np.unique(preds_conv)}")

        precision, recall, f1 = compute_metrics(y_val_conv, preds_conv)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1)
    
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_score_list)

    return avg_precision, avg_recall, avg_f1