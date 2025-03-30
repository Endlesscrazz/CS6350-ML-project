import argparse
import numpy as np
import sys
import random
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

# Append project source directory.
project_src = Path(__file__).resolve().parents[1]
sys.path.append(str(project_src))

from common.data_loader import load_data
from common.cross_validation import grid_search_cv_generic
from common.preprocessing import standardize_train, log_transform, remove_low_variance_features, apply_truncated_svd
from models import (
    build_model_decision_tree,
    build_standard_perceptron,
    build_averaged_perceptron,
    build_margin_perceptron,
    build_ensemble_model,
    build_model_adaboost

)

def identity_conversion(labels):
    return labels

def convert_neg1_to0(labels):
    return np.where(labels == -1, 0, labels)

model_builders = {
    "dt": build_model_decision_tree,
    "perc": build_standard_perceptron,
    "avgperc": build_averaged_perceptron,
    "marginperc": build_margin_perceptron,
    "ensemble": build_ensemble_model,
    "adaboost": build_model_adaboost
}

def evaluate_params(params, X_train, y_train, model_builder, k, label_conversion):
    print("Evaluating params:", params, type(params))
    try:
        best_params, f1 = grid_search_cv_generic(X_train, y_train, model_builder, [params], k=k, 
                                                label_conversion=label_conversion,stratified=True)
    except Exception as e:
        print(f"Exception for params {params}: {e}")
        f1 = -np.inf

    return params, f1

def main():
    parser = argparse.ArgumentParser(description="Generic Hyperparameter Tuning Script")
    parser.add_argument('--model', type=str, default='dt', choices=['dt', 'perc', 'avgperc', 'marginperc', 'ensemble'],
                        help="Select model to tune: 'dt' for decision tree, 'perc' for standard perceptron, 'avgperc' for averaged perceptron, 'marginperc' for margin perceptron, 'ensemble' for ensemble model")
    parser.add_argument('--k', type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument('--n_iter', type=int, default=20, help="Number of random hyperparameter combinations to try")
    args = parser.parse_args()
    
    # Load training data.
    X_train, y_train = load_data("data/train.csv", label_column="label")

    # Preprocess data.
    X_train = log_transform(X_train)
    X_train, train_mean, train_std = standardize_train(X_train)

    # Feature Engineering
    X_train, _ , _ = remove_low_variance_features(X_train, X_train, threshold=1e-4)
    X_train, _ , _ = apply_truncated_svd(X_train, X_train, n_components=50)

    
    # For perceptron-based models, convert labels to {-1, +1}.
    if args.model in ['perc', 'avgperc', 'marginperc']:
        y_train = np.where(y_train == 0, -1, 1)

    # Set label conversion function.
    if args.model == 'dt':
        label_conversion = identity_conversion
    else:
        label_conversion = convert_neg1_to0

    # Define hyperparameter grid.
    if args.model == 'dt':
        hyperparam_grid = [
            {"max_depth": d, "min_samples_split": s}
            for d in [5, 10]
            for s in [2, 5, 10]
        ]
    elif args.model == 'perc':
        hyperparam_grid = [
            {"epochs": e, "lr": lr, "decay_lr": decay, "mu": 0}
            for e in [10, 20, 30]
            for lr in [0.1, 0.5, 1.0]
            for decay in [False, True]
        ]
    elif args.model == 'avgperc':
        hyperparam_grid = [
            {"epochs": e, "lr": lr}
            for e in [10, 20, 30]
            for lr in [0.1, 0.5, 1.0]
        ]
    elif args.model == 'marginperc':
        hyperparam_grid = [
            {"epochs": e, "lr": lr, "mu": mu}
            for e in [10, 20, 30]
            for lr in [0.1, 0.5, 1.0]
            for mu in [0.8, 1.0, 1.2]
        ]
    elif args.model == "ensemble":
        hyperparam_grid = [
            {
                "dt_params": {"max_depth": d, "min_samples_split": s},
                "perc_params": {"epochs": e, "lr": lr, "decay_lr": decay_lr, "mu": 0},
                "w_dt": w_dt,
                "w_perc": 1 - w_dt
            }
            for d in [5, 10]
            for s in [2, 5]
            for e in [10, 20]
            for lr in [0.1, 0.5]
            for decay_lr in [False, True]
            for w_dt in [0.5, 0.6, 0.7]
        ]
    elif args.model == "adaboos":
        hyperparam_grid = [
            {"n_estimators":n}
            for n in [50, 100, 150]
        ]
    else:
        raise ValueError("Unknown model type.")
    
    # Random search: sample a subset if grid is larger than n_iter.
    if len(hyperparam_grid) > args.n_iter:
        original_grid_size = len(hyperparam_grid)
        hyperparam_grid = random.sample(hyperparam_grid, args.n_iter)
        print(f"Random search: evaluating {args.n_iter} out of {original_grid_size} hyperparameter combinations.")
    
    if args.model == "ensemble":
        def ensemble_builder_wrapper(X, y, params):

            # min_val_samples = max(20, min(5, len(X) // 5))  # 20% or at least 1 sample
            # if len(X) - min_val_samples < 2:  # Need at least 2 training samples
            #     min_val_samples = len(X) - 2 if len(X) > 2 else 0

            # if min_val_samples > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2,stratify=y, random_state=42)
            #print(f"\nDEBUG: Training on {len(X_train)} samples, validating on {len(X_val)} samples")
            model = build_ensemble_model(X_train, y_train, params)
            #print("DEBUG: Before weight update - w_dt:", model.w_dt, "w_perc:", model.w_perc)
            model.update_weights(X_val, y_val)
            #print("DEBUG: After weight update - w_dt:", model.w_dt, "w_perc:", model.w_perc)

            # else:
            #     model = build_ensemble_model(X, y, params)
            return model
        model_builder = ensemble_builder_wrapper
    else:
        model_builder = model_builders[args.model]

    # Parallelize evaluation of hyperparameter combinations.
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_params)(params, X_train, y_train, model_builder, args.k, label_conversion)
        for params in hyperparam_grid
    )
    
    # results = [evaluate_params(params, X_train, y_train, model_builder, args.k, label_conversion)
    #        for params in hyperparam_grid]

    best_params, best_metric = max(results, key=lambda x: x[1])
    print(f"Best hyperparameters for {args.model}: {best_params} with best metric: {best_metric:.3f}")

if __name__ == '__main__':
    # import cProfile
    # cProfile.run('main()', 'profile_stats')
    main()
