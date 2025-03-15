import argparse
import numpy as np
import sys
from pathlib import Path

project_src = Path(__file__).resolve().parents[1]
sys.path.append(str(project_src))

from common.data_loader import load_data
from common.cross_validation import grid_search_cv_generic
from models.decision_tree.builders import build_model_decision_tree
from models.perceptron.builders import build_standard_perceptron


model_builders = {
    "dt" : build_model_decision_tree,
    "perc": build_standard_perceptron
}

def main():
    parser = argparse.ArgumentParser(description="Generic Hyperparameter Tuning Script")
    parser.add_argument('--model', type=str, default='dt', choices=['dt', 'perc'],
                        help="Select model to tune: 'dt' for decision tree, 'perc' for standard perceptron")
    parser.add_argument('--k', type=int, default=5, help="Number of folds for cross-validation")
    args = parser.parse_args()
    
    # Load training data
    X_train, y_train = load_data("data/train.csv", label_column="label")
    
    # For perceptron, convert labels to {-1, +1}
    if args.model == 'perc':
        y_train = np.where(y_train == 0, -1, 1)

    # For decision tree, we assume labels are already {0,1}.
    if args.model == 'dt':
        label_conversion = lambda labels: labels  # identity function
    elif args.model == 'perc':
        # Convert from {-1,1} to {0,1} for evaluation.
        label_conversion = lambda labels: np.where(labels == -1, 0, labels)
    else:
        label_conversion = lambda labels: labels
    
    # Define hyperparameter grid based on the model type.
    if args.model == 'dt':
        # Example grid for a decision tree:
        hyperparam_grid = [
            {"max_depth": d, "min_samples_split": s}
            for d in [5, 10, 15]
            for s in [2, 5, 10]
        ]
    elif args.model == 'perc':
        # Example grid for a standard perceptron:
        hyperparam_grid = [
            {"epochs": e, "lr": lr, "decay_lr": decay, "mu": 0}
            for e in [5, 10, 15]
            for lr in [0.1, 0.5, 1.0]
            for decay in [False, True]
        ]
    else:
        raise ValueError("Unknown model type.")
    
    model_builder = model_builders[args.model]
    
    # Run grid search with cross-validation
    label_conversion = lambda labels: np.where(labels == -1, 0, labels)
    best_params, best_metric = grid_search_cv_generic(X_train, y_train, model_builder, hyperparam_grid, k=args.k, label_conversion=label_conversion)
    print(f"Best hyperparameters for {args.model}: {best_params} with best metric (F1 or Accuracy): {best_metric:.3f}")

if __name__ == '__main__':
    main()