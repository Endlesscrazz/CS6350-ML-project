#!/usr/bin/env python3
import argparse
import numpy as np
import sys
from pathlib import Path

# Add project src to sys.path so that common modules can be imported.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common.data_loader import load_data
from common.evaluation import compute_metrics
from common.preprocessing import log_transform, standardize_train, standardize_test


from models.decision_tree.builders import build_model_decision_tree
from models.perceptron.builders import build_standard_perceptron, build_margin_perceptron
from models.ensemble.ensemble_dt_perc import ensemble_predict

def convert_labels(y):
    """
    Convert labels from {0,1} to {-1,+1} for perceptron training.
    """
    return np.where(y == 0, -1, 1)

def main():
    parser = argparse.ArgumentParser(
        description="Train ensemble model (Decision Tree + Perceptron) and evaluate training and test F1 scores."
    )
    # Hyperparameters for decision tree:
    parser.add_argument('--max_depth', type=int, default=15, help="(Decision Tree) Maximum depth.")
    parser.add_argument('--min_samples_split', type=int, default=2, help="(Decision Tree) Minimum samples to split.")
    # Hyperparameters for perceptron:
    parser.add_argument('--epochs', type=int, default=10, help="(Perceptron) Number of epochs.")
    parser.add_argument('--lr', type=float, default=0.1, help="(Perceptron) Learning rate.")
    parser.add_argument('--decay_lr', action='store_true', help="(Perceptron) Use learning rate decay.")
    parser.add_argument('--mu', type=float, default=0.0, help="(Perceptron) Margin parameter (if applicable).")
    parser.add_argument('--w_dt', type=float, default=0.6, help="Weight for decision tree predictions in ensemble.")
    parser.add_argument('--w_perc', type=float, default=0.4, help="Weight for perceptron predictions in ensemble.")
    args = parser.parse_args()

    # Load full training and test data.
    X_train_full, y_train_full = load_data("data/train.csv", label_column="label")
    X_test, y_test = load_data("data/test.csv", label_column="label")

    # Preprocess training data.
    X_train_full = log_transform(X_train_full)
    X_train_full, train_mean, train_std = standardize_train(X_train_full)
    # Preprocess test data using training parameters.
    X_test = log_transform(X_test)
    X_test = standardize_test(X_test, train_mean, train_std)

    # For decision tree, we use y_train as is (assumed {0,1}).
    # For perceptron, we need to convert training labels to {-1,1}.
    y_train_perc = convert_labels(y_train_full)

    # Set hyperparameters for each model.
    hyperparams_dt = {
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split
    }
    hyperparams_perc = {
        "epochs": args.epochs,
        "lr": args.lr,
        "decay_lr": args.decay_lr,
        "mu": args.mu  
    }

    #Training
    dt_model = build_model_decision_tree(X_train_full, y_train_full, hyperparams_dt)
    perc_model = build_margin_perceptron(X_train_full, y_train_perc, hyperparams_perc)

    # Predictions
    dt_train_preds = dt_model.predict(X_train_full)  # assumed in {0,1}
    perc_train_preds = perc_model.predict(X_train_full)  # in {-1,1}; convert them:
    perc_train_preds = np.where(np.array(perc_train_preds) == -1, 0, perc_train_preds)

    # Ensemble the training predictions.
    ensemble_train_preds = ensemble_predict(
        [dt_train_preds, perc_train_preds],
        weights=[args.w_dt, args.w_perc]
    )
    
    # Compute training metrics.
    train_prec, train_rec, train_f1 = compute_metrics(y_train_full, ensemble_train_preds)
    print("Final Training Metrics (Ensemble):")
    print(f"Precision: {train_prec:.3f}, Recall: {train_rec:.3f}, F1-score: {train_f1:.3f}")

    # Get predictions on test set.
    dt_test_preds = dt_model.predict(X_test)  # {0,1}
    perc_test_preds = perc_model.predict(X_test)  # {-1,1}; convert:
    perc_test_preds = np.where(np.array(perc_test_preds) == -1, 0, perc_test_preds)
    
    ensemble_test_preds = ensemble_predict(
        [dt_test_preds, perc_test_preds],
        weights=[args.w_dt, args.w_perc]
    )
    
    if y_test is not None:
        # If test labels are in {0,1}, no conversion is necessary.
        test_prec, test_rec, test_f1 = compute_metrics(y_test, ensemble_test_preds)
        print("Test Metrics (Ensemble):")
        print(f"Precision: {test_prec:.3f}, Recall: {test_rec:.3f}, F1-score: {test_f1:.3f}")
    else:
        print("Test labels not available; ready for submission.")

if __name__ == '__main__':
    main()
