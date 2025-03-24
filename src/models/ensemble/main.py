#!/usr/bin/env python3
import argparse
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

# Add project src to sys.path so that common modules can be imported.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common.data_loader import load_data
from common.evaluation import compute_metrics
from common.preprocessing import log_transform, standardize_train, standardize_test
from common.plots import plot_learning_curves

from models.decision_tree.builders import build_model_decision_tree
from models.perceptron.builders import build_standard_perceptron, build_margin_perceptron, build_averaged_perceptron
from models.ensemble.builders import build_ensemble_model

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
    # Optional flag to plot learning curves.
    parser.add_argument('--plot', action='store_true', help="Plot learning curves for the ensemble model.")
    #Number of epochs for plotting learning curve 
    parser.add_argument('--plot_epochs', type=int, default=20, help="Number of epochs for plotting learning curve.")
    args = parser.parse_args()
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

    ensemble_hyperparams = {
        "dt_params": {
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split
        },
        "perc_params": {
            "epochs": args.epochs,
            "lr": args.lr,
            "decay_lr": args.decay_lr,
            "mu": args.mu
        },
        "w_dt": args.w_dt,
        "w_perc": args.w_perc
    }

    #Training
    # dt_model = build_model_decision_tree(X_train_full, y_train_full, hyperparams_dt)
    # perc_model = build_averaged_perceptron(X_train_full, y_train_perc, hyperparams_perc)

    model = build_ensemble_model(X_train_full, y_train_full, ensemble_hyperparams)

    # Evaluate on training set.
    train_preds = model.predict(X_train_full)
    train_prec, train_rec, train_f1 = compute_metrics(y_train_full, train_preds)
    print("Final Training Metrics (Ensemble):")
    print(f"Precision: {train_prec:.3f}, Recall: {train_rec:.3f}, F1-score: {train_f1:.3f}")

    # Evaluate on test set.
    test_preds = model.predict(X_test)
    if y_test is not None:
        test_prec, test_rec, test_f1 = compute_metrics(y_test, test_preds)
        print("Test Metrics (Ensemble):")
        print(f"Precision: {test_prec:.3f}, Recall: {test_rec:.3f}, F1-score: {test_f1:.3f}")
    else:
        print("Test labels not available; ready for submission.")

    # Persist the trained model to disk.
    joblib.dump(model, 'output/best_model_{model}.pkl')
    print("Trained model saved to output/best_model.pkl")

    

if __name__ == '__main__':
    main()
