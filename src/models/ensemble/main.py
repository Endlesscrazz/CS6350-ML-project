#!/usr/bin/env python3
import argparse
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import time

from sklearn.model_selection import train_test_split

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common.data_loader import load_data
from common.evaluation import compute_metrics
from common.preprocessing import preprocessing_pipeline
from common.plots import plot_learning_curves

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

    # Load full training and test data.
    X_train_full, y_train_full = load_data("data/train.csv", label_column="label")
    X_test, y_test = load_data("data/test.csv", label_column="label")

    # Preprocess data
    X_train_full_trans = preprocessing_pipeline.fit_transform(X_train_full)
    X_test_trans = preprocessing_pipeline.transform(X_test)

    joblib.dump(preprocessing_pipeline, 'output/preprocessing_pipeline.pkl')

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

    X_train, X_val, y_train, y_val = train_test_split(X_train_full_trans, y_train_full, test_size=0.2, random_state=42)
    
    ensemble_model = build_ensemble_model(X_train_full_trans, y_train_full, ensemble_hyperparams)

    # Updating the ensemble weights dynamically using validation data.
    ensemble_model.update_weights(X_val, y_val)

    if args.plot:

        # Plotting the learning curves
        plot_learning_curves(
            model_builder=build_ensemble_model,
            hyperparams=ensemble_hyperparams,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=20
        )
    
    # Evaluating the ensemble model using the updated instance.
    train_preds = ensemble_model.predict(X_train_full_trans)
    train_prec, train_rec, train_f1 = compute_metrics(y_train_full, train_preds)
    print("Final Training Metrics (Ensemble):")
    print(f"Precision: {train_prec:.3f}, Recall: {train_rec:.3f}, F1-score: {train_f1:.3f}")

    test_preds = ensemble_model.predict(X_test_trans)
    if y_test is not None:
        test_prec, test_rec, test_f1 = compute_metrics(y_test, test_preds)
        print("Test Metrics (Ensemble):")
        print(f"Precision: {test_prec:.3f}, Recall: {test_rec:.3f}, F1-score: {test_f1:.3f}")
    else:
        print("Test labels not available; ready for submission.")

    # Persist the trained ensemble model to disk.
    #timestamp = time.strftime("%Y%m%d_%H%M%S")
    joblib.dump(ensemble_model, f'output/best_model_ensemble.pkl')
    print("Trained model saved to output/best_model.pkl")

if __name__ == '__main__':
    main()