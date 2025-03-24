#!/usr/bin/env python3
import argparse
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project src to sys.path so that common modules can be imported.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common.data_loader import load_data
from common.evaluation import compute_metrics
from common.preprocessing import log_transform, standardize_train, standardize_test
from common.plots import plot_learning_curves

from models.decision_tree.builders import build_model_decision_tree
from models.perceptron.builders import build_standard_perceptron, build_margin_perceptron, build_averaged_perceptron
from models.ensemble.ensemble_dt_perc import ensemble_predict

def convert_labels(y):
    """
    Convert labels from {0,1} to {-1,+1} for perceptron training.
    """
    return np.where(y == 0, -1, 1)

def plot_ensemble_learning_curve(X_train, y_train, X_val, y_val,
                                 hyperparams_dt, hyperparams_perc,
                                 epochs, weights, convert_func):
    """
    For each epoch count from 1 to 'epochs', retrain the ensemble (decision tree and margin perceptron)
    on the training data and compute the F1 score on the validation set.
    Then, plot the validation F1 score vs. epochs.
    """
    val_f1_scores = []
    epoch_list = []
    
    for e in range(1, epochs+1):
        # Train decision tree (non-iterative; retrain for consistency)
        dt_model = build_model_decision_tree(X_train, y_train, hyperparams_dt)
        # Update perceptron hyperparameters for current epoch count.
        perc_params = hyperparams_perc.copy()
        perc_params["epochs"] = e
        perc_model = build_margin_perceptron(X_train, convert_labels(y_train), perc_params)
        
        # Get validation predictions
        dt_preds = dt_model.predict(X_val)  # assumed in {0,1}
        perc_preds = perc_model.predict(X_val)  # returns {-1,1}
        perc_preds = np.where(np.array(perc_preds) == -1, 0, perc_preds)
        
        # Ensemble predictions
        ensemble_preds = ensemble_predict([dt_preds, perc_preds], weights=weights)
        
        # Compute F1 score on validation set (assume y_val is in {0,1})
        _, _, f1_val = compute_metrics(y_val, ensemble_preds)
        val_f1_scores.append(f1_val)
        epoch_list.append(e)
    
    plt.plot(epoch_list, val_f1_scores, marker='o', label="Validation F1 (Ensemble)")
    plt.xlabel("Epochs (Perceptron Training)")
    plt.ylabel("F1 Score")
    plt.title("Ensemble Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


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
    perc_model = build_averaged_perceptron(X_train_full, y_train_perc, hyperparams_perc)

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
    
    # Optionally, plot learning curves for the ensemble.
    if args.plot:
        # For plotting, we need to split training data into train/val.
        # We'll use a simple train/validation split.
        def train_val_split(X, y, test_size=0.2, random_state=42):
            np.random.seed(random_state)
            indices = np.arange(len(y))
            np.random.shuffle(indices)
            split = int(len(y) * (1 - test_size))
            train_idx = indices[:split]
            val_idx = indices[split:]
            return X[train_idx], X[val_idx], y[train_idx], y[val_idx]
        
        X_train_plot, X_val_plot, y_train_plot, y_val_plot = train_val_split(X_train_full, y_train_full, test_size=0.2)
        # For the perceptron branch, convert training labels.
        y_train_plot_perc = convert_labels(y_train_plot)
        # Now, define a function to train and ensemble for a given number of perceptron epochs.
        train_f1_scores = []
        val_f1_scores = []
        epoch_list = []
        for e in range(1, args.plot_epochs + 1):
            # Train dt model (we retrain dt each time for consistency, though dt may not change much)
            dt_model_plot = build_model_decision_tree(X_train_plot, y_train_plot, hyperparams_dt)
            # Train perceptron model for e epochs.
            perc_model_plot = build_margin_perceptron(X_train_plot, y_train_plot_perc, {**hyperparams_perc, "epochs": e})
            
            dt_val_preds = dt_model_plot.predict(X_val_plot)
            perc_val_preds = perc_model_plot.predict(X_val_plot)
            perc_val_preds = np.where(np.array(perc_val_preds) == -1, 0, perc_val_preds)
            ensemble_val_preds = ensemble_predict([dt_val_preds, perc_val_preds], weights=[args.w_dt, args.w_perc])
            _, _, f1_val = compute_metrics(y_val_plot, ensemble_val_preds)
            val_f1_scores.append(f1_val)
            epoch_list.append(e)
        
        plt.plot(epoch_list, val_f1_scores, marker='o', label="Validation F1 (Ensemble)")
        plt.xlabel("Epochs (Perceptron Training)")
        plt.ylabel("F1 Score")
        plt.title("Ensemble Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    

if __name__ == '__main__':
    main()
