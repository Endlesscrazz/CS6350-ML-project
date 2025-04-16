# --- START OF MODIFIED FILE models/nn/main.py ---

import argparse
import numpy as np
from pathlib import Path
import joblib
import os
import sys
import torch
from sklearn.model_selection import train_test_split # Import train_test_split

# Add project src to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common.data_loader import load_data
from common.evaluation import compute_metrics
from models.nerual_network.nn import build_model_nn, TorchWrapper
from sklearn.base import clone

def find_optimal_threshold(model_wrapper: TorchWrapper, X_val: np.ndarray, y_val_01: np.ndarray):
    """
    Finds the optimal prediction threshold on the validation set based on F1 score.

    Args:
        model_wrapper: Trained TorchWrapper instance.
        X_val: Validation features.
        y_val_01: Validation true labels in {0, 1} format.

    Returns:
        best_threshold: The threshold maximizing F1 score.
        best_f1: The maximum F1 score achieved on the validation set.
    """
    print("\nFinding optimal threshold on validation set...")
    val_logits = model_wrapper.predict_logits(X_val)

    best_f1 = -1
    best_threshold = 0.0
    # Define candidate thresholds - adjust range and number based on logit distribution if needed
    thresholds = np.linspace(np.min(val_logits) - 1e-3, np.max(val_logits) + 1e-3, 200)

    if len(np.unique(y_val_01)) < 2:
        print("Warning: Validation set contains only one class. Cannot compute F1 score reliably.")
        
        return 0.0, 0.0

    f1_scores = []
    for threshold in thresholds:
        val_preds_01 = np.where(val_logits >= threshold, 1, 0)
        # compute_metrics returns (precision, recall, f1)
        _, _, f1 = compute_metrics(y_val_01, val_preds_01)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Optimal threshold found: {best_threshold:.4f} with Validation F1: {best_f1:.4f}")

    # DEBUG: Plot F1 vs Threshold
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Prediction Threshold on Validation Set")
    plt.axvline(best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.2f}')
    plt.legend()
    plt.grid(True)
    # plt.savefig(project_root / "output/nn_threshold_f1_plot.png") # Save the plot
    # print(f"Saved threshold plot to {project_root / 'output/nn_threshold_f1_plot.png'}")

    return best_threshold, best_f1


def main():
    parser = argparse.ArgumentParser(description="Train Neural Network model, find optimal threshold, and evaluate.")
    # Optional command-line overrides for scalar hyperparameters.
    parser.add_argument('--lr', type=float, help="Learning rate (optional override).")
    parser.add_argument('--epochs', type=int, help="Max epochs (optional override).")
    parser.add_argument('--batch_size', type=int, help="Batch size (optional override).")
    parser.add_argument('--dropout', type=float, help="Dropout rate (optional override).")
    parser.add_argument('--weight_decay', type=float, help="Weight decay (optional override).")
    parser.add_argument('--val_size', type=float, default=0.2, help="Proportion of training data for validation split to find threshold.")
    args = parser.parse_args()

    model_type = "nn"
    tuned_config_path = project_root / "tuned_models" / f"{model_type}_best_model_config.pkl"
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if tuned_config_path.exists():
        print(f"Loading tuned configuration from {tuned_config_path}")
        best_config = joblib.load(tuned_config_path)
        preprocessing_pipeline_config = best_config["pipeline"]
        tuned_hyperparams = best_config["hyperparameters"]
        print("Loaded hyperparameters:", tuned_hyperparams)
        print("Loaded pipeline config:", preprocessing_pipeline_config)
    else:
        raise FileNotFoundError(f"Tuned configuration for model '{model_type}' not found at {tuned_config_path}. Please run tune_models.py first.")

    # Allow command-line overrides
    if args.lr is not None: tuned_hyperparams["lr"] = args.lr
    if args.epochs is not None: tuned_hyperparams["epochs"] = args.epochs
    if args.batch_size is not None: tuned_hyperparams["batch_size"] = args.batch_size
    if args.dropout is not None: tuned_hyperparams["dropout"] = args.dropout
    if args.weight_decay is not None: tuned_hyperparams["weight_decay"] = args.weight_decay

    # --- Load Data ---
    X_train_orig, y_train_orig = load_data(project_root / "data/train.csv", label_column="label")
    X_test_orig, y_test_orig = load_data(project_root / "data/test.csv", label_column="label")

    # --- Preprocessing ---
    # Fit pipeline on the *entire* original training data
    preprocessing_pipeline = clone(preprocessing_pipeline_config)
    print("\nFitting preprocessing pipeline on full original training data...")
    X_train_full_trans = preprocessing_pipeline.fit_transform(X_train_orig)
    print("Transforming original test data...")
    X_test_trans = preprocessing_pipeline.transform(X_test_orig)
    print(f"Full training data shape after transform: {X_train_full_trans.shape}")
    print(f"Test data shape after transform: {X_test_trans.shape}")

    # --- Create Train/Validation Split for Threshold Finding ---
    # Split the *transformed* full training data
    # Use original labels ({0, 1}) for stratification here
    # Use a fixed random state for this split if reproducibility is desired
    split_seed = tuned_hyperparams.get("random_state", 42) + 10 # Use a different seed
    print(f"\nSplitting transformed training data into train/validation ({1-args.val_size:.0%}/{args.val_size:.0%}) for model training and threshold tuning...")
    X_train_fold, X_val_fold, y_train_fold_01, y_val_fold_01 = train_test_split(
        X_train_full_trans,
        y_train_orig, # Use original {0, 1} labels for stratify and later validation F1 calc
        test_size=args.val_size,
        stratify=y_train_orig,
        random_state=split_seed
    )
    print(f"Train fold shape: {X_train_fold.shape}, Validation fold shape: {X_val_fold.shape}")

    # --- Convert Labels for NN Training ---
    # build_model_nn expects y in {-1, 1}
    y_train_fold_m1p1 = np.where(y_train_fold_01 == 0, -1, 1)

    # --- Build and Train Model on the Training Fold ---
    # Pass the tuned hyperparameters
    print("\nBuilding and training Neural Network model on the training fold...")
    # Ensure random state is passed for internal early stopping split consistency if needed
    tuned_hyperparams["random_state"] = split_seed + 1 # Ensure different seed for internal split
    model_wrapper = build_model_nn(X_train_fold, y_train_fold_m1p1, tuned_hyperparams)
    print("Model training complete.")

    # --- Find Optimal Threshold on Validation Fold ---
    optimal_threshold, val_f1 = find_optimal_threshold(model_wrapper, X_val_fold, y_val_fold_01)

    # --- Evaluation ---
    # Evaluate on the training FOLD using the optimal threshold
    print("\nEvaluating on training fold (used for actual training)...")
    train_fold_preds_01 = model_wrapper.predict_submission(X_train_fold, threshold=optimal_threshold)
    train_prec, train_rec, train_f1 = compute_metrics(y_train_fold_01, train_fold_preds_01)
    print("Training Fold Metrics (using optimal threshold):")
    print(f"Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1-score: {train_f1:.4f}")

    # Evaluate on the validation FOLD using the optimal threshold (should match val_f1 found)
    print("\nEvaluating on validation fold (used for threshold tuning)...")
    val_fold_preds_01 = model_wrapper.predict_submission(X_val_fold, threshold=optimal_threshold)
    val_prec, val_rec, val_f1_check = compute_metrics(y_val_fold_01, val_fold_preds_01)
    print("Validation Fold Metrics (using optimal threshold):")
    print(f"Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1-score: {val_f1_check:.4f} (should approx match {val_f1:.4f})")


    # Evaluate on the final TEST set using the optimal threshold
    print("\nEvaluating on final test set...")
    test_preds_01 = model_wrapper.predict_submission(X_test_trans, threshold=optimal_threshold)
    test_prec, test_rec, test_f1 = compute_metrics(y_test_orig, test_preds_01)
    print("Test Set Metrics (using optimal threshold):")
    print(f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1-score: {test_f1:.4f}")

    # --- Save Model and Threshold ---
    # Save the TorchWrapper object, the fitted pipeline, and the threshold
    output_data = {
        'model_wrapper': model_wrapper,
        'preprocessing_pipeline': preprocessing_pipeline, # Already fitted on full train data
        'optimal_threshold': optimal_threshold,
        'val_f1_at_threshold': val_f1,
        'test_f1_at_threshold': test_f1 # Store test F1 for reference
    }
    save_path = output_dir / 'nn_model_with_threshold.pkl'
    joblib.dump(output_data, save_path)
    print(f"\nModel wrapper, fitted pipeline, and optimal threshold saved to {save_path}")
    print(f"(Optimal Threshold: {optimal_threshold:.4f})")


if __name__ == '__main__':
    main()

