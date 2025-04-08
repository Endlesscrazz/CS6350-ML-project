# models/adaboost/main.py
import argparse
import numpy as np
from pathlib import Path
import joblib
import os
import sys

# Add project src to sys.path to import common modules
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common.data_loader import load_data
from common.evaluation import compute_metrics
from models.adaboost.builders import build_model_adaboost

def main():
    parser = argparse.ArgumentParser(description="Train AdaBoost model and evaluate performance.")
    # Optional command-line overrides for hyperparameters.
    parser.add_argument('--n_estimators', type=int, help="Number of boosting rounds (optional override).")
    parser.add_argument('--n_thresholds', type=int, help="Number of thresholds (optional override).")
    parser.add_argument('--n_weak_learner_d', type=int, help="Depth of weak learner (optional override).")
    args = parser.parse_args()

    model_type = "adaboost"
    tuned_config_path = os.path.join("tuned_models", f"{model_type}_best_model_config.pkl")
    
    if os.path.exists(tuned_config_path):
        best_config = joblib.load(tuned_config_path)
        print(f"Loaded tuned configuration from {tuned_config_path}")
        # Extract the best pre-processing pipeline and tuned hyperparameters.
        preprocessing_pipeline = best_config["pipeline"]
        tuned_hyperparams = best_config["hyperparameters"]
    else:
        print(f"Tuned configuration for model '{model_type}' not found. Using default preprocessing pipeline.")
        from common.preprocessing import preprocessing_pipeline
        preprocessing_pipeline = preprocessing_pipeline
        tuned_hyperparams = {}

    # Allow command-line overrides if specified.
    if args.n_estimators is not None:
        tuned_hyperparams["n_estimators"] = args.n_estimators
    if args.n_thresholds is not None:
        tuned_hyperparams["n_thresholds"] = args.n_thresholds
    if args.n_weak_learner_d is not None:
        tuned_hyperparams["n_weak_learner_d"] = args.n_weak_learner_d

    # Load data.
    X_train, y_train = load_data("data/train.csv", label_column="label")
    X_test, y_test = load_data("data/test.csv", label_column="label")

    # Preprocess data using the tuned or default pipeline.
    X_train_trans = preprocessing_pipeline.fit_transform(X_train)
    X_test_trans = preprocessing_pipeline.transform(X_test)

    # Persist the used preprocessing pipeline.
    joblib.dump(preprocessing_pipeline, 'output/preprocessing_pipeline.pkl')
    print("Preprocessing pipeline saved to output/preprocessing_pipeline.pkl")

    # Convert labels as AdaBoost typically expects {-1, +1}.
    y_train = np.where(y_train == 0, -1, 1)
    y_test_conv = np.where(y_test == 0, -1, 1)

    # Build the AdaBoost model using the tuned hyperparameters.
    model = build_model_adaboost(X_train_trans, y_train, tuned_hyperparams)

    # Evaluate on training data.
    train_preds = model.predict(X_train_trans)
    train_prec, train_rec, train_f1 = compute_metrics(y_train, train_preds)
    print("Training Metrics:")
    print(f"Precision: {train_prec:.3f}, Recall: {train_rec:.3f}, F1-score: {train_f1:.3f}")

    # Evaluate on test data.
    test_preds = model.predict(X_test_trans)
    test_prec, test_rec, test_f1 = compute_metrics(y_test_conv, test_preds)
    print("Test Metrics:")
    print(f"Precision: {test_prec:.3f}, Recall: {test_rec:.3f}, F1-score: {test_f1:.3f}")

    joblib.dump(model, 'output/adaboost_model.pkl')
    print("Model saved to output/adaboost_model.pkl")

if __name__ == '__main__':
    main()
