import argparse
import numpy as np
from pathlib import Path
import joblib
import sys
import os

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common.data_loader import load_data
from common.evaluation import compute_metrics
from models.svm import build_model_svm

def main():
    parser = argparse.ArgumentParser(description="Train SVM model and evaluate performance.")
    # Optional command-line overrides for hyperparameters.
    parser.add_argument('--lr', type=float, help="Learning rate for SVM (optional override).")
    parser.add_argument('--lambda_param', type=float, help="Regularization strength (optional override).")
    parser.add_argument('--n_epochs', type=int, help="Number of epochs (optional override).")
    args = parser.parse_args()

    # Define model type for this main.py.
    model_type = "svm"
    tuned_config_path = os.path.join("tuned_models", f"{model_type}_best_model_config.pkl")
    
    if os.path.exists(tuned_config_path):
        best_config = joblib.load(tuned_config_path)
        print(f"Loaded tuned configuration from {tuned_config_path}")
        # Extract the best preprocessing pipeline and hyperparameters.
        preprocessing_pipeline = best_config["pipeline"]
        tuned_hyperparams = best_config["hyperparameters"]
    else:
        print(f"Tuned configuration for model '{model_type}' not found. Using default preprocessing pipeline.")
        from common.preprocessing import preprocessing_pipeline
        preprocessing_pipeline = preprocessing_pipeline
        tuned_hyperparams = {}

    # Allow command-line overrides if specified.
    if args.lr is not None:
        tuned_hyperparams["lr"] = args.lr
    if args.lambda_param is not None:
        tuned_hyperparams["lambda_param"] = args.lambda_param
    if args.n_epochs is not None:
        tuned_hyperparams["n_epochs"] = args.n_epochs

    # Load data.
    X_train, y_train = load_data("data/train.csv", label_column="label")
    X_test, y_test = load_data("data/test.csv", label_column="label")

    # Preprocess data using the tuned or default pipeline.
    X_train_trans = preprocessing_pipeline.fit_transform(X_train)
    X_test_trans = preprocessing_pipeline.transform(X_test)

    joblib.dump(preprocessing_pipeline, 'output/preprocessing_pipeline.pkl')
    print("Preprocessing pipeline saved to output/preprocessing_pipeline.pkl")

    # SVM expects labels in {-1, +1}.
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    model = build_model_svm(X_train_trans, y_train, tuned_hyperparams)

    # Evaluate on training data.
    train_preds = model.predict(X_train_trans)
    train_prec, train_rec, train_f1 = compute_metrics(y_train, train_preds)
    print("Training Metrics:")
    print(f"Precision: {train_prec:.3f}, Recall: {train_rec:.3f}, F1-score: {train_f1:.3f}")

    # Evaluate on test data.
    test_preds = model.predict(X_test_trans)
    if y_test is not None:
        test_prec, test_rec, test_f1 = compute_metrics(y_test, test_preds)
        print("Test Metrics:")
        print(f"Precision: {test_prec:.3f}, Recall: {test_rec:.3f}, F1-score: {test_f1:.3f}")
    else:
        print("Test labels not available; model ready for submission.")

    # Save the trained model.
    joblib.dump(model, 'output/svm_model.pkl')
    print("Model saved to output/svm_model.pkl")

if __name__ == '__main__':
    main()
