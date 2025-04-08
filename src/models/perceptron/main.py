import argparse
import numpy as np
import sys
import os
import joblib
from pathlib import Path

# Add project root to sys.path to import common modules.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common.data_loader import load_data
from common.evaluation import compute_metrics
from common.preprocessing import log_transform
from models.perceptron import Perceptron, AveragedPerceptron, MarginPerceptron

def convert_labels(y):
    """
    Convert labels from {0,1} to {-1,+1} for perceptron.
    """
    return np.where(y == 0, -1, 1)

def train_val_split(X, y, test_size=0.2, random_state=42):
    """
    Splits X and y into training and validation sets.
    """
    np.random.seed(random_state)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    split = int(len(y) * (1 - test_size))
    train_idx = indices[:split]
    val_idx = indices[split:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

def build_and_train_perceptron(X_train, y_train, hyperparams, algo):
    """
    Builds and trains a perceptron model based on the variant.
    """
    epochs = hyperparams.get("epochs", 10)
    lr = hyperparams.get("lr", 1.0)
    
    if algo == 'perc':
        decay_lr = hyperparams.get("decay_lr", False)
        mu = hyperparams.get("mu", 0)
        model = Perceptron(num_features=X_train.shape[1], lr=lr, decay_lr=decay_lr, mu=mu)
    elif algo == 'avgperc':
        model = AveragedPerceptron(num_features=X_train.shape[1], lr=lr)
    elif algo == 'marginperc':
        mu = hyperparams.get("mu", 1.0)
        model = MarginPerceptron(num_features=X_train.shape[1], lr=lr, mu=mu)
    else:
        raise ValueError("Unknown perceptron variant.")
    
    model.train(X_train, y_train, epochs=epochs)
    return model

def main():
    parser = argparse.ArgumentParser(description="Train/test perceptron models with hyperparameter tuning.")
    parser.add_argument('--algo', type=str, default='perc', choices=['perc', 'avgperc', 'marginperc'],
                        help="Select the perceptron variant: 'perc' (standard), 'avgperc' (averaged), or 'marginperc' (margin).")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=1.0, help="Learning rate.")
    parser.add_argument('--mu', type=float, default=1.0, help="Margin parameter.")
    parser.add_argument('--decay_lr', type=bool, default=False, help="Decay learning rate (for standard perceptron).")
    args = parser.parse_args()
    
    # Use the algorithm name to locate a tuned configuration.
    model_type = args.algo
    tuned_config_path = os.path.join("tuned_models", f"{model_type}_best_model_config.pkl")
    
    # If a tuned configuration exists, load the tuned preprocessing pipeline and hyperparameters.
    if os.path.exists(tuned_config_path):
        best_config = joblib.load(tuned_config_path)
        print(f"Loaded tuned configuration from {tuned_config_path}")
        preprocessing_pipeline = best_config["pipeline"]
        tuned_hyperparams = best_config["hyperparameters"]
    else:
        print(f"Tuned configuration for model '{model_type}' not found. Using default preprocessing.")
        preprocessing_pipeline = None
        tuned_hyperparams = {}
    
    # Override tuned hyperparameters with command-line parameters (if specified).
    if args.epochs is not None:
        tuned_hyperparams["epochs"] = args.epochs
    if args.lr is not None:
        tuned_hyperparams["lr"] = args.lr
    if args.mu is not None:
        tuned_hyperparams["mu"] = args.mu
    if args.algo == 'perc':
        tuned_hyperparams["decay_lr"] = args.decay_lr
    
    # Load full training and test data.
    X_train_full, y_train_full = load_data("data/train.csv", label_column="label")
    X_test, y_test = load_data("data/test.csv", label_column="label")
    
    # Preprocess data.
    if preprocessing_pipeline is not None:
        # Use the tuned pipeline (assumed to implement fit_transform / transform)
        X_train_full_trans = preprocessing_pipeline.fit_transform(X_train_full)
        X_test_trans = preprocessing_pipeline.transform(X_test)
    else:
        # Fallback: manually apply log transformation and standardization.
        raise ValueError("Pipeline not found")
    
    # Convert training labels for perceptron.
    y_train_full_conv = convert_labels(y_train_full)
    
    # Split a hold-out set from full training data for validation.
    X_train, X_val, y_train, y_val = train_val_split(X_train_full_trans, y_train_full_conv, test_size=0.2)
    print(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")
    
    # Set any missing hyperparameters to defaults if not provided by tuned config.
    if "epochs" not in tuned_hyperparams:
        tuned_hyperparams["epochs"] = args.epochs
    if "lr" not in tuned_hyperparams:
        tuned_hyperparams["lr"] = args.lr
    if args.algo == 'perc' and "mu" not in tuned_hyperparams:
        tuned_hyperparams["mu"] = 0  # standard perceptron
    elif args.algo in ['marginperc', 'avgperc'] and "mu" not in tuned_hyperparams:
        tuned_hyperparams["mu"] = args.mu
    
    # Train the model on the training split and evaluate on validation set.
    model = build_and_train_perceptron(X_train, y_train, tuned_hyperparams, args.algo)
    val_preds = np.array(model.predict(X_val))
    val_preds_conv = np.where(val_preds == -1, 0, 1)
    y_val_conv = np.where(y_val == -1, 0, y_val)
    
    val_prec, val_rec, val_f1 = compute_metrics(y_val_conv, val_preds_conv)
    print("Held-Out Validation Metrics:")
    print(f"Precision: {val_prec:.3f}, Recall: {val_rec:.3f}, F1-score: {val_f1:.3f}")
    
    # Retrain on the full training data.
    model_full = build_and_train_perceptron(X_train_full_trans, y_train_full_conv, tuned_hyperparams, args.algo)
    train_preds = np.array(model_full.predict(X_train_full_trans))
    test_preds = np.array(model_full.predict(X_test_trans))
    
    train_preds_conv = np.where(train_preds == -1, 0, 1)
    test_preds_conv = np.where(test_preds == -1, 0, 1)
    
    train_prec, train_rec, train_f1 = compute_metrics(y_train_full, train_preds_conv)
    print("Final Training Metrics:")
    print(f"Precision: {train_prec:.3f}, Recall: {train_rec:.3f}, F1-score: {train_f1:.3f}")
    
    if y_test is not None:
        # For evaluation on test data, assume test labels remain in {0,1} (or convert appropriately).
        y_test_conv = np.where(y_test == -1, 0, y_test)
        test_prec, test_rec, test_f1 = compute_metrics(y_test_conv, test_preds_conv)
        print("Test Metrics:")
        print(f"Precision: {test_prec:.3f}, Recall: {test_rec:.3f}, F1-score: {test_f1:.3f}")
    
    # If a tuned preprocessing pipeline was used, persist it.
    if preprocessing_pipeline is not None:
        joblib.dump(preprocessing_pipeline, 'output/preprocessing_pipeline.pkl')
        print("Tuned preprocessing pipeline saved to output/preprocessing_pipeline.pkl")

if __name__ == '__main__':
    main()
