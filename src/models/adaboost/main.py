# models/adaboost/main.py
import argparse
import numpy as np
from pathlib import Path
import joblib

# Add project src to sys.path to import common modules
import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common.data_loader import load_data
from common.evaluation import compute_metrics
from common.preprocessing import log_transform, standardize_train, standardize_test, remove_low_variance_features, apply_truncated_svd, apply_pca
from models.adaboost.builders import build_model_adaboost

def main():
    parser = argparse.ArgumentParser(description="Train AdaBoost model and evaluate performance.")
    parser.add_argument('--n_estimators', type=int, default=50, help="Number of boosting rounds.")
    parser.add_argument('--n_thresholds', type=int, default=20, help="Number of threshold.")
    #parser.add_argument('--w_method', type=str, default='uniform', help="type of weight initiliazation")
    parser.add_argument('--n_weak_learner_d', type=int, default=1, help="depth of weak learner")

    args = parser.parse_args()

    # Load data (assuming same format as your other models)
    X_train, y_train = load_data("data/train.csv", label_column="label")
    X_test, y_test = load_data("data/test.csv", label_column="label")

    # Preprocess data
    X_train = log_transform(X_train)
    X_train, mean, std = standardize_train(X_train)
    X_test = log_transform(X_test)
    X_test = standardize_test(X_test, mean, std)

    # Feature engineering 
    X_train, X_test, _ = remove_low_variance_features(X_train, X_test, threshold=1e-4)
   #X_train, X_test, _ = apply_truncated_svd(X_train, X_test, n_components=50)
    X_train, X_test, _ = apply_pca(X_train, X_test, n_components=50)

    # Convert labels for AdaBoost: typically AdaBoost expects {-1, +1}
    y_train = np.where(y_train == 0, -1, 1)

    hyperparams = {"n_estimators": args.n_estimators}
    model = build_model_adaboost(X_train, y_train, hyperparams)

    train_preds = model.predict(X_train)
    train_prec, train_rec, train_f1 = compute_metrics(y_train, train_preds)
    print("Training Metrics:")
    print(f"Precision: {train_prec:.3f}, Recall: {train_rec:.3f}, F1-score: {train_f1:.3f}")

    test_preds = model.predict(X_test)
    if y_test is not None:
        # Convert test labels similarly if needed
        y_test_conv = np.where(y_test == 0, -1, 1)
        test_prec, test_rec, test_f1 = compute_metrics(y_test_conv, test_preds)
        print("Test Metrics:")
        print(f"Precision: {test_prec:.3f}, Recall: {test_rec:.3f}, F1-score: {test_f1:.3f}")
    else:
        print("Test labels not available; model ready for submission.")

    joblib.dump(model, 'output/adaboost_model.pkl')
    print("Model saved to output/adaboost_model.pkl")

if __name__ == '__main__':
    main()
