import argparse
import numpy as np
from pathlib import Path
import joblib
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common.data_loader import load_data
from common.evaluation import compute_metrics
from common.preprocessing import preprocessing_pipeline  
from models.svm import build_model_svm

def main():
    parser = argparse.ArgumentParser(description="Train SVM model and evaluate performance.")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for SVM.")
    parser.add_argument('--lambda_param', type=float, default=0.01, help="Regularization strength.")
    parser.add_argument('--n_epochs', type=int, default=1000, help="Number of epochs.")
    args = parser.parse_args()

    # Load data.
    X_train, y_train = load_data("data/train.csv", label_column="label")
    X_test, y_test = load_data("data/test.csv", label_column="label")

    # Preprocess data 
    X_train_trans = preprocessing_pipeline.fit_transform(X_train)
    X_test_trans = preprocessing_pipeline.transform(X_test)

    #Persist pipeline
    joblib.dump(preprocessing_pipeline, 'output/preprocessing_pipeline.pkl')

    # SVM expects labels in {-1, +1}
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    hyperparams = {
        "lr": args.lr,
        "lamda_param": args.lambda_param,
        "n_epochs": args.n_epochs

    }

    model = build_model_svm(X_train_trans, y_train, hyperparams)

    train_preds = model.predict(X_train_trans)
    train_prec, train_rec, train_f1 = compute_metrics(y_train, train_preds)
    print("Training Metrics:")
    print(f"Precision: {train_prec:.3f}, Recall: {train_rec:.3f}, F1-score: {train_f1:.3f}")

    test_preds = model.predict(X_test_trans)
    if y_test is not None:
        test_prec, test_rec, test_f1 = compute_metrics(y_test, test_preds)
        print("Test Metrics:")
        print(f"Precision: {test_prec:.3f}, Recall: {test_rec:.3f}, F1-score: {test_f1:.3f}")
    else:
        print("Test labels not available; model ready for submission.")

    joblib.dump(model, 'output/svm_model.pkl')
    print("Model saved to output/svm_model.pkl")

if __name__ == '__main__':
    main()


