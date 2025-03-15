import argparse
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from common.data_loader import load_data
from common.evaluation import compute_metrics
from models.decision_tree.decision_tree import build_tree, predict
from models.perceptron.standard import Perceptron

def main():
    parser = argparse.ArgumentParser(description="Run a selected model with cross-validation and hyperparameter tuning.")
    parser.add_argument('--model', type=str, default='dt', choices=['dt', 'perc', 'nn'],
                        help="Select the model family: 'dt' for decision tree, 'perc' for perceptron, 'nn' for neural network")
    parser.add_argument('--tune', action='store_true', help="Perform hyperparameter tuning.")
    # Common hyperparameters
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs (if applicable).")
    parser.add_argument('--lr', type=float, default=1.0, help="Learning rate (if applicable).")
    args = parser.parse_args()

    #Loading data
    X_train, y_train = load_data("data/train.csv", label_column="label")
    X_test, y_test = load_data("data/test.csv", label_column="label")

    if args.model == 'dt':
        
        model = build_tree(X_train, y_train, max_depth=10, min_samples_split=5)
        train_preds = predict(model, X_train)
        test_preds = predict(model, X_test)
    elif args.model == 'perc':
        
        model = Perceptron(num_features=X_train.shape[1], lr=args.lr, decay_lr=False, mu=0)
        
        y_train_conv = np.where(y_train == 0, -1, 1)
        y_test_conv = np.where(y_test == 0, -1, 1)

        model.train(X_train, y_train_conv, epochs=args.epochs)
        train_preds = np.array(model.predict(X_train))
        test_preds = np.array(model.predict(X_test))
        
        # Convert predictions back to {0,1}
        train_preds = np.where(train_preds == -1, 0, 1)
        test_preds = np.where(test_preds == -1, 0, 1)
    elif args.model == 'nn':
        # You could add your neural network code here.
        raise NotImplementedError("Neural network not implemented yet.")
    else:
        raise ValueError("Unknown model selected.")

    # Evaluate performance.
    train_prec, train_rec, train_f1 = compute_metrics(y_train, train_preds)
    print(f"Training Metrics -> Precision: {train_prec:.3f}, Recall: {train_rec:.3f}, F1: {train_f1:.3f}")
    if y_test is not None:
        test_prec, test_rec, test_f1 = compute_metrics(y_test, test_preds)
        print(f"Test Metrics -> Precision: {test_prec:.3f}, Recall: {test_rec:.3f}, F1: {test_f1:.3f}")

if __name__ == '__main__':
    main()
