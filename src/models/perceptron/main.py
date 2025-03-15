import argparse
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common.data_loader import load_data
from common.evaluation import compute_metrics
from common.cross_validation import grid_search_cv_generic, cross_validations_generic

from models.perceptron import Perceptron

def convert_labels(y):
    """
    Convert labels from {0,1} to {-1,+1} for perceptron
    """
    return np.where(y ==0, -1, 1)

def build_and_train_perceptron(X_train, y_train, hyperparams, algo):
    """
    Builds and trains a perceptron model based on the specified variant.
    
    Parameters:
      X_train, y_train: Training data.
      hyperparams (dict): Dictionary with hyperparameters (e.g., epochs, lr, decay_lr, mu).
      algo (str): Which perceptron variant to use: 'perc' (standard), 'avgperc' (averaged), or 'marginperc' (margin).
      
    Returns:
      A trained model instance (with a predict() method).
    """
    epochs = hyperparams.get("epochs", 10)
    lr = hyperparams.get("lr", 1.0)
    
    if algo == 'perc':
        decay_lr = hyperparams.get("decay_lr", False)
        mu = hyperparams.get("mu", 0)
        model = Perceptron(num_features=X_train.shape[1], lr=lr, decay_lr=decay_lr, mu=mu)
    # elif algo == 'avgperc':
    #     model = AveragedPerceptron(num_features=X_train.shape[1], lr=lr)
    # elif algo == 'marginperc':
    #     mu = hyperparams.get("mu", 1.0)
    #     model = MarginPerceptron(num_features=X_train.shape[1], mu=mu)
    else:
        raise ValueError("Unknown perceptron variant.")
    
    model.train(X_train, y_train, epochs=epochs)
    return model

def main():
    parser = argparse.ArgumentParser(description="Train/test perceptron models with cross-validation and hyperparameter tuning.")
    parser.add_argument('--algo', type=str, default='perc', choices=['perc', 'avgperc', 'marginperc'],
                        help="Select the perceptron variant: 'perc' (standard), 'avgperc' (averaged), or 'marginperc' (margin/aggressive).")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=1.0, help="Learning rate (for standard and averaged perceptron).")
    parser.add_argument('--mu', type=float, default=1.0, help="Margin parameter (for margin perceptron, or optionally standard if used).")
    args = parser.parse_args()

    X_train, y_train = load_data("data/train.csv", label_column="label")
    X_test, y_test = load_data("data/test.csv", label_column="label")

    #For perceptron
    y_train_conv = convert_labels(y_train)
    y_test_conv = convert_labels(y_test)

    best_params = {"epochs": args.epochs, "lr": args.lr}
    if args.algo == 'perc':
        best_params["decay_lr"] = False
        best_params["mu"] = 0
    elif args.algo == 'marginperc':
        best_params["mu"] = args.mu
    
    #Training the model
    model = build_and_train_perceptron(X_train, y_train, best_params, args.algo)
    
    #Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    #Evaluation
    train_preds_conv = np.where(np.array(train_preds) == -1, 0, 1)
    test_preds_conv = np.where(np.array(test_preds) == -1, 0, 1)

    train_prec, train_rec, train_f1 = compute_metrics(y_train, train_preds_conv)
    print("Training Metrics:")
    print(f"Precision: {train_prec:.3f}, Recall: {train_rec:.3f}, F1-score: {train_f1:.3f}")

    if y_test is not None:
        test_prec, test_rec, test_f1 = compute_metrics(y_test, test_preds_conv)
        print("Test Metrics:")
        print(f"Precision: {test_prec:.3f}, Recall: {test_rec:.3f}, F1-score: {test_f1:.3f}")

if __name__ == '__main__':
    main()