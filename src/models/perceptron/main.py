import argparse
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from common.data_loader import load_data
from common.evaluation import compute_metrics
from common.preprocessing import standardize_train, standardize_test, log_transform

from models.perceptron import Perceptron, AveragedPerceptron, MarginPerceptron

def convert_labels(y):
    """
    Convert labels from {0,1} to {-1,+1} for perceptron
    """
    return np.where(y ==0, -1, 1)

def train_val_split(X, y, test_size = 0.2, random_state = 42):
    """
    Function to split X and y into training and validation sets
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
    elif algo == 'avgperc':
        model = AveragedPerceptron(num_features=X_train.shape[1], lr=lr)
    elif algo == 'marginperc':
        mu = hyperparams.get("mu", 1.0)
        lr = hyperparams.get("lr", 1.0)
        model = MarginPerceptron(num_features=X_train.shape[1], lr=lr, mu=mu)
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
    parser.add_argument('--decay_lr', type=bool, default=False, help="decay learning rate parameter ")
    args = parser.parse_args()

    X_train_full, y_train_full = load_data("data/train.csv", label_column="label")
    X_test, y_test = load_data("data/test.csv", label_column="label")

    #Preprocessing data
    X_train_full = log_transform(X_train_full)
    X_train_full, train_mean, train_std = standardize_train(X_train_full)
    X_test = log_transform(X_test)
    X_test = standardize_test(X_test, train_mean, train_std)

    #For perceptron
    y_train_full_conv = convert_labels(y_train_full)

    #Splitting data into training and evaluation set
    X_train, X_val, y_train, y_val = train_val_split(X_train_full, y_train_full_conv, test_size=0.2)
    print(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")

    #set hyperparameters
    hyperparams = {"epochs": args.epochs, "lr": args.lr}
    if args.algo == 'perc':
        hyperparams["decay_lr"] = False
        hyperparams["mu"] = 0
    elif args.algo == 'marginperc':
        hyperparams["mu"] = args.mu
    
    #Training the model
    model = build_and_train_perceptron(X_train, y_train, hyperparams, args.algo)

    #Evaluation on held-out validation set
    val_preds = np.array(model.predict(X_val))

    #Converting predictions from {-1,1} to {0,1} for evaluation
    val_preds_conv = np.where(val_preds == -1, 0, 1)

    y_val_conv = np.where(y_val == -1, 0, y_val)

    val_prec, val_rec, val_f1 = compute_metrics(y_val_conv, val_preds_conv)
    print("Held-Out Validation Metrics:")
    print(f"Precision: {val_prec:.3f}, Recall: {val_rec:.3f}, F1-score: {val_f1:.3f}")


    #Retraining on full trainind data and evalating on test set
    model_full = build_and_train_perceptron(X_train_full, y_train_full_conv, hyperparams, args.algo)
    train_preds = np.array(model_full.predict(X_train_full))
    test_preds = np.array(model_full.predict(X_test))
    #print("Unique training predictions after full training:", np.unique(train_preds))

    #Evaluation
    train_preds_conv = np.where(np.array(train_preds) == -1, 0, 1)
    test_preds_conv = np.where(np.array(test_preds) == -1, 0, 1)
    # print("Unique converted training predictions:", np.unique(train_preds_conv))
    # print("Unique converted test predictions:", np.unique(test_preds_conv))

    train_prec, train_rec, train_f1 = compute_metrics(y_train_full, train_preds_conv)
    print("Final Training Metrics:")
    print(f"Precision: {train_prec:.3f}, Recall: {train_rec:.3f}, F1-score: {train_f1:.3f}")

    if y_test is not None:
        y_test_conv = np.where(y_test == -1, 0, y_test)
        test_prec, test_rec, test_f1 = compute_metrics(y_test_conv, test_preds_conv)
        print("Test Metrics:")
        print(f"Precision: {test_prec:.3f}, Recall: {test_rec:.3f}, F1-score: {test_f1:.3f}")

if __name__ == '__main__':
    main()