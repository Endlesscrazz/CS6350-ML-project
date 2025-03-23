import sys
import argparse
import pandas as pd
from pathlib import Path
import numpy as np

project_src = Path(__file__).resolve().parents[1]
sys.path.append(str(project_src))

from common.data_loader import load_data
from common.preprocessing import log_transform, standardize_train, standardize_test

from models.decision_tree.builders import build_model_decision_tree
from models.perceptron.builders import build_standard_perceptron, build_averaged_perceptron, build_margin_perceptron
from models.ensemble.ensemble_dt_perc import ensemble_predict

model_builders = {
    "dt": build_model_decision_tree,
    "perc":build_standard_perceptron,
    "avgperc": build_averaged_perceptron,
    "marginperc": build_margin_perceptron
}

def create_submission(model_name, hyperparams):
    
    X_eval, _ = load_data("data/eval.anon.csv", label_column="label")

    X_train, y_train = load_data("data/train.csv",label_column="label")

    #Preprocessing training data
    X_train = log_transform(X_train)
    X_train, train_mean, train_std = standardize_train(X_train)

    # Preprocess evaluation data using the same parameters as training.
    X_eval = log_transform(X_eval)
    X_eval = standardize_test(X_eval, train_mean, train_std)

    if model_name in ["perc", "avgperc", "marginperc"]:
        y_train = (y_train == 1) * 1 + (y_train==0) * -1
    

    if model_name != "ensemble":
        builder = model_builders.get(model_name)
        if builder is None:
            raise ValueError("Unkown model type.")
        model = builder(X_train, y_train, hyperparams)
        eval_preds = model.predict(X_eval)
        if model_name in ["perc", "avgperc", "marginperc"]:
            eval_preds = [0 if pred == -1 else 1 for pred in eval_preds]
    else:
        # Ensemble model using dt and marginperc

        hyperparams_dt = {
            "max_depth": hyperparams.get("max_depth", 10),
            "min_samples_split": hyperparams.get("min_samples_split", 5)
        }
        hyperparams_perc = {
            "epochs": hyperparams.get("epochs", 10),
            "lr": hyperparams.get("lr", 1.0),
            "decay_lr": hyperparams.get("decay_lr", False),
            "mu": hyperparams.get("mu", 0)
        }
        dt_model = build_model_decision_tree(X_train, y_train, hyperparams_dt)
        
        perc_model = build_margin_perceptron(X_train, y_train, hyperparams_perc)
        dt_preds = dt_model.predict(X_eval)  
        perc_preds = perc_model.predict(X_eval)  # in {-1,1}
        perc_preds = np.where(np.array(perc_preds) == -1, 0, perc_preds)
        # Combining predictions
        eval_preds = ensemble_predict([dt_preds, perc_preds], weights=[0.6, 0.4])
    
    eval_ids = pd.read_csv("data/eval.id", header=None, names=["example_id"])

    submission = pd.DataFrame({
        "example_id": eval_ids["example_id"],
        "label": eval_preds
    })

    #Saving as csv
    submission.to_csv("output/submission.csv", index=False)
    print("Submission file saved to output/submission.csv")

def main():
    parser = argparse.ArgumentParser(description="Generate submission file for selected model.")
    parser.add_argument('--model', type=str, default='dt', choices=['dt', 'perc', 'avgperc', 'marginperc', 'ensemble'],
                        help="Select model: 'dt' for decision tree, 'perc' for standard perceptron")
    # You can pass hyperparameters as command-line arguments.
    parser.add_argument('--max_depth', type=int, default=10, help="(For decision tree) Maximum depth.")
    parser.add_argument('--min_samples_split', type=int, default=5, help="(For decision tree) Minimum samples to split.")
    parser.add_argument('--epochs', type=int, default=10, help="(For perceptron) Number of epochs.")
    parser.add_argument('--lr', type=float, default=1.0, help="(For perceptron) Learning rate.")
    parser.add_argument('--decay_lr', action='store_true', help="(For perceptron) Use learning rate decay.")
    parser.add_argument('--mu', type=float, default=0.0, help="(For perceptron) Margin parameter.")
    args = parser.parse_args()

    if args.model == "dt":
        hyperparams = {
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split
        }
    elif args.model == "perc":
        hyperparams = {
            "epochs": args.epochs,
            "lr": args.lr,
            "decay_lr": args.decay_lr,
            "mu": args.mu
        }
    elif args.model == "avgperc":
        hyperparams = {
            "epochs": args.epochs,
            "lr": args.lr
        }
    elif args.model == "marginperc":
        hyperparams = {
            "epochs": args.epochs,
            "lr": args.lr,
            "mu": args.mu
        }
    elif args.model == "ensemble":
        hyperparams = {
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "epochs": args.epochs,
            "lr": args.lr,
            "decay_lr": args.decay_lr,
            "mu": args.mu
        }
    else:
        hyperparams = {}
    
    create_submission(args.model, hyperparams)

if __name__ == '__main__':
    main()
