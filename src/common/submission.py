import sys
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import joblib


project_src = Path(__file__).resolve().parents[1]
sys.path.append(str(project_src))

from common.data_loader import load_data
from common.preprocessing import log_transform, standardize_train, standardize_test

def create_submission(model_file):
    
    X_eval, _ = load_data("data/eval.anon.csv", label_column="label")
    X_train, _ = load_data("data/train.csv", label_column="label")
    
    # Preprocessing training data 
    X_train = log_transform(X_train)
    X_train, train_mean, train_std = standardize_train(X_train)
    
    # Preprocessing evaluation data using the same parameters.
    X_eval = log_transform(X_eval)
    X_eval = standardize_test(X_eval, train_mean, train_std)
    
    try:
        model = joblib.load(model_file)
        print(f"Loaded pre-trained model from {model_file}")
    except Exception as e:
        print(f"Failed to load the model from {model_file}: {e}")
        sys.exit(1)
    
    eval_preds = model.predict(X_eval)
    
    eval_ids = pd.read_csv("data/eval.id", header=None, names=["example_id"])
    submission = pd.DataFrame({
        "example_id": eval_ids["example_id"],
        "label": eval_preds
    })
    
    submission.to_csv("output/submission.csv", index=False)
    print("Submission file saved to output/submission.csv")

def main():
    parser = argparse.ArgumentParser(description="Generate submission file using a pre-trained model.")
    parser.add_argument('--model', type=str, default='dt', choices=['dt', 'perc', 'avgperc', 'marginperc', 'ensemble'],
                        help="Select model type: 'dt', 'perc', 'avgperc', 'marginperc', or 'ensemble'")
    parser.add_argument('--model_file', type=str, default=None,
                        help="Path to the pre-trained model file. Defaults to output/best_model_{model}.pkl if not specified.")
    args = parser.parse_args()
    
    # If no model_file is provided, use a default based on the model type.
    if args.model_file is None:
        args.model_file = f"output/best_model_{args.model}.pkl"
    
    create_submission(args.model_file)

if __name__ == '__main__':
    main()
