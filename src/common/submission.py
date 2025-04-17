# submission.py

import sys
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import joblib

# add project src to path
PROJECT_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_SRC))

from common.data_loader import load_data

def create_submission(model_type, model_file, pipeline_file=None):
    """
    For non‐NN models: load preprocessing pipeline and model separately.
    For nn: load the bundled nn_model_with_threshold.pkl which contains
    'preprocessing_pipeline', 'model_wrapper', and 'optimal_threshold'.
    """
    # load eval data
    X_eval, _ = load_data("data/eval.anon.csv", label_column="label")

    if model_type == "nn":
        # the nn bundle includes both pipeline + wrapper + threshold
        bundle = joblib.load(model_file)
        preprocessing_pipeline = bundle["preprocessing_pipeline"]
        model_wrapper         = bundle["model_wrapper"]
        threshold             = bundle["optimal_threshold"]

        print(f"Loaded NN bundle from {model_file}")
        X_eval_t = preprocessing_pipeline.transform(X_eval)

        # get 0/1 predictions at the tuned threshold
        eval_preds = model_wrapper.predict_submission(X_eval_t, threshold=threshold)

    else:
        # classic workflow: load pipeline then model
        if pipeline_file is None:
            print("Error: --pipeline_file must be specified for non‐NN models")
            sys.exit(1)

        preprocessing_pipeline = joblib.load(pipeline_file)
        print(f"Loaded pipeline from {pipeline_file}")
        X_eval_t = preprocessing_pipeline.transform(X_eval)

        model = joblib.load(model_file)
        print(f"Loaded model from {model_file}")

        if hasattr(model, "predict_submission"):
            eval_preds = model.predict_submission(X_eval_t)
        else:
            eval_preds = model.predict(X_eval_t)

    # read eval IDs and write submission.csv
    eval_ids = pd.read_csv("data/eval.id", header=None, names=["example_id"])
    submission = pd.DataFrame({
        "example_id": eval_ids["example_id"],
        "label": eval_preds
    })
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    submission_path = out_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate submission.csv from a tuned model."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["dt","perc","avgperc","marginperc","ensemble","adaboost","svm","nn"],
        help="Which model to use"
    )
    parser.add_argument(
        "--model_file", type=str,
        help="Path to tuned model file. "
             "For nn use output/nn_model_with_threshold.pkl; "
             "for others something like tuned_models/{model}_best_model_config.pkl"
    )
    parser.add_argument(
        "--pipeline_file", type=str, default=None,
        help="Preprocessing pipeline .pkl (not needed for nn)."
    )
    args = parser.parse_args()

    if args.model == "nn":
        if args.model_file is None:
            args.model_file = "output/nn_model_with_threshold.pkl"
    else:
        if args.model_file is None:
            args.model_file = f"tuned_models/{args.model}_best_model_config.pkl"
        if args.pipeline_file is None:
            args.pipeline_file = "output/preprocessing_pipeline.pkl"

    create_submission(args.model, args.model_file, args.pipeline_file)


if __name__ == "__main__":
    main()
