#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import joblib
import pandas as pd

# make sure we can import common/
PROJECT_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_SRC))

from common.data_loader import load_data

def create_submission(model_type, model_file, pipeline_file=None):
    # 1) load eval features
    X_eval, _ = load_data("data/eval.anon.csv", label_column="label")

    if model_type == "nn":
        # --- NN bundle: pipeline + wrapper + threshold ---
        bundle = joblib.load(model_file)
        print(f"[submission] loaded NN bundle from {model_file}")
        pipe    = bundle["preprocessing_pipeline"]
        wrapper = bundle["model_wrapper"]
        thr     = bundle.get("optimal_threshold", 0.0)

        X_t = pipe.transform(X_eval)
        preds = wrapper.predict_submission(X_t, threshold=thr)

    else:
        # --- classical models: need both pipeline and model separately ---
        if pipeline_file is None:
            sys.stderr.write("ERROR: --pipeline_file is required for non‑NN models\n")
            sys.exit(1)

        pipe = joblib.load(pipeline_file)
        print(f"[submission] loaded pipeline from {pipeline_file}")
        X_t = pipe.transform(X_eval)

        model = joblib.load(model_file)
        print(f"[submission] loaded model    from {model_file}")
        if hasattr(model, "predict_submission"):
            preds = model.predict_submission(X_t)
        else:
            preds = model.predict(X_t)

    # assemble submission dataframe
    ids = pd.read_csv("data/eval.id", header=None, names=["example_id"])
    submission = pd.DataFrame({
        "example_id": ids["example_id"],
        "label": preds
    })

    out = Path("output"); out.mkdir(exist_ok=True)
    submission_path = out / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"[submission] wrote {submission_path}")


def main():
    p = argparse.ArgumentParser(description="Generate Kaggle submission from trained model.")
    p.add_argument('--model', required=True,
                   choices=["dt","perc","avgperc","marginperc","ensemble","adaboost","svm","nn"],
                   help="Which model to use")
    p.add_argument('--model_file', help="Path to the saved model file (or NN bundle).")
    p.add_argument('--pipeline_file', help="Path to preprocessing pipeline (not used for NN).")
    args = p.parse_args()

    # set sensible defaults
    if not args.model_file:
        if args.model == "nn":
            args.model_file = "output/nn_model_with_threshold.pkl"
        else:
            args.model_file = f"output/{args.model}_model.pkl"

    # only non-NN need a pipeline file
    if args.model != "nn" and not args.pipeline_file:
        args.pipeline_file = "output/preprocessing_pipeline.pkl"

    create_submission(args.model, args.model_file, args.pipeline_file)


if __name__ == "__main__":
    main()
