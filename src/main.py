#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import joblib
import numpy as np

from sklearn.base            import clone
from sklearn.model_selection import train_test_split

# ─── Add project root so we can import our code ───────────────────────────────
PROJECT_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_SRC))
# ────────────────────────────────────────────────────────────────────────────────

from common.evaluation import compute_metrics
from common.data_loader import load_data
from models.nerual_network.nn import TorchWrapper
from models import (
    build_model_decision_tree,
    build_standard_perceptron,
    build_averaged_perceptron,
    build_margin_perceptron,
    build_ensemble_model,
    build_model_adaboost,
    build_model_svm,
    build_model_nn,
)

# Map model‐name → builder function
MODEL_BUILDERS = {
    "dt":       build_model_decision_tree,
    "perc":     build_standard_perceptron,
    "avgperc":  build_averaged_perceptron,
    "marginperc": build_margin_perceptron,
    "ensemble": build_ensemble_model,
    "adaboost": build_model_adaboost,
    "svm":      build_model_svm,
    "nn":       build_model_nn,
}

def find_optimal_threshold(
    wrapper: TorchWrapper,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> tuple[float, float]:
    """
    Sweep a grid of thresholds to maximize F1 on (X_val, y_val∈{0,1}).
    Returns (best_threshold, best_f1).
    """
    logits = wrapper.predict_logits(X_val)
    best_thr, best_f1 = 0.0, -1.0

    # 200 evenly spaced candidates
    thresholds = np.linspace(logits.min() - 1e-3,
                             logits.max() + 1e-3, 200)
    for t in thresholds:
        preds = (logits >= t).astype(int)
        _, _, f1 = compute_metrics(y_val, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    return best_thr, best_f1


def main():
    p = argparse.ArgumentParser(
        description="Retrain a tuned model on the full training set and save."
    )
    p.add_argument(
        "--model",
        required=True,
        choices=MODEL_BUILDERS.keys(),
        help="Which model to retrain (must match tune_models.py)"
    )
    p.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="(nn only) fraction of data held out for threshold tuning"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits & reproducibility"
    )
    args = p.parse_args()

    # 1) Load the tuned config
    tuned_path = Path("tuned_models") / f"{args.model}_best_model_config.pkl"
    if not tuned_path.exists():
        sys.stderr.write(f"ERROR: tuned config not found at {tuned_path}\n")
        sys.exit(1)

    cfg = joblib.load(tuned_path)
    pipeline_cfg = cfg["pipeline"]
    hyperparams  = cfg["hyperparameters"]
    print(f"\nLoaded tuned config '{args.model}':")
    print("  hyperparameters =", hyperparams)
    print("  pipeline        =", pipeline_cfg, "\n")

    # 2) Load & preprocess full training set
    X_full, y_full = load_data("data/train.csv", label_column="label")
    pipe = clone(pipeline_cfg)
    print("Fitting preprocessing on full training set…")
    X_full_t = pipe.fit_transform(X_full)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # 3a) Classical models
    if args.model != "nn":
        # convert labels for perceptrons, adaboost, svm
        if args.model in ["perc", "avgperc", "marginperc", "adaboost", "svm"]:
            y_train = np.where(y_full == 0, -1, 1)
        else:
            y_train = y_full

        builder = MODEL_BUILDERS[args.model]
        print(f"Training '{args.model}' on the full set…")
        model = builder(X_full_t, y_train, hyperparams)

        # save preprocessing + model
        joblib.dump(pipe,    out_dir / "preprocessing_pipeline.pkl")
        joblib.dump(model,   out_dir / f"{args.model}_model.pkl")
        print(f"\nSaved pipeline → {out_dir/'preprocessing_pipeline.pkl'}")
        print(f"Saved model    → {out_dir/f'{args.model}_model.pkl'}")

    # 3b) Neural network: threshold tuning, bundle everything
    else:
        # carve off a validation fold *after* transformation
        X_tr, X_val, y_tr0, y_val0 = train_test_split(
            X_full_t, y_full,
            test_size=args.val_size,
            stratify=y_full,
            random_state=args.seed
        )
        # convert to {-1, +1} for build_model_nn
        y_tr = np.where(y_tr0 == 0, -1, 1)
        hyperparams["random_state"] = args.seed

        print("Training neural network on train fold…")
        nn_wrapper = build_model_nn(X_tr, y_tr, hyperparams)

        print("Searching best threshold on validation fold…")
        thr, val_f1 = find_optimal_threshold(nn_wrapper, X_val, y_val0)
        print(f" → chosen threshold = {thr:.4f} (val F1 = {val_f1:.4f})\n")

        bundle = {
            "preprocessing_pipeline": pipe,
            "model_wrapper":          nn_wrapper,
            "optimal_threshold":      thr
        }
        joblib.dump(bundle, out_dir / "nn_model_with_threshold.pkl")
        print(f"Saved NN bundle → {out_dir/'nn_model_with_threshold.pkl'}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
