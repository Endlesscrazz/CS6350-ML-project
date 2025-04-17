# src/models/nerual_network/main.py

import argparse
import numpy as np
from pathlib import Path
import joblib
import os
import sys
import torch
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.base import clone
import matplotlib.pyplot as plt

# ─── fix imports ────────────────────────────────────────────────────────────────
# file is at: <repo>/src/models/nerual_network/main.py
SRC_ROOT = Path(__file__).resolve().parents[2]   # <repo>/src
REPO_ROOT = SRC_ROOT.parent                     # <repo>
sys.path.insert(0, str(SRC_ROOT))               # add src to python path
# ────────────────────────────────────────────────────────────────────────────────

from common.data_loader import load_data
from common.evaluation import compute_metrics
from models.nerual_network.nn import build_model_nn, TorchWrapper


def find_optimal_threshold(model_wrapper: TorchWrapper, X_val: np.ndarray, y_val_01: np.ndarray):
    """
    Finds the optimal prediction threshold on the validation set based on F1 score.
    """
    print("\nFinding optimal threshold on validation set...")
    val_logits = model_wrapper.predict_logits(X_val)

    best_f1 = -1.0
    best_threshold = 0.0
    thresholds = np.linspace(val_logits.min() - 1e-3, val_logits.max() + 1e-3, 200)

    if len(np.unique(y_val_01)) < 2:
        print("Warning: Validation set contains only one class. Cannot compute F1 reliably.")
        return 0.0, 0.0

    f1_scores = []
    for t in thresholds:
        preds = np.where(val_logits >= t, 1, 0)
        _, _, f1 = compute_metrics(y_val_01, preds)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    print(f"Optimal threshold: {best_threshold:.4f} with F1: {best_f1:.4f}")

    # debug plot
    plt.figure()
    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Threshold")
    plt.axvline(best_threshold, color='r', linestyle='--',
                label=f"best={best_threshold:.2f}")
    plt.legend()
    plt.grid(True)
    # plt.savefig(REPO_ROOT/"output"/"nn_threshold_f1.png")
    return best_threshold, best_f1


def main():
    parser = argparse.ArgumentParser(
        description="Train NN, find threshold, evaluate & save."
    )
    parser.add_argument('--lr', type=float, help="override learning rate")
    parser.add_argument('--epochs', type=int, help="override epochs")
    parser.add_argument('--batch_size', type=int, help="override batch size")
    parser.add_argument('--dropout', type=float, help="override dropout")
    parser.add_argument('--weight_decay', type=float, help="override weight decay")
    parser.add_argument('--val_size', type=float, default=0.2,
                        help="validation split fraction")
    args = parser.parse_args()

    # load tuned config
    model_type = "nn"
    tuned_path = REPO_ROOT / "tuned_models" / f"{model_type}_best_model_config.pkl"
    if not tuned_path.exists():
        raise FileNotFoundError(
            f"Tuned config not found at {tuned_path}. Run tune_models.py first."
        )
    print(f"Loading tuned config from {tuned_path}")
    best_cfg = joblib.load(tuned_path)
    pipeline_cfg = best_cfg["pipeline"]
    hyperparams  = best_cfg["hyperparameters"]
    print("Hyperparams:", hyperparams)
    print("Pipeline:", pipeline_cfg)

    # allow overrides
    for k in ("lr","epochs","batch_size","dropout","weight_decay"):
        v = getattr(args, k)
        if v is not None:
            hyperparams[k] = v

    # load data
    X_train_full, y_train_full = load_data(REPO_ROOT/"data"/"train.csv",  label_column="label")
    X_test,       y_test       = load_data(REPO_ROOT/"data"/"test.csv",   label_column="label")

    # preprocess
    preprocessing = clone(pipeline_cfg)
    print("\nFitting pipeline on full training set…")
    X_train_full_t = preprocessing.fit_transform(X_train_full)
    X_test_t       = preprocessing.transform(X_test)
    print("Train shape:", X_train_full_t.shape, "Test shape:", X_test_t.shape)

    # split for threshold tuning
    seed = hyperparams.get("random_state", 42) + 10
    X_train, X_val, y_train01, y_val01 = train_test_split(
        X_train_full_t, y_train_full,
        test_size=args.val_size,
        stratify=y_train_full,
        random_state=seed
    )

    # convert labels for build_model_nn
    y_train_m1p1 = np.where(y_train01 == 0, -1, 1)
    hyperparams["random_state"] = seed + 1

    # train
    print("\nTraining model…")
    wrapper = build_model_nn(X_train, y_train_m1p1, hyperparams)

    # threshold
    thr, val_f1 = find_optimal_threshold(wrapper, X_val, y_val01)

    # eval on train fold
    print("\nEvaluation on training fold:")
    train_preds = wrapper.predict_submission(X_train, threshold=thr)
    p_tr, r_tr, f_tr = compute_metrics(y_train01, train_preds)
    print(f"  P {p_tr:.3f} R {r_tr:.3f} F1 {f_tr:.3f}")

    # eval on val fold
    print("\nEvaluation on validation fold:")
    val_preds = wrapper.predict_submission(X_val, threshold=thr)
    p_v, r_v, f_v = compute_metrics(y_val01, val_preds)
    print(f"  P {p_v:.3f} R {r_v:.3f} F1 {f_v:.3f}")

    # eval on test set
    print("\nEvaluation on test set:")
    test_preds = wrapper.predict_submission(X_test_t, threshold=thr)
    p_te, r_te, f_te = compute_metrics(y_test, test_preds)
    print(f"  P {p_te:.3f} R {r_te:.3f} F1 {f_te:.3f}")

    # save final model + pipeline + threshold
    outdir = REPO_ROOT / "output"
    outdir.mkdir(exist_ok=True)
    save_obj = {
        "model_wrapper": wrapper,
        "preprocessing_pipeline": preprocessing,
        "optimal_threshold": thr,
        "val_f1": val_f1,
        "test_f1": f_te,
    }
    joblib.dump(save_obj, outdir/"nn_model_with_threshold.pkl")
    print(f"\nSaved to {outdir/'nn_model_with_threshold.pkl'}")

if __name__ == '__main__':
    main()
