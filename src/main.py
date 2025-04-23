#!/usr/bin/env python3
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.base import clone
from common.data_loader import load_data
from common.evaluation import compute_metrics
from common.preprocessing import preprocessing_pipelines
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
from common.plots import plot_model_learning_curve

MODEL_BUILDERS = {
    "dt": build_model_decision_tree,
    "perc": build_standard_perceptron,
    "avgperc": build_averaged_perceptron,
    "marginperc": build_margin_perceptron,
    "ensemble": build_ensemble_model,
    "adaboost": build_model_adaboost,
    "svm": build_model_svm,
    "nn": build_model_nn,
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=MODEL_BUILDERS.keys())
    p.add_argument('--plot_lr_curve', action='store_true',
                   help="Plot learning curve after training")
    p.add_argument('--epochs', type=int, default=None,
                   help="(NN only) override epochs for learning curve")
    args = p.parse_args()

    # 1) load tuned config
    tuned = joblib.load(Path("tuned_models")/f"{args.model}_best_model_config.pkl")
    pipeline_cfg = tuned["pipeline"]
    hyperparams  = tuned["hyperparameters"]

    # 2) load full train/test
    X_full, y_full = load_data("data/train.csv", label_column="label")
    X_test,  y_test = load_data("data/test.csv",  label_column="label")

    # 3) preprocess
    pipe = clone(pipeline_cfg)
    X_full_t = pipe.fit_transform(X_full)
    X_test_t = pipe.transform(X_test)

    # 4) retrain final model (without val‐split for classical)
    builder = MODEL_BUILDERS[args.model]
    if args.model == 'nn':

        y_tr = np.where(y_full==0, -1, 1)
        wrapper = builder(X_full_t, y_tr, hyperparams)
        preds = wrapper.predict_submission(X_test_t, threshold=tuned.get("optimal_threshold",0.0))
        p, r, f1 = compute_metrics(y_test, preds)
    else:
        # classical
        y_tr = np.where(y_full==0, -1, 1) if args.model in ['marginperc','svm','adaboost'] else y_full
        model = builder(X_full_t, y_tr, hyperparams)
        preds_raw = (model.predict_submission(X_test_t)
                 if hasattr(model,'predict_submission')
                 else model.predict(X_test_t))
        
        preds_array = np.array(preds_raw) # Ensure it's a numpy array
        print(f"Debug: Unique raw predictions: {np.unique(preds_array)}") #  debug print
        preds_binary = np.where(preds_array <= 0, 0, 1)
        print(f"Debug: Unique binary predictions: {np.unique(preds_binary)}") #  debug print


        print("Calculating metrics...")

        print(f"Debug: Unique y_test: {np.unique(y_test)}") # Optional debug print
        p, r, f1 = compute_metrics(y_test, preds_binary) # Use preds_binary here
        #p, r, f1 = compute_metrics(y_test, preds)

    print(f"Test F1 for {args.model}: {f1:.4f}")

    # 5) optionally plot learning curve
    if args.plot_lr_curve:
        plot_model_learning_curve(
            model_builder=builder,
            hyperparams=hyperparams,
            pipeline=pipe,
            X=X_full, y=y_full,
            model_type=args.model,
            epochs=args.epochs
        )

if __name__ == '__main__':
    main()
