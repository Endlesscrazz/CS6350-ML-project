#!/usr/bin/env python3
import argparse
import numpy as np
import sys
import random
import os
from pathlib import Path
from joblib import Parallel, delayed, dump
from sklearn.base import clone

# Append project source directory.
project_src = Path(__file__).resolve().parents[1]
sys.path.append(str(project_src))

from common.data_loader import load_data
from common.cross_validation import grid_search_cv_generic
from common.preprocessing import preprocessing_pipelines
from models import (
    build_model_decision_tree,
    build_standard_perceptron,
    build_averaged_perceptron,
    build_margin_perceptron,
    build_ensemble_model,
    build_model_adaboost,
    build_model_svm,
    build_model_nn
)

def identity_conversion(labels):
    return labels

def convert_neg1_to0(labels):
    return np.where(labels == -1, 0, labels)

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

def evaluate_params(params, X, y, builder, k, label_conv):
    """Return (params, avg_f1) via CV on pre‐transformed X."""
    try:
        _, f1 = grid_search_cv_generic(
            X, y, builder, [params],
            k=k, label_conversion=label_conv, stratified=True
        )
    except Exception as e:
        print(f"  ↳ params {params} failed: {e}")
        f1 = -np.inf
    return params, f1

def main():
    p = argparse.ArgumentParser(description="Hyperparameter tuning w/ pipeline caching")
    p.add_argument('--model',  choices=MODEL_BUILDERS.keys(), required=True)
    p.add_argument('--k',      type=int, default=3,    help="CV folds")
    p.add_argument('--n_iter', type=int, default=20,   help="Max hyperparam trials")
    p.add_argument('--seed',   type=int, default=42,   help="Random seed")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1) Load all training data (for k‐fold CV)
    X_train, y_train = load_data("data/train.csv", label_column="label")

    # 2) Convert labels for certain models
    if args.model in ['perc','avgperc','marginperc','adaboost','svm','nn']:
        y_train = np.where(y_train == 0, -1, 1)
        label_conv = convert_neg1_to0
    else:
        label_conv = identity_conversion

    # 3) Builder
    builder = MODEL_BUILDERS[args.model]

    # 4) Hyperparameter grids
    if args.model == 'dt':
        hyper_grid = [
            {"max_depth": d, "min_samples_split": s}
            for d in [5,10] for s in [2,5,10]
        ]
    elif args.model == 'perc':
        hyper_grid = [
            {"epochs": e, "lr": lr, "decay_lr": d, "mu": 0}
            for e in [10,20,30]
            for lr in [0.1,0.5,1.0]
            for d in [False,True]
        ]
    elif args.model == 'avgperc':
        hyper_grid = [
            {"epochs": e, "lr": lr}
            for e in [10,20,30]
            for lr in [0.1,0.5,1.0]
        ]
    elif args.model == 'marginperc':
        hyper_grid = [
            {"epochs": e, "lr": lr, "mu": mu}
            for e in [10,20,30]
            for lr in [0.1,0.5,1.0]
            for mu in [0.8,1.0,1.2]
        ]
    elif args.model == 'ensemble':
        hyper_grid = [
            {
                "dt_params":   {"max_depth": d, "min_samples_split": s},
                "perc_params": {"epochs": e, "lr": lr, "decay_lr": decay, "mu": mu},
                "w_dt": w_dt, "w_perc": 1-w_dt
            }
            for d in [5,10]
            for s in [2,5]
            for e in [10,20]
            for lr in [0.1,0.5]
            for decay in [False,True]
            for w_dt in [0.5,0.6]
            for mu in [0.8,1.0]
        ]
    elif args.model == 'adaboost':
        hyper_grid = [
            {"n_estimators": n, "n_thresholds": t, "weak_learner_depth": d}
            for n in [20,50,100]
            for t in [5,10,20]
            for d in [2,3]
        ]
    elif args.model == 'svm':
        hyper_grid = [
            {"lr": lr, "lambda_param": lp, "n_epochs": n, "batch_size": bs}
            for lr in [0.001,0.01,0.1]
            for lp in [0.001,0.01,0.1]
            for n in [500,1000]
            for bs in [32,64]
        ]
    elif args.model == 'nn':
        hyper_grid = [
            {
                "hidden_dims": hd,
                "lr": lr,
                "epochs": e,
                "batch_size": bs,
                "dropout": do,
                "optimizer": opt,
                "weight_decay": wd,
                "random_state": args.seed,
                "early_stopping_patience": 15
            }
            for hd in [(64,32),(128,64),(256,128)]
            for lr in [0.003,0.005,0.007]
            for e in [100,150]
            for bs in [32,64]
            for do in [0.2,0.3]
            for opt in ["adam","adamw"]
            for wd in [1e-5,1e-4]
        ]
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # 5) Random subsample if too many
    if len(hyper_grid) > args.n_iter:
        hyper_grid = random.sample(hyper_grid, args.n_iter)
        print(f"→ Random search: {len(hyper_grid)} hyperparam sets")

    # 6) Prefit & cache each pipeline variant ONCE
    X_cache = {}
    candidate_pca = [50, 100]
    for name, pipe in preprocessing_pipelines.items():
        if 'pca' in pipe.named_steps:
            for nc in candidate_pca:
                p = clone(pipe)
                p.set_params(pca__n_components=nc)
                print(f"Fitting '{name}' with PCA={nc}…")
                X_cache[(name,nc)] = p.fit_transform(X_train)
        else:
            p = clone(pipe)
            print(f"Fitting '{name}' (no PCA)…")
            X_cache[(name,None)] = p.fit_transform(X_train)

    # 7) Build evaluation tasks
    tasks = [(n,c,params) for (n,c) in X_cache for params in hyper_grid]

    # 8) Parallel evaluate
    print(f"Running {len(tasks)} jobs in parallel…")
    def task(nc_tuple, params):
        n, nc = nc_tuple
        return n, nc, *evaluate_params(params, X_cache[(n,nc)], y_train, builder, args.k, label_conv)

    results = Parallel(n_jobs=-1)(
        delayed(task)((n, nc), params)
        for n,nc,params in tasks
    )

    # 9) Pick best per‐pipeline
    best_pipe = {}
    for name,nc,params,f1 in results:
        key=(name,nc)
        if key not in best_pipe or f1>best_pipe[key][1]:
            best_pipe[key]=(params,f1)

    # 10) Report & save
    best_score=-np.inf
    for (name,nc),(params,f1) in best_pipe.items():
        print(f"→ {name} PCA={nc or 'raw'} → F1={f1:.3f}, params={params}")
        if f1>best_score:
            best_score, best_name, best_nc, best_params = f1,name,nc,params

    print(f"\nBEST: pipeline={best_name}, PCA={best_nc}, F1={best_score:.4f}, params={best_params}")
    os.makedirs("tuned_models",exist_ok=True)
    tuned = {
        "model": args.model,
        "pipeline_name": best_name,
        "pca_n_components": best_nc,
        "pipeline": clone(preprocessing_pipelines[best_name])
                       .set_params(**({"pca__n_components":best_nc} if best_nc else {})),
        "hyperparameters": best_params,
        "f1_score": best_score
    }
    dump(tuned, Path("tuned_models")/f"{args.model}_best_model_config.pkl")
    print("Saved tuned config.")

if __name__=='__main__':
    main()
