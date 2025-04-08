import argparse
import numpy as np
import sys
import random
import os
from pathlib import Path
from joblib import Parallel, delayed, dump
from sklearn.model_selection import train_test_split
from sklearn.base import clone  # used to create independent copies of pipelines

# Append project source directory.
project_src = Path(__file__).resolve().parents[1]
sys.path.append(str(project_src))

from common.data_loader import load_data
from common.cross_validation import grid_search_cv_generic
from common.preprocessing import preprocessing_pipelines  # Import all defined pipelines
from models import (
    build_model_decision_tree,
    build_standard_perceptron,
    build_averaged_perceptron,
    build_margin_perceptron,
    build_ensemble_model,
    build_model_adaboost,
    build_model_svm
)

def identity_conversion(labels):
    return labels

def convert_neg1_to0(labels):
    return np.where(labels == -1, 0, labels)

model_builders = {
    "dt": build_model_decision_tree,
    "perc": build_standard_perceptron,
    "avgperc": build_averaged_perceptron,
    "marginperc": build_margin_perceptron,
    "ensemble": build_ensemble_model,
    "adaboost": build_model_adaboost,
    "svm": build_model_svm,
}

def evaluate_params(params, X_train, y_train, model_builder, k, label_conversion):
    print("Evaluating params:", params)
    try:
        best_params, f1 = grid_search_cv_generic(
            X_train, y_train, model_builder, [params],
            k=k, label_conversion=label_conversion, stratified=True
        )
    except Exception as e:
        print(f"Exception for params {params}: {e}")
        f1 = -np.inf
    return params, f1

def main():
    parser = argparse.ArgumentParser(description="Generic Hyperparameter Tuning Script")
    parser.add_argument('--model', type=str, default='dt',
                        choices=['dt', 'perc', 'avgperc', 'marginperc', 'ensemble', 'adaboost', 'svm'],
                        help="Select model to tune")
    parser.add_argument('--k', type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument('--n_iter', type=int, default=20, help="Number of random hyperparameter combinations to try")
    args = parser.parse_args()
    
    # Load training data.
    X_train, y_train = load_data("data/train.csv", label_column="label")

    # For perceptron-based models, convert labels to {-1, +1}.
    if args.model in ['perc', 'avgperc', 'marginperc', 'adaboost', 'svm']:
        y_train = np.where(y_train == 0, -1, 1)

    # Set label conversion function.
    if args.model == 'dt':
        label_conversion = identity_conversion
    else:
        label_conversion = convert_neg1_to0

    # Define hyperparameter grid based on the model type.
    if args.model == 'dt':
        hyperparam_grid = [
            {"max_depth": d, "min_samples_split": s}
            for d in [5, 10]
            for s in [2, 5, 10]
        ]
    elif args.model == 'perc':
        hyperparam_grid = [
            {"epochs": e, "lr": lr, "decay_lr": decay, "mu": 0}
            for e in [10, 20, 30]
            for lr in [0.1, 0.5, 1.0]
            for decay in [False, True]
        ]
    elif args.model == 'avgperc':
        hyperparam_grid = [
            {"epochs": e, "lr": lr}
            for e in [10, 20, 30]
            for lr in [0.1, 0.5, 1.0]
        ]
    elif args.model == 'marginperc':
        hyperparam_grid = [
            {"epochs": e, "lr": lr, "mu": mu}
            for e in [10, 20, 30]
            for lr in [0.1, 0.5, 1.0]
            for mu in [0.8, 1.0, 1.2]
        ]
    elif args.model == "ensemble":
        hyperparam_grid = [
            {
                "dt_params": {"max_depth": d, "min_samples_split": s},
                "perc_params": {"epochs": e, "lr": lr, "decay_lr": decay_lr, "mu": mu},
                "w_dt": w_dt,
                "w_perc": 1 - w_dt
            }
            for d in [5, 10]
            for s in [2, 5]
            for e in [10, 20]
            for lr in [0.1, 0.5]
            for decay_lr in [False, True]
            for w_dt in [0.5, 0.6, 0.7]
            for mu in [0.8, 1.0, 1.2]
        ]
    elif args.model == "adaboost":
        hyperparam_grid = [
            {"n_estimators": n, "n_thresholds": t, "weak_learner_depth": d}
            for n in [20, 50, 100]
            for t in [5, 10, 20]
            for d in [2, 3]
        ]
    elif args.model == "svm":
        hyperparam_grid = [
            {"lr": lr, "lambda_param": lp, "n_epochs": n}
            for lr in [0.001, 0.01, 0.1]
            for lp in [0.001, 0.01, 0.1]
            for n in [500, 1000]
        ]
    else:
        raise ValueError("Unknown model type.")
    
    # Set model builder.
    if args.model == "ensemble":
        def ensemble_builder_wrapper(X, y, params):
            X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42)
            model = build_ensemble_model(X_train_sub, y_train_sub, params)
            model.update_weights(X_val, y_val)
            return model
        model_builder = ensemble_builder_wrapper
    else:
        model_builder = model_builders[args.model]

    # Define candidate PCA n_components values.
    candidate_n_components = [50, 70, 100]

    best_overall_f1 = -np.inf
    best_pipeline_name = None
    best_pipeline_obj = None
    best_hyperparams_overall = None
    best_n_components = None

    # Loop over each pre-processing pipeline and candidate n_components.
    for pipeline_name, pipeline in preprocessing_pipelines.items():
        for n_components in candidate_n_components:
            # Clone the pipeline to ensure an independent copy.
            current_pipeline = clone(pipeline)
            current_pipeline.set_params(pca__n_components=n_components)
            
            print(f"\nEvaluating pre-processing pipeline: {pipeline_name} with PCA n_components = {n_components}")
            X_train_trans = current_pipeline.fit_transform(X_train)
            
            # Sample hyperparameter grid if necessary.
            if len(hyperparam_grid) > args.n_iter:
                hyperparam_grid_sub = random.sample(hyperparam_grid, args.n_iter)
                print(f"Random search: evaluating {args.n_iter} out of {len(hyperparam_grid)} hyperparameter combinations.")
            else:
                hyperparam_grid_sub = hyperparam_grid

            results = Parallel(n_jobs=-1)(
                delayed(evaluate_params)(params, X_train_trans, y_train, model_builder, args.k, label_conversion)
                for params in hyperparam_grid_sub
            )
            
            best_params, best_metric = max(results, key=lambda x: x[1])
            print(f"Best hyperparameters with pipeline {pipeline_name} (PCA n_components={n_components}): {best_params} with F1: {best_metric:.3f}")
            
            # Update overall best configuration if this combination is superior.
            if best_metric > best_overall_f1:
                best_overall_f1 = best_metric
                best_pipeline_name = pipeline_name
                best_pipeline_obj = current_pipeline  # current_pipeline already has its PCA set.
                best_hyperparams_overall = best_params
                best_n_components = n_components

    print(f"\nOverall best pre-processing pipeline: {best_pipeline_name} with PCA n_components = {best_n_components}")
    print(f"Best hyperparameters for model {args.model}: {best_hyperparams_overall} with F1: {best_overall_f1:.3f}")

    # Create folder tuned_models if it doesn't exist.
    tuned_folder = "tuned_models"
    if not os.path.exists(tuned_folder):
        os.makedirs(tuned_folder)

    # Save best pipeline configuration and hyperparameters using model name in file.
    tuned_config_file = os.path.join(tuned_folder, f"{args.model}_best_model_config.pkl")
    best_config = {
        "pipeline_name": best_pipeline_name,
        "pca_n_components": best_n_components,
        "pipeline": best_pipeline_obj,
        "hyperparameters": best_hyperparams_overall,
        "model": args.model,
        "f1_score": best_overall_f1
    }
    dump(best_config, tuned_config_file)
    print(f"Saved best pipeline configuration and hyperparameters to {tuned_config_file}")

if __name__ == '__main__':
    main()
