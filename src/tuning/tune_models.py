import argparse
import numpy as np
import sys
import random
import os
from pathlib import Path
from joblib import Parallel, delayed, dump
from sklearn.model_selection import train_test_split
from sklearn.base import clone  
from sklearn.pipeline import Pipeline # Needed for isinstance check

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

model_builders = {
    "dt": build_model_decision_tree,
    "perc": build_standard_perceptron,
    "avgperc": build_averaged_perceptron,
    "marginperc": build_margin_perceptron,
    "ensemble": build_ensemble_model,
    "adaboost": build_model_adaboost,
    "svm": build_model_svm,
    "nn" : build_model_nn
}

def evaluate_params(params, X_train, y_train, model_builder, k, label_conversion, stratified):
    # Note: Added stratified flag propagation
    print("Evaluating params:", params)
    try:
        # Pass stratified flag to the CV function
        best_params, f1 = grid_search_cv_generic(
            X_train, y_train, model_builder, [params],
            k=k, label_conversion=label_conversion, stratified=stratified
        )
    except Exception as e:
        # Print traceback for detailed debugging
        import traceback
        print(f"Exception for params {params}: {e}")
        traceback.print_exc()
        f1 = -np.inf
    return params, f1

def main():
    parser = argparse.ArgumentParser(description="Generic Hyperparameter Tuning Script")
    parser.add_argument('--model', type=str, default='dt',
                        choices=['dt', 'perc', 'avgperc', 'marginperc', 'ensemble', 'adaboost', 'svm', 'nn'],
                        help="Select model to tune")
    parser.add_argument('--k', type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument('--n_iter', type=int, default=20, help="Number of random hyperparameter combinations to try")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility") # Added seed argument
    args = parser.parse_args()

    # --- Reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load training data.
    X_train_orig, y_train_orig = load_data("data/train.csv", label_column="label")

    # --- Model Specific Setup ---
    y_train = y_train_orig.copy() 
    stratified_cv = True # Default to stratified CV

    if args.model in ['perc', 'avgperc', 'marginperc', 'adaboost', 'svm', 'nn']: 
        y_train = np.where(y_train_orig == 0, -1, 1) 
        # NN's build_model_nn handles conversion back to 0/1, so label_conversion is correct
        label_conversion = convert_neg1_to0 # Converts {-1, 1} back to {0, 1} for F1 score
    else: # dt
        label_conversion = identity_conversion # Expects {0, 1}

    # Define hyperparameter grid based on the model type.
    if args.model == 'dt':
        hyperparam_grid = [
            {"max_depth": d, "min_samples_split": s}
            for d in [5, 10, 15] 
            for s in [2, 5, 10]
        ]
    elif args.model == 'perc':
        hyperparam_grid = [
            {"epochs": e, "lr": lr, "decay_lr": decay, "mu": 0}
            for e in [10, 20, 30]
            for lr in [0.01, 0.1, 0.5] # Adjusted LR
            for decay in [False, True]
        ]
    elif args.model == 'avgperc':
         hyperparam_grid = [
            {"epochs": e, "lr": lr}
            for e in [10, 20, 30]
            for lr in [0.01, 0.1, 0.5] # Adjusted LR
        ]
    elif args.model == 'marginperc':
        hyperparam_grid = [
            {"epochs": e, "lr": lr, "mu": mu}
            for e in [10, 20, 30]
            for lr in [0.01, 0.1, 0.5] # Adjusted LR
            for mu in [0.5, 0.8, 1.0, 1.2] # Expanded mu
        ]
    elif args.model == "ensemble":

        hyperparam_grid = [
             {
                "dt_params": {"max_depth": d, "min_samples_split": s},
                "perc_params": {"epochs": e, "lr": lr, "decay_lr": decay_lr, "mu": mu}, # Ensure mu is included if MarginPerceptron is used
                "w_dt": w_dt,
                "w_perc": 1 - w_dt
            }
            for d in [5, 10]
            for s in [2, 5]
            for e in [10, 20]
            for lr in [0.1, 0.5]
            for decay_lr in [False, True]
            for w_dt in [0.5, 0.6, 0.7]
            for mu in [0.8, 1.0, 1.2] # Added mu for MarginPerceptron possibility
        ]
    elif args.model == "adaboost":
        hyperparam_grid = [
            {"n_estimators": n, "n_thresholds": t, "weak_learner_depth": d}
            for n in [20, 50, 100, 150] # Expanded estimators
            for t in [5, 10, 20]
            for d in [1, 2, 3] 
        ]
    elif args.model == "svm":
         hyperparam_grid = [
            {"lr": lr, "lambda_param": lp, "n_epochs": n}
            for lr in [1e-4, 1e-3, 0.01, 0.1] # Expanded LR
            for lp in [1e-4, 1e-3, 0.01, 0.1] # Expanded lambda
            for n in [500, 1000, 1500] # Expanded epochs
        ]
    elif args.model == "nn":
        hyperparam_grid = [
            {
                "hidden_dims" : hd, 
                "lr": lr,
                "epochs": e, # Max epochs, early stopping applies
                "batch_size": bs,
                "dropout":do,
                "optimizer":opt,
                "weight_decay": wd, 
                "random_state": args.seed, 
                "early_stopping_patience": 15 
            }
            for hd in [(64, 32), (128,64), (256,128)] 
            for lr in [0.003, 0.004, 0.005, 0.006, 0.007]
            for e in [100, 150] 
            for bs in [32, 64, 128]
            for do in [0.2, 0.25, 0.3, 0.35] 
            for opt in ["adam", "adamw"]
            for wd in [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 2e-4, 5e-4] # ADDED weight decay values
        ]
    else:
        raise ValueError("Unknown model type.")

    if args.model == "ensemble":
         model_builder = build_ensemble_model
    else:
        model_builder = model_builders[args.model]


    pipeline_configs_to_test = {}
    candidate_n_components = [50, 70, 100] 

    # Add the pipelines from preprocessing_pipelines with PCA variations
    for name, pipeline in preprocessing_pipelines.items():
        if 'pca' in pipeline.named_steps:
            for n_comp in candidate_n_components:
                config_name = f"{name}_pca{n_comp}"
                current_pipeline = clone(pipeline)
                current_pipeline.set_params(pca__n_components=n_comp)
                pipeline_configs_to_test[config_name] = current_pipeline
        else:
            # If pipeline doesn't have PCA
            pipeline_configs_to_test[name] = clone(pipeline)

    best_overall_f1 = -np.inf
    best_pipeline_name = None
    best_pipeline_obj = None
    best_hyperparams_overall = None

    for pipeline_name, current_pipeline in pipeline_configs_to_test.items():

        # --- Special case: Only test nn_raw_scaled pipeline if model is NOT nn? ---
        # Decision: Let's test all defined pipelines for all models for maximum exploration,
        # unless a pipeline is fundamentally incompatible (e.g., requires specific data type).
        # If 'nn_raw_scaled' performs poorly for non-NN models, the results will reflect that.
        # if args.model != "nn" and pipeline_name == "nn_raw_scaled":
        #    print(f"\nSkipping pipeline {pipeline_name} for model {args.model}")
        #    continue # Optional: skip raw pipeline for non-NN models

        print(f"\n>>> Evaluating Pipeline: {pipeline_name} for Model: {args.model}")

        # Apply preprocessing
        # Use y_train_orig if pipeline steps require original labels (e.g., some sampling methods)
        # Here, fit_transform usually only needs X.
        print("Fitting and transforming data...")
        try:
             # Pass y if any step requires it (e.g., SMOTE, but not used here)
            X_train_trans = current_pipeline.fit_transform(X_train_orig)
            print(f"Transformed data shape: {X_train_trans.shape}")
        except Exception as e:
            print(f"Error during pipeline {pipeline_name} fit_transform: {e}")
            import traceback
            traceback.print_exc()
            continue # Skip this pipeline if transformation fails

        if len(hyperparam_grid) > args.n_iter:
            hyperparam_grid_sub = random.sample(hyperparam_grid, args.n_iter)
            print(f"Random search: evaluating {args.n_iter} of {len(hyperparam_grid)} hyperparameter combinations.")
        else:
            hyperparam_grid_sub = hyperparam_grid
            print(f"Grid search: evaluating all {len(hyperparam_grid)} hyperparameter combinations.")

        # CV
        print(f"Starting parallel evaluation for {len(hyperparam_grid_sub)} param sets...")
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_params)(
                params,
                X_train_trans, #  transformed data
                y_train,       #  potentially {-1, 1} converted labels
                model_builder,
                args.k,
                label_conversion, # Function to convert back for F1 metric
                stratified=stratified_cv # Pass stratification flag
            )
            for params in hyperparam_grid_sub
        )

        # Find best result for this pipeline
        if not results:
            print("No results obtained for this pipeline.")
            continue

        # Filter out results where F1 is -inf (indicating an error during evaluation)
        valid_results = [res for res in results if res[1] > -np.inf]
        if not valid_results:
            print("All parameter evaluations failed for this pipeline.")
            continue

        best_params_pipeline, best_metric_pipeline = max(valid_results, key=lambda x: x[1])

        print(f"Best hyperparameters with pipeline '{pipeline_name}': {best_params_pipeline} | F1: {best_metric_pipeline:.4f}")

        # Update overall best configuration
        if best_metric_pipeline > best_overall_f1:
            print(f"*** New Best Overall F1 Found: {best_metric_pipeline:.4f} (improved from {best_overall_f1:.4f}) ***")
            best_overall_f1 = best_metric_pipeline
            best_pipeline_name = pipeline_name
            best_pipeline_obj = current_pipeline
            best_hyperparams_overall = best_params_pipeline
            # Extract PCA components if applicable for reporting
            try:
                best_n_components = best_pipeline_obj.named_steps['pca'].n_components
            except KeyError:
                best_n_components = None # No PCA step in this pipeline


    # --- Reporting and Saving ---
    print("\n" + "="*50)
    print("Hyperparameter Tuning Summary")
    print("="*50)
    if best_pipeline_obj:
        print(f"Best Performing Model: {args.model}")
        print(f"Best Pipeline Configuration Name: {best_pipeline_name}")
        # Extract PCA components if applicable for reporting
        try:
            best_n_components_report = best_pipeline_obj.named_steps['pca'].n_components
            print(f"   (PCA n_components: {best_n_components_report})")
        except KeyError:
            print("   (No PCA step in the best pipeline)")

        print(f"Best Cross-Validation F1 Score: {best_overall_f1:.4f}")
        print(f"Best Hyperparameters:")
        for key, value in best_hyperparams_overall.items():
            print(f"  - {key}: {value}")

        # Create folder tuned_models if it doesn't exist.
        tuned_folder = "tuned_models"
        os.makedirs(tuned_folder, exist_ok=True)

        # Save best pipeline configuration and hyperparameters using model name in file.
        tuned_config_file = os.path.join(tuned_folder, f"{args.model}_best_model_config.pkl")
        best_config = {
            "pipeline_name": best_pipeline_name,
            "pipeline": best_pipeline_obj, # Save the fitted pipeline object
            "hyperparameters": best_hyperparams_overall,
            "model": args.model,
            "f1_score": best_overall_f1,
            "seed": args.seed # Save seed used for tuning run
        }
        dump(best_config, tuned_config_file)
        print(f"\nSaved best pipeline configuration and hyperparameters to {tuned_config_file}")
    else:
        print("No successful model configurations found.")

if __name__ == '__main__':
    main()