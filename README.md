# Android Malware Detection

This repository implements an end-to-end pipeline to detect malicious Android applications from system-call fingerprints. It covers:

1.  Data loading & preprocessing
2.  Model implementations (Decision Tree, Perceptrons, Ensemble(perc and dt), AdaBoost, SVM, and a PyTorch MLP)
3.  Hyperparameter tuning with cross-validation
4.  Retraining the best configuration on the full training set
5.  Threshold tuning for the neural network
6.  Generating a Kaggle-style `submission.csv`

---

## Repository Structure

.
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── eval.anon.csv
│   └── eval.id
│
├── src/
│   ├── common/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── cross_validation.py
│   │   ├── evaluation.py
│   │   └── submission.py
│   │
│   ├── models/
│   │   ├── decision_tree/
│   │   │   ├── builders.py
│   │   │   ├── decision_tree.py
│   │   │   └── main.py
│   │   ├── perceptron/
|   |   |   ├── _init_.py
│   │   │   ├── averaged.py
│   │   │   └── main.py
│   │   │   ├── builders.py
│   │   │   ├── margin.py
│   │   │   └── standard.py
│   │   ├── ensemble/
│   │   │   ├── builders.py
│   │   │   ├── ensemble_dt_perc.py
│   │   │   └── main.py
│   │   ├── adaboost/
│   │   │   ├── builders.py
│   │   │   ├── adaboost.py
│   │   │   └── main.py
│   │   ├── svm/
│   │   │   ├── builders.py
│   │   │   ├── svm.py
│   │   │   └── main.py
│   │   └── neural_network/
│   │       ├── nn.py
│   │       └── main.py
│   │
│   ├── tuning/
│   │   └── tune_models.py
│   │
│   └── main.py
│
├── tuned_models/
│   ├── dt_best_model_config.pkl
│   ├── perc_best_model_config.pkl
│   ├── nn_best_model_config.pkl
│   └── ...
│
├── output/
│   ├── preprocessing_pipeline.pkl # Or model-specific pipelines
│   ├── dt_model.pkl               # Or model-specific models
│   ├── nn_model_with_threshold.pkl
│   ├── submission.csv
│   └── ...
│
└── README.md

## Requirements

* **Python** 3.8+
* **Libraries** :
    * `numpy`, `pandas`, `scikit-learn`, `joblib`
    * `torch` (PyTorch), `matplotlib`

## If any libraries are not installed in CADE machines, You can install them via the requirements.txt
```bash
pip3 install requirements.txt
```

## How to Run

### 1. Hyperparameter Tuning

Run from the project root directory:

```bash
python3 src/tuning/tune_models.py \
  --model <MODEL> \
  --k 5 \
  --n_iter 20 \
  --seed 42
  ```

### 2. Retrain on Full Training Set

Run from the project root directory:
# Classical models (Example: dt,perc,svm..)

```bash
python3 src/main.py --model dt --seed 42


# Neural network 
python3 src/main.py --seed 42 
```

This saves training artifacts to the output/ directory:

output/<MODEL>_model.pkl 
output/pipeline_<MODEL>.pkl 
output/nn_model_with_threshold.pkl (for NN)

### 3. Generate Kaggle-Style Submission

Run from the project root directory:

``` bash
python3 src/common/submission.py --model <MODEL>
```

## File Descriptions

### src/tuning/tune_models.py
Searches over preprocessing pipelines and hyperparameters via cross‑validation, saves best config.
### src/main.py
Loads tuned config, refits preprocessing & model on full train set, tunes NN threshold if needed, saves artifacts.
### src/common/submission.py
Loads pipeline + model (or NN bundle), transforms eval.anon.csv, writes output/submission.csv.
### src/models/…
Implements each algorithm; neural_network/nn.py defines a PyTorch MLP and wrapper.