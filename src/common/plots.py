# src/plots.py

import matplotlib.pyplot as plt
import numpy as np
from common.evaluation import compute_metrics
from sklearn.base import clone
from sklearn.model_selection import train_test_split

# for nn threshold search
from models.nerual_network.main import find_optimal_threshold

def plot_model_learning_curve(model_builder,
                              hyperparams,
                              pipeline,
                              X, y,
                              model_type,
                              *,
                              val_fraction=0.2,
                              num_steps=10,
                              epochs=None,
                              random_state=42):
    """
    Plot learning curve for any model.
    
    - Classical models: vary training‐set fraction in `num_steps` and report train/val F1.
    - NN: train once for `epochs` and record epoch-by-epoch F1 on train & val.
    """

    # 1) Preprocess ONCE
    X_all = pipeline.transform(X)

    if model_type != 'nn':
        # ----- Classical models -----
        # 2) single split for val
        X_tr_full, X_val, y_tr_full, y_val = train_test_split(
            X_all, y, test_size=val_fraction,
            stratify=y, random_state=random_state
        )

        fracs = np.linspace(0.1, 1.0, num_steps)
        train_f1 = []
        val_f1   = []

        for frac in fracs:
            n_sub = int(len(X_tr_full) * frac)
            X_sub, y_sub = X_tr_full[:n_sub], y_tr_full[:n_sub]

            # 3) convert labels for models that need {-1,+1}
            if model_type in ['marginperc','avgperc','marginperc','svm','adaboost','ensemble']:
                y_train_labels = np.where(y_sub == 0, -1, 1)
                model = model_builder(X_sub, y_train_labels, hyperparams)
                raw_tr_preds = model.predict(X_sub)
                raw_val_preds = model.predict(X_val)
                # back to {0,1}
                preds_tr = np.where(raw_tr_preds == -1, 0, 1)
                preds_val = np.where(raw_val_preds == -1, 0, 1)
            else:
                # dt
                model = model_builder(X_sub, y_sub, hyperparams)
                preds_tr = model.predict(X_sub)
                preds_val = model.predict(X_val)

            _, _, f_tr = compute_metrics(y_sub, preds_tr)
            _, _, f_v  = compute_metrics(y_val, preds_val)

            train_f1.append(f_tr)
            val_f1.append(f_v)

        plt.plot(fracs, train_f1, '-o', label="Train F1")
        plt.plot(fracs, val_f1,   '-o', label="Val F1")
        plt.xlabel("Training set fraction")
        plt.ylabel("F1 Score")
        plt.title(f"Learning Curve: {model_type}")
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        # ----- Neural network -----
        if epochs is None:
            epochs = hyperparams.get('epochs', 50)

        # single split
        X_tr, X_val, y_tr0, y_val0 = train_test_split(
            X_all, y, test_size=val_fraction,
            stratify=y, random_state=random_state
        )
        y_tr = np.where(y_tr0 == 0, -1, 1)

        train_f1 = []
        val_f1   = []

        # train once for all epochs, recording after each epoch
        # (You need to modify your build_model_nn to expose per‐epoch stats;
        # otherwise the naive re‐train-from-scratch loop will be O(epochs²).)
        for ep in range(1, epochs + 1):
            hp = hyperparams.copy()
            hp.update({'epochs': ep, 'random_state': random_state})

            wrapper = model_builder(X_tr, y_tr, hp)
            thr, _ = find_optimal_threshold(wrapper, X_val, y_val0)

            preds_tr = wrapper.predict_submission(X_tr, threshold=thr)
            preds_val= wrapper.predict_submission(X_val, threshold=thr)

            _, _, f_tr = compute_metrics(y_tr0, preds_tr)
            _, _, f_v  = compute_metrics(y_val0, preds_val)

            train_f1.append(f_tr)
            val_f1.append(f_v)

        plt.plot(range(1, epochs + 1), train_f1, label="Train F1")
        plt.plot(range(1, epochs + 1), val_f1,   label="Val F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("Neural Network Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.show()
