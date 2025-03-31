import numpy as np

from common.evaluation import compute_metrics

from models.decision_tree.builders import build_model_decision_tree
from models.perceptron.builders import build_averaged_perceptron, build_margin_perceptron
from models.ensemble.ensemble_dt_perc import ensemble_predict

class EnsembleModel:
    def __init__(self, dt_model, perc_model, w_dt, w_perc):
        self.dt_model = dt_model
        self.perc_model = perc_model
        self.w_dt = 0.5 if w_dt is None else w_dt
        self.w_perc = 0.5 if w_perc is None else w_perc
        # Store initial weights for reference
        self.initial_w_dt = self.w_dt
        self.initial_w_perc = self.w_perc

    def train(self, X, y, epochs=1):
        y_perc = np.where( y==0, -1, 1)

        self.perc_model.train(X, y_perc, epochs)

    def update_weights(self, X_val, y_val):
        """
        Function to update ensemble weights based on the F1-scores 
        of the individual models on a validation set.
        """
        #print("\nDEBUG: Starting weight update")
        #print(f"DEBUG: Validation set size: {len(X_val)} samples")
        if len(X_val) == 0:
            print("Warning: Empty validation set, keeping original weights")
            return
        
        dt_preds = self.dt_model.predict(X_val)
        perc_preds = self.perc_model.predict(X_val)
        # Converting perceptron predictions from {-1,1} to {0,1}
        perc_preds = np.where(np.array(perc_preds) == -1, 0, perc_preds)

        _, _, f1_dt = compute_metrics(y_val, dt_preds)
        _, _, f1_perc = compute_metrics(y_val, perc_preds)

        total = f1_dt + f1_perc

        #print(f"DEBUG: f1_dt = {f1_dt:.5f}, f1_perc = {f1_perc:.5f}, total = {total:.5f}")

        if total < 1e-8 or np.isnan(total) or np.isinf(total):
            print("Warning: Invalid F1 scores, using initial weights")
            self.w_dt, self.w_perc = 0.5, 0.5
        else:
            self.w_dt = f1_dt/total
            self.w_perc = f1_perc/total

        print(f"Updated ensemble weights: Decision Tree = {self.w_dt:.3f}, Perceptron = {self.w_perc:.3f}")


    def predict(self, X):
        dt_preds = self.dt_model.predict(X)
        perc_preds = self.perc_model.predict(X)
        # Converting perceptron predictions from {-1,1} to {0,1}
        perc_preds = np.where(np.array(perc_preds) == -1, 0, perc_preds)

        ensemble_preds = ensemble_predict([dt_preds, perc_preds], weights=[self.w_dt, self.w_perc])
        return ensemble_preds

def build_ensemble_model(X_train, y_train, hyperparams):
    
    dt_model = build_model_decision_tree(X_train, y_train, hyperparams['dt_params'])
    
    # For perceptron, convert labels from {0,1} to {-1,1}.
    y_train_perc = np.where(y_train == 0, -1, 1)
    perc_model = build_margin_perceptron(X_train, y_train_perc, hyperparams['perc_params'])

    #initial_weights = (hyperparams.get('w_dt', 0.5), hyperparams.get('w_perc', 0.5))
    
    w_dt = hyperparams['w_dt']
    w_perc = hyperparams['w_perc']
    
    return EnsembleModel(dt_model, perc_model, w_dt, w_perc)
