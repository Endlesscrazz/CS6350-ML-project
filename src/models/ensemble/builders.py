import numpy as np
from models.decision_tree.builders import build_model_decision_tree
from models.perceptron.builders import build_averaged_perceptron, build_margin_perceptron
from models.ensemble.ensemble_dt_perc import ensemble_predict

class EnsembleModel:
    def __init__(self, dt_model, perc_model, w_dt, w_perc):
        self.dt_model = dt_model
        self.perc_model = perc_model
        self.w_dt = w_dt
        self.w_perc = w_perc

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
    perc_model = build_averaged_perceptron(X_train, y_train_perc, hyperparams['perc_params'])
    
    w_dt = hyperparams['w_dt']
    w_perc = hyperparams['w_perc']
    
    return EnsembleModel(dt_model, perc_model, w_dt, w_perc)
