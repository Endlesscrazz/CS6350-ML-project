from models.adaboost.adaboost import AdaBoostModel
import numpy as np

def build_model_adaboost(X, y, hyperparams):
    """
    Builder function for AdaBoost. Expects y in {-1, +1}.
    hyperparams can include:
       - n_estimators: number of boosting rounds.
    """
    n_estimators = hyperparams.get('n_estimators', 50)
    model = AdaBoostModel(n_estimators=n_estimators)
    model.train(X, y)
    return model