from models.adaboost.adaboost import AdaBoostModel
import numpy as np

def build_model_adaboost(X, y, hyperparams):
    """
    Builder function for AdaBoost. Expects y in {-1, +1}.
    hyperparams can include:
       - n_estimators: number of boosting rounds.
    """
    n_estimators = hyperparams.get('n_estimators', 50)
    n_thresholds = hyperparams.get('n_thresholds', 10)
    weak_learner_depth = hyperparams.get('weak_learner_depth', 2)
    #init_weights_method = hyperparams.get('init_weights_method', 'uniform')
    model = AdaBoostModel(n_estimators=n_estimators,
                          n_thresholds=n_thresholds,
                          weak_learner_depth=weak_learner_depth)
    model.train(X, y)
    return model