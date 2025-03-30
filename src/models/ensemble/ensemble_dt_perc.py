import numpy as np

def ensemble_predict(preds_list, weights=None, threshold=0.5):
    """
    Ensemble model to combine predictions of different models

    Parameters:
      preds_list (list of arrays): Each element is an array of predictions (in {0, 1}).
      weights (list of floats): Optional weights for each model's prediction. 
                                If None, equal weights are assumed.
      threshold (float): Threshold for deciding class 1.
    
    Returns:
      final_preds (np.array): The ensembled predictions as an array of 0's and 1's.
    """

    preds_array = np.array(preds_list)

    if weights is None:
        weights = np.ones(preds_array.shape[0]) / preds_array.shape[0]
    else:
        weights = np.array(weights)

        if weights.sum() == 0:
            weights = np.ones_like(weights)/len(weights)

    combined = np.average(preds_array, axis=0, weights=weights)
    final_preds = np.where(combined >= threshold, 1, 0)
    return final_preds