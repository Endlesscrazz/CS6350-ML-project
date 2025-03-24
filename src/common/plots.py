import matplotlib.pyplot as plt
import numpy as np
from common.evaluation import compute_metrics

def plot_learning_curves(model_builder, hyperparams, X_train, y_train, X_val, y_val, epochs, 
                         convert_func=None):
    """
    Generic function to plot learning curves.
    
    Parameters:
      model_builder: A function that builds and returns a model.
      hyperparams: Hyperparameter dictionary for the model.
      X_train, y_train: Training data and labels.
      X_val, y_val: Validation data and labels.
      epochs: Total number of epochs to train.
      convert_func: Optional function to convert predictions (e.g., from {-1,1} to {0,1}).
    """
    train_f1_list = []
    val_f1_list = []
    epoch_list = list(range(1, epochs+1))
    
    model = model_builder(X_train, y_train, hyperparams)
    
    for epoch in range(epochs):
        model.train(X_train, y_train, 1)
        
        train_preds = np.array(model.predict(X_train))
        val_preds = np.array(model.predict(X_val))
        
        if convert_func is not None:
            train_preds = convert_func(train_preds)
            val_preds = convert_func(val_preds)
        
        _, _, train_f1 = compute_metrics(y_train, train_preds)
        _, _, val_f1 = compute_metrics(y_val, val_preds)
        
        train_f1_list.append(train_f1)
        val_f1_list.append(val_f1)
    
    # Plot learning curves.
    plt.plot(epoch_list, train_f1_list, label="Training F1")
    plt.plot(epoch_list, val_f1_list, label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Learning Curves")
    plt.legend()
    plt.show()
