from models.svm import SVM

def build_model_svm(X, y, hyperparams):
    """
    Builder function for SVM.
    """

    lr = hyperparams.get('lr',0.01)
    lambda_param = hyperparams.get('lambda_param', 0.01)
    n_epochs = hyperparams.get('n_epochs', 1000)

    model = SVM(lr=lr, lambda_param=lambda_param, n_epochs=n_epochs)
    model.train(X, y)
    
    return model