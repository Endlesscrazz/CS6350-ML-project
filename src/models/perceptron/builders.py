

from models.perceptron.standard import Perceptron
from models.perceptron.averaged import AveragedPerceptron
from models.perceptron.margin import MarginPerceptron

def build_standard_perceptron(X_train, y_train, hyperparams):
    epochs = hyperparams.get("epochs", 10)
    lr = hyperparams.get("lr", 1.0)
    decay_lr = hyperparams.get("decay_lr", False)
    # Standard perceptron: mu is fixed to 0.
    model = Perceptron(num_features=X_train.shape[1], lr=lr, decay_lr=decay_lr, mu=0)
    model.train(X_train, y_train, epochs)
    return model

def build_averaged_perceptron(X_train, y_train, hyperparams):
    epochs = hyperparams.get("epochs", 10)
    lr = hyperparams.get("lr", 1.0)
    model = AveragedPerceptron(num_features=X_train.shape[1], lr=lr)
    model.train(X_train, y_train, epochs)
    return model

def build_margin_perceptron(X_train, y_train, hyperparams):
    epochs = hyperparams.get("epochs", 10)
    mu = hyperparams.get("mu", 1.0)
    model = MarginPerceptron(num_features=X_train.shape[1], mu=mu)
    model.train(X_train, y_train, epochs)
    return model
