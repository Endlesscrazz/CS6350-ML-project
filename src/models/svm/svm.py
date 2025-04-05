import numpy as np

class SVM:
    """
    Simple Linear svm impkemented using SGD on hinge loss

    Hyperparams:
    -lr : learning rate
    lambda_param: regularizaton strength
    -n_epochs: number of epochs
    """
    def __init__(self, lr=0.01, lambda_param=0.01, n_epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.W = None
        self.b = 0

    def train(self, X, y):
        """
        Train SVM model on data X with labels y (should be +1 and -1)
        """
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)

        for epoch in range(self.n_epochs):
            for idx,x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b)
                if condition < 1:
                    self.w = self.w - self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b = self.b - self.lr * (-y[idx])
                else:
                    self.w = self.w - self.lr * (2 * self.lambda_param * self.w)
    
    def predict(self, X):

        approx = np.dot(X, self.w) + self.b
        return np.where(approx >= 0, 1, -1)
    
    def predict_submission(self, X):
        raw_preds = self.predict(X)
        return np.where(raw_preds == -1, 0, raw_preds)
        