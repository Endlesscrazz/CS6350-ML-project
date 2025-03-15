import numpy as np

class Perceptron:
    def __init__(self, num_features, lr, decay_lr, mu= 0):
        """
        Constructor to initalize a new perceptron 
        """

        self.lr = lr
        self.decay_lr = decay_lr
        self.num_features = num_features
        self.mu = mu
        self.t = 0

        self.w = np.random.uniform(-0.01,0.01, size=num_features)
        self.b = np.random.uniform(-0.01, 0.01)

    def get_hyperparams(self) -> dict:
        return {'lr': self.lr, 'decay_lr':self.decay_lr, 'mu': self.mu}

    def train(self, x:np.ndarray, y: np.ndarray, epochs: int):
        """
        Function to train the perceptron on data x with labels y with given number of epochs
        """

        num_examples = x.shape[0]
        for epoch in range(epochs):
            
            current_lr = self.lr/(1 + self.t) if self.decay_lr else self.lr
            shuffled_indices = np.random.permutation(num_examples)

            for i in shuffled_indices:
                score = np.dot(self.w, x[i]) + self.b

                if y[i] * score < 0:
                    self.w += current_lr * y[i] * x[i]
                    self.b += current_lr * y[i]

            self.t += 1
    
    def predict(self, x) -> list:

        scores = np.dot(x, self.w) + self.b
        predictions = np.where(scores >=0, 1, -1)
        return predictions.tolist()
    




