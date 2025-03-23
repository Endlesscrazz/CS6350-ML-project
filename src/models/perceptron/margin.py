import numpy as np

class MarginPerceptron:
    def __init__(self, num_features,lr, mu=0):
        """
        Constructor to initalize a new perceptron 
        """

        self.num_features = num_features
        self.mu = mu
        self.lr = lr
        self.t = 0

        self.w = np.random.uniform(-0.01,0.01, size=num_features)
        self.b = np.random.uniform(-0.01, 0.01)

    def get_hyperparams(self) -> dict:
        return {'lr': self.lr, 'mu': self.mu}

    def train(self, x:np.ndarray, y: np.ndarray, epochs: int):
        """
        Function to train the perceptron on data x with labels y with given number of epochs
        """

        num_examples = x.shape[0]
        for epoch in range(epochs):
            
            shuffled_indices = np.random.permutation(num_examples)
            errors = 0
            for i in shuffled_indices:
                score = np.dot(self.w, x[i]) + self.b

                if y[i] * score < self.mu:

                    step_size = (self.mu - y[i] * score) / (np.dot(x[i], x[i]) + 1)
                    self.w += self.lr * step_size * y[i] * x[i]
                    self.b += self.lr * step_size * y[i]
                    errors += 1

            #print(f"Epoch {epoch +1}: erros={errors}, weight_norm = {np.linalg.norm(self.w):.4f}")
            self.t += 1
    
    def predict(self, x) -> list:

        scores = np.dot(x, self.w) + self.b
        predictions = np.where(scores >=0, 1, -1)
        return predictions.tolist()
    




