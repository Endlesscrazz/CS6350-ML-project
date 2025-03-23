import numpy as np

class AveragedPerceptron:
    def __init__(self, num_features, lr,):
        """
        Constructor to initalize a new perceptron 
        """

        self.lr = lr
        self.num_features = num_features
        self.t = 0
        self.count = 0

        self.w = np.random.uniform(-0.01,0.01, size=num_features)
        self.b = np.random.uniform(-0.01, 0.01)

        self.w_sum = np.zeros(num_features)
        self.b_sum = 0.0

    def get_hyperparams(self) -> dict:
        return {'lr': self.lr}

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

                if y[i] * score <= 0:
                    self.w += self.lr * y[i] * x[i]
                    self.b += self.lr * y[i]
                    errors += 1
                
                self.w_sum += self.w
                self.b_sum += self.b
                self.count += 1
            
        self.w = self.w_sum / self.count
        self.b = self.b_sum / self.count

            #print(f"Epoch {epoch +1}: erros={errors}, weight_norm = {np.linalg.norm(self.w):.4f}")
                
    def predict(self, x) -> list:

        scores = np.dot(x, self.w) + self.b
        predictions = np.where(scores >=0, 1, -1)
        return predictions.tolist()
    




