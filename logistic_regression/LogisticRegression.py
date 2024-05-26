import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.weights = np.zeros(self.n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            # forward
            y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)

            # compute grad
            dl_dw = -np.mean((y - y_pred) @ X)
            dl_db = -np.mean(y - y_pred)

            # backward pass
            self.weights -= self.lr * dl_dw
            self.bias -= self.lr * dl_db
            

    def sigmoid(self, X):
        return (1 / (1 + np.exp(-X)))
    
    def predict(self, X):
        return self.step(self.sigmoid(np.dot(X, self.weights) + self.bias))
    
    def step(self, x):
        return [1 if xi >= 0 else 0 for xi in x]
        