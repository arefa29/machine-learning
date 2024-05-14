import numpy as np

class LinearRegression:
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
            # forward pass compute y_pred
            y_pred = np.dot(X, self.weights) + self.bias

            # compute gradients
            dl_dw = (1/self.n_samples) * 2 * np.dot(X.T, y_pred - y)
            dl_db = (1/self.n_samples) * 2 * np.sum(y_pred - y)

            # update weights
            self.weights = self.weights - self.lr * dl_dw
            self.bias = self.bias - self.lr * dl_db
        
    def predict(self, X):
        """Returns: List of predictions y_pred"""
        return (np.dot(X, self.weights) + self.bias)
