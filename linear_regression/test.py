from sklearn import datasets
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

def mse(y_true, y_pred):
    return np.mean((y_true-  y_pred) ** 2)

def plot_regression_line(pred, X, y, X_test):
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(9, 6))
    plt.scatter(X[:, 0], y, cmap=cmap, marker='o', s=10)
    plt.plot(X_test[:, 0], pred, color='m')
    plt.show()

if __name__ == '__main__':

    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    pred = reg_model.predict(X_test)
    error = mse(y_test, pred)

    plot_regression_line(pred, X, y, X_test)
