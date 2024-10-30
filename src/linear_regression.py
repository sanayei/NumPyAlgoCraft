import numpy as np

# ----------------------------
#    Linear Regression
# ----------------------------


import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    def fit(self, X, y):
        # Flatten y to avoid shape mismatch
        y = y.flatten()

        m, n = X.shape
        X_b = np.hstack([np.ones((m, 1)), X])  # Add bias term
        self.theta = np.zeros(n + 1)

        for iteration in range(self.n_iterations):
            predictions = X_b.dot(self.theta)
            errors = predictions - y
            gradient = (2 / m) * X_b.T.dot(errors)
            self.theta -= self.lr * gradient

    def predict(self, X):
        m = X.shape[0]
        X_b = np.hstack([np.ones((m, 1)), X])  # Add bias term
        return X_b.dot(self.theta)
