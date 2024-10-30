import numpy as np

# ----------------------------
#   Logistic Regression
# ----------------------------


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        X_b = np.hstack([np.ones((m, 1)), X])
        self.theta = np.zeros(n + 1)

        for iteration in range(self.n_iterations):
            z = X_b.dot(self.theta)
            predictions = self.sigmoid(z)
            errors = predictions - y
            gradient = (1 / m) * X_b.T.dot(errors)
            self.theta -= self.lr * gradient

            if iteration % 100 == 0:
                loss = -(
                    y * np.log(predictions + 1e-15)
                    + (1 - y) * np.log(1 - predictions + 1e-15)
                ).mean()
                print(f"Iteration {iteration}: Loss {loss}")

    def predict_proba(self, X):
        m = X.shape[0]
        X_b = np.hstack([np.ones((m, 1)), X])
        return self.sigmoid(X_b.dot(self.theta))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
