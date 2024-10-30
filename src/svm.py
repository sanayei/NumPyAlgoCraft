import numpy as np


# ----------------------------
#   Support Vector Machines (SVM)
# ----------------------------


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Convert labels to {-1, 1}
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for iteration in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]

            if iteration % 100 == 0:
                loss = self._compute_loss(X, y_)
                print(f"Iteration {iteration}: Loss {loss}")

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def _compute_loss(self, X, y):
        distances = 1 - y * (np.dot(X, self.w) - self.b)
        distances = np.where(distances > 0, distances, 0)
        hinge_loss = np.mean(distances)
        return hinge_loss + self.lambda_param * np.dot(self.w, self.w)
