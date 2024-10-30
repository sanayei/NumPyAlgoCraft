import numpy as np
# ----------------------------
#   Collaborative Filtering
# ----------------------------


class CollaborativeFiltering:
    def __init__(
        self, n_factors=10, learning_rate=0.01, n_iterations=1000, regularization=0.1
    ):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.reg = regularization
        self.P = None  # User latent factors
        self.Q = None  # Item latent factors

    def fit(self, X, n_users, n_items):
        self.P = np.random.normal(
            scale=1.0 / self.n_factors, size=(n_users, self.n_factors)
        )
        self.Q = np.random.normal(
            scale=1.0 / self.n_factors, size=(n_items, self.n_factors)
        )

        for iteration in range(self.n_iterations):
            np.random.shuffle(X)
            total_loss = 0
            for user, item, rating in X:
                prediction = self.P[user].dot(self.Q[item].T)
                error = rating - prediction
                total_loss += error**2
                # Update latent factors
                self.P[user] += self.lr * (
                    error * self.Q[item] - self.reg * self.P[user]
                )
                self.Q[item] += self.lr * (
                    error * self.P[user] - self.reg * self.Q[item]
                )
            loss = total_loss / len(X)
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss {loss}")

    def predict(self, user, item):
        return self.P[user].dot(self.Q[item].T)
