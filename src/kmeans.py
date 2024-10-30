import numpy as np
# ----------------------------
#   K-Means Clustering
# ----------------------------


class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, init="random"):
        self.k = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):
        if self.init == "random":
            indices = np.random.choice(X.shape[0], self.k, replace=False)
            return X[indices]
        elif self.init == "kmeans++":
            centroids = [X[np.random.randint(0, X.shape[0])]]
            for _ in range(1, self.k):
                dist_sq = np.min(
                    [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0
                )
                probabilities = dist_sq / dist_sq.sum()
                cumulative_prob = np.cumsum(probabilities)
                r = np.random.rand()
                for idx, prob in enumerate(cumulative_prob):
                    if r < prob:
                        centroids.append(X[idx])
                        break
            return np.array(centroids)
        else:
            raise ValueError("Unsupported initialization method.")

    def compute_distances(self, X, centroids):
        return np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)

        for i in range(self.max_iters):
            distances = self.compute_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array(
                [
                    X[labels == j].mean(axis=0)
                    if len(X[labels == j]) > 0
                    else self.centroids[j]
                    for j in range(self.k)
                ]
            )
            shift = np.linalg.norm(self.centroids - new_centroids)
            print(f"Iteration {i}: centroid shift {shift}")
            if shift < self.tol:
                break
            self.centroids = new_centroids
        self.labels = labels

    def predict(self, X):
        distances = self.compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
