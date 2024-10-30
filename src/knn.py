import numpy as np
from collections import Counter

# ----------------------------
# 3. K-Nearest Neighbors (K-NN)
# ----------------------------

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distance(self, x):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(self.X_train - x), axis=1)
        else:
            raise ValueError("Unsupported distance metric.")

    def predict(self, X):
        predictions = []
        for x in X:
            distances = self.compute_distance(x)
            neighbor_idxs = np.argsort(distances)[:self.k]
            neighbor_labels = self.y_train[neighbor_idxs]
            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)
