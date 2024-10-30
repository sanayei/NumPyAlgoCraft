import numpy as np
# ----------------------------
#    Decision Trees
# ----------------------------


class DecisionTreeNode:
    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes


class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def information_gain(self, y, y_left, y_right):
        parent_entropy = self.entropy(y)
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        if n_left == 0 or n_right == 0:
            return 0
        child_entropy = (n_left / n) * self.entropy(y_left) + (
            n_right / n
        ) * self.entropy(y_right)
        return parent_entropy - child_entropy

    def best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        parent_entropy = self.entropy(y)
        best_gain = 0
        split_idx, split_threshold = None, None

        for feature_index in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, feature_index], y)))
            for i in range(1, m):
                if classes[i] == classes[i - 1]:
                    continue
                threshold = (thresholds[i] + thresholds[i - 1]) / 2
                y_left, y_right = y[:i], y[i:]
                gain = self.information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_index
                    split_threshold = threshold

        return split_idx, split_threshold

    def build_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = DecisionTreeNode(value=predicted_class)

        if depth < self.max_depth:
            split_idx, split_threshold = self.best_split(X, y)
            if split_idx is not None:
                indices_left = X[:, split_idx] < split_threshold
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                if (
                    len(y_left) >= self.min_samples_split
                    and len(y_right) >= self.min_samples_split
                ):
                    node.feature_index = split_idx
                    node.threshold = split_threshold
                    node.left = self.build_tree(X_left, y_left, depth + 1)
                    node.right = self.build_tree(X_right, y_right, depth + 1)
                    node.value = None
        return node

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] < node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.root) for x in X])
