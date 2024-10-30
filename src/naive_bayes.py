import numpy as np
# ----------------------------
#   Naive Bayes Classifier
# ----------------------------


class NaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.vars = {}
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.means[c] = X_c.mean(axis=0)
            self.vars[c] = X_c.var(axis=0) + self.var_smoothing

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            class_conditional = -0.5 * np.sum(np.log(2 * np.pi * self.vars[c]))
            class_conditional -= 0.5 * np.sum(((x - self.means[c]) ** 2) / self.vars[c])
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
