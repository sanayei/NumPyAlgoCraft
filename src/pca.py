import numpy as np
# ----------------------------
#   Principal Component Analysis (PCA)
# ----------------------------


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ = None

    def fit(self, X):
        # Standardize the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        # Compute covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        # Sort eigenvectors by decreasing eigenvalues
        sorted_idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_idx]
        sorted_eigenvalues = eigenvalues[sorted_idx]
        # Select the top n_components
        self.components = sorted_eigenvectors[:, : self.n_components]
        # Store the explained variance for the selected components
        self.explained_variance_ = sorted_eigenvalues[: self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered.dot(self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
