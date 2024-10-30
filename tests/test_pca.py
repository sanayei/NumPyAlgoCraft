import numpy as np
import pytest
from src.pca import PCA
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def pca_data():
    X, _ = make_blobs(
        n_samples=100, n_features=5, centers=3, cluster_std=1.0, random_state=42
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def test_pca_fit_transform(pca_data):
    X = pca_data
    n_components = 2

    # Your implementation
    my_pca = PCA(n_components=n_components)
    X_reduced_my = my_pca.fit_transform(X)

    # scikit-learn implementation
    sklearn_pca = SklearnPCA(n_components=n_components)
    X_reduced_sklearn = sklearn_pca.fit_transform(X)

    # Compare explained variance ratios
    my_explained_variance_ratio = np.sum(my_pca.explained_variance_) / np.sum(
        np.var(X, axis=0, ddof=1)
    )
    sklearn_explained_variance_ratio = np.sum(sklearn_pca.explained_variance_ratio_)

    assert (
        abs(my_explained_variance_ratio - sklearn_explained_variance_ratio) < 0.1
    )  # 10% tolerance

    # Compare transformed data (up to a rotation/scaling factor)
    # Since PCA can differ by sign or rotation, a direct comparison isn't straightforward
    # Instead, compare pairwise distances or angles
    from sklearn.metrics import pairwise_distances

    distance_my = pairwise_distances(X_reduced_my)
    distance_sklearn = pairwise_distances(X_reduced_sklearn)
    assert np.allclose(
        distance_my, distance_sklearn, atol=1e-1
    )  # Adjust tolerance as needed
