from src.kmeans import KMeans
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
import warnings
import pytest

# Suppress any deprecation warnings if necessary
warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture
def clustering_data():
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    return X, y


def test_kmeans_fit_predict(clustering_data):
    X, y_true = clustering_data
    n_clusters = 4

    # Your implementation
    my_kmeans = KMeans(n_clusters=n_clusters, max_iters=100, tol=1e-4, init="kmeans++")
    my_kmeans.fit(X)
    my_labels = my_kmeans.predict(X)

    # scikit-learn implementation
    sklearn_kmeans = SklearnKMeans(
        n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=300, random_state=0
    )
    sklearn_kmeans.fit(X)
    sklearn_labels = sklearn_kmeans.predict(X)

    # Compare Adjusted Rand Index
    ari = adjusted_rand_score(y_true, my_labels)
    sklearn_ari = adjusted_rand_score(y_true, sklearn_labels)

    # Assert that your ARI is within a reasonable range of scikit-learn's
    assert (
        abs(ari - sklearn_ari) < 0.1
    ), f"K-Means Adjusted Rand Index ({ari}) does not match scikit-learn's implementation ({sklearn_ari})."
