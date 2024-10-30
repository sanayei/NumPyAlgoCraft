import pytest
from src.knn import KNN
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


@pytest.fixture
def knn_data():
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X, y


def test_knn_fit_predict(knn_data):
    X, y = knn_data
    k = 5

    # Your implementation
    my_knn = KNN(k=k, distance_metric="euclidean")
    my_knn.fit(X, y)
    my_predictions = my_knn.predict(X)

    # scikit-learn implementation
    sklearn_knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    sklearn_knn.fit(X, y)
    sklearn_predictions = sklearn_knn.predict(X)

    # Compare accuracy
    my_accuracy = accuracy_score(y, my_predictions)
    sklearn_accuracy = accuracy_score(y, sklearn_predictions)

    # Assert that your accuracy is within a reasonable range of scikit-learn's
    assert (
        abs(my_accuracy - sklearn_accuracy) < 0.1
    ), "K-NN accuracy does not match scikit-learn's implementation."
