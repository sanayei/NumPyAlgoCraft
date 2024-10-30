import pytest
from src.svm import SVM
from sklearn.datasets import make_classification
import numpy as np
from sklearn.svm import SVC as SklearnSVC
from sklearn.metrics import accuracy_score


@pytest.fixture
def svm_data():
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )
    y = np.where(y == 0, -1, 1)  # Convert to {-1, 1} for SVM
    return X, y


def test_svm_fit_predict(svm_data):
    X, y = svm_data

    # Your implementation
    my_svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
    my_svm.fit(X, y)
    my_predictions = my_svm.predict(X)

    # scikit-learn implementation
    sklearn_svm = SklearnSVC(kernel="linear", C=1.0, max_iter=1000)
    sklearn_svm.fit(X, y)
    sklearn_predictions = sklearn_svm.predict(X)

    # Compare accuracy
    my_accuracy = accuracy_score(y, my_predictions)
    sklearn_accuracy = accuracy_score(y, sklearn_predictions)

    # Assert that your accuracy is within a reasonable range of scikit-learn's
    assert (
        abs(my_accuracy - sklearn_accuracy) < 0.1
    ), "SVM accuracy does not match scikit-learn's implementation."
