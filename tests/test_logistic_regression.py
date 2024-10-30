import pytest
from src.logistic_regression import LogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X, y


def test_logistic_regression_fit_predict(classification_data):
    X, y = classification_data

    # Your implementation
    my_lr = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    my_lr.fit(X, y)
    my_predictions = my_lr.predict(X)

    # scikit-learn implementation
    sklearn_lr = SklearnLogisticRegression(max_iter=1000)
    sklearn_lr.fit(X, y)
    sklearn_predictions = sklearn_lr.predict(X)

    # Compare accuracy
    my_accuracy = accuracy_score(y, my_predictions)
    sklearn_accuracy = accuracy_score(y, sklearn_predictions)

    # Assert that your accuracy is within a reasonable range of scikit-learn's
    assert (
        abs(my_accuracy - sklearn_accuracy) < 0.1
    ), "Logistic Regression accuracy does not match scikit-learn's implementation."
