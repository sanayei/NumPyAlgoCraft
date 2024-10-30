import numpy as np
import pytest
from src.linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture
def regression_data():
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y


def test_linear_regression_fit_predict(regression_data):
    X, y = regression_data

    # Your implementation
    my_lr = LinearRegression(learning_rate=0.01, n_iterations=1000)
    my_lr.fit(X, y)
    my_predictions = my_lr.predict(X)

    # scikit-learn implementation
    sklearn_lr = SklearnLinearRegression()
    sklearn_lr.fit(X, y)
    sklearn_predictions = sklearn_lr.predict(X)

    # Compare coefficients using assert_allclose
    my_theta = my_lr.theta
    sklearn_theta = np.hstack([sklearn_lr.intercept_, sklearn_lr.coef_.flatten()])
    np.testing.assert_allclose(
        my_theta,
        sklearn_theta,
        rtol=1e-2,
        atol=1e-2,
        err_msg="Coefficients do not match scikit-learn's implementation.",
    )

    # Compare predictions using mean_squared_error
    my_mse = mean_squared_error(y, my_predictions)
    sklearn_mse = mean_squared_error(y, sklearn_predictions)
    assert my_mse == pytest.approx(sklearn_mse, rel=1e-2)
