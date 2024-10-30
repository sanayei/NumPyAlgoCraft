## Logistic Regression

**Logistic Regression** is a supervised learning algorithm used for binary classification problems. It models the probability that a given input $\mathbf{X}$ belongs to a particular class (e.g., class 1) using a sigmoid function. The output is a probability value between 0 and 1.

#### 1. **Sigmoid Function**:
The logistic regression model uses the sigmoid function to map the output of the linear combination of inputs to a probability:

$$
\text{sigmoid}(z) = \frac{1}{1 + e^{-z}}
$$

where:
- $z = \mathbf{X_b} \cdot \boldsymbol{\theta}$
- $\mathbf{X_b}$ is the input matrix with an additional bias term (a column of ones).
- $\boldsymbol{\theta}$ is the parameter vector.

### 2. **Hypothesis (Prediction) Function**:
The hypothesis for logistic regression is defined as:

$$
h_{\boldsymbol{\theta}}(\mathbf{X}) = \text{sigmoid}(\mathbf{X_b} \cdot \boldsymbol{\theta}) = \frac{1}{1 + e^{-\mathbf{X_b} \cdot \boldsymbol{\theta}}}
$$

### 3. **Cost Function**:
Logistic regression uses a **log loss** or **cross-entropy loss** function to measure the error between predicted values and actual values. The cost function $J(\boldsymbol{\theta})$ is given by:

$$
J(\boldsymbol{\theta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_{\boldsymbol{\theta}}(\mathbf{X}^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\boldsymbol{\theta}}(\mathbf{X}^{(i)})) \right]
$$

where:
- $m$ is the number of training examples.
- $y^{(i)}$ is the true label (either 0 or 1).
- $h_{\boldsymbol{\theta}}(\mathbf{X}^{(i)})$ is the predicted probability for the $i$-th example.

### 4. **Gradient Calculation**:
The gradient of the cost function with respect to each parameter $\theta_j$ is given by:

$$
\frac{\partial J(\boldsymbol{\theta})}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left[ h_{\boldsymbol{\theta}}(\mathbf{X}^{(i)}) - y^{(i)} \right] \cdot X_j^{(i)}
$$

**In Matrix Form**:
$$
\text{gradient} = \frac{1}{m} \mathbf{X_b}^T \cdot \left( h_{\boldsymbol{\theta}}(\mathbf{X_b}) - \mathbf{y} \right)
$$

where:
- $\mathbf{X_b}$ is the augmented input matrix with a column of ones.
- $h_{\boldsymbol{\theta}}(\mathbf{X_b})$ is the vector of predictions.
- $\mathbf{y}$ is the vector of true labels.

### 5. **Gradient Descent Step**:
To update the parameters, we subtract the product of the learning rate and the gradient from the current parameter values:

$$
\boldsymbol{\theta} = \boldsymbol{\theta} - \text{learning\_rate} \times \text{gradient}
$$

### Explanation in the Code

#### 1. **Sigmoid Function Calculation**:
The `sigmoid` function in the class is implemented as:

```python
def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
```

#### 2. **Gradient Calculation**:
In the `fit` method, the gradient is calculated using:

```python
gradient = (1 / m) * X_b.T.dot(errors)
```

where:
- `errors` is the difference between the predicted probabilities and the true labels ($\text{errors} = \text{predictions} - y$).
- The matrix multiplication $X_b.T$ with the `errors` vector computes the sum of gradients for all training examples in a vectorized form.

#### 3. **Parameter Update**:
The parameters are updated using:

```python
self.theta -= self.lr * gradient
```

This step reduces the value of $\theta$ in the direction of the negative gradient, thus minimizing the cost function.