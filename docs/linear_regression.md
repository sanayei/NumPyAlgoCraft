## Linear Regression
1. **Model Representation:**

   The hypothesis (prediction) function for linear regression can be expressed as:
  $$
   \hat{y} = X_b \cdot \theta
  $$
   where:
   -$\hat{y}$represents the vector of predictions.
   -$X_b$is the input matrix with an additional bias (column of ones) to account for the intercept term.
   -$\theta$is the vector of parameters (weights and bias).

2. **Error Calculation:**

   The difference between the predicted values and the actual target values ($y$) is given by:
  $$
   \text{errors} = \hat{y} - y = X_b \cdot \theta - y
  $$

3. **Cost Function:**

   The cost function for linear regression is the **Mean Squared Error (MSE)**:
  $$
   J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2
  $$
   where$m$is the number of training examples. The goal of gradient descent is to minimize this cost function.

4. **Gradient Calculation:**

   The gradient of the cost function with respect to each parameter$\theta_j$(for$j$ranging from$0$to$n$) is calculated as:
  $$
   \frac{\partial J}{\partial \theta_j} = \frac{2}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}
  $$

   **Matrix Form**: The above formula can be expressed in matrix form as:
  $$
   \text{gradient} = \frac{2}{m} X_b^T \cdot \text{errors}
  $$

   -$X_b^T$ is the transpose of the matrix $X_b$.
   -$\text{errors}$ is a vector containing the differences between predictions and actual values.

   This results in a gradient vector containing partial derivatives for each parameter in$\theta$.

5. **Gradient Descent Step:**

   Once we have calculated the gradient, we update the parameters$\theta$using:
  $$
   \theta = \theta - \text{learning\_rate} \times \text{gradient}
  $$

### Explanation in Code

- The **gradient** is calculated using the line:
  ```python
  gradient = (2 / m) * X_b.T.dot(errors)
  ```
  This line computes the derivative of the cost function with respect to each parameter in$\theta$. The matrix multiplication$X_b.T \cdot \text{errors}$aggregates the gradients across all training examples.

- The **parameter update** step uses:
  ```python
  self.theta -= self.lr * gradient
  ```
  This line applies the computed gradient scaled by the learning rate to update the parameters.