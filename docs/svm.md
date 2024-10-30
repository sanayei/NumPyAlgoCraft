## Support Vector Machine (SVM)

**Support Vector Machine (SVM)** is a supervised learning algorithm primarily used for binary classification tasks. SVM finds an optimal hyperplane that separates data points from different classes with the maximum margin. The key idea behind SVM is to find a decision boundary that maximizes the distance (or margin) between the two classes, which improves the generalization ability of the classifier.

### Key Components of SVM

1. **Hyperplane**: A hyperplane is a decision boundary that separates the classes in the feature space. In a 2D space, this is a line; in a 3D space, it's a plane; and in higher dimensions, it's called a hyperplane.

2. **Margin**: The margin is the distance between the hyperplane and the nearest data points from either class. SVM aims to find a hyperplane with the **maximum margin** to the closest points from each class. These closest points are called **support vectors**.

3. **Hinge Loss**: SVM uses hinge loss to measure the misclassification error. It penalizes points that lie within the margin or on the wrong side of the decision boundary.

### Steps in SVM Algorithm

1. **Convert Labels**: SVM works with labels $\{-1, 1\}$ instead of $\{0, 1\}$. Convert all 0s to -1.
2. **Initialize Parameters**: Initialize the weights $\mathbf{w}$ and bias $b$ to zero.
3. **Optimize the Objective Function**: Use gradient descent to minimize the objective function, which includes the hinge loss and regularization terms.
4. **Predict**: Use the learned weights $\mathbf{w}$ and bias $b$ to make predictions on new data points.

### Mathematical Formulas for SVM

1. **Decision Function**:
   The decision function (or score) for a data point $x_i$ is given by:
   $$
   f(x_i) = \mathbf{w} \cdot x_i - b
   $$

2. **Hinge Loss**:
   The hinge loss function is defined as:
   $$
   L(y_i, f(x_i)) = \max(0, 1 - y_i \cdot f(x_i))
   $$
   where $y_i$ is the true label of $x_i$ (which is either -1 or 1). The hinge loss penalizes misclassified points and those within the margin.

3. **Objective Function**:
   The SVM objective function combines hinge loss with a regularization term to prevent overfitting:
   $$
   J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^m \max(0, 1 - y_i \cdot f(x_i)) + \lambda \lVert \mathbf{w} \rVert^2
   $$
   where:
   - $\lambda$ is the regularization parameter.
   - $\lVert \mathbf{w} \rVert^2$ is the squared norm of the weight vector, which acts as a regularization term.

4. **Gradient Descent Update**:
   The weight update rule depends on whether a point lies inside or outside the margin:

   - If $y_i \cdot f(x_i) \geq 1$ (correctly classified and outside the margin):
     $$
     \mathbf{w} = \mathbf{w} - \eta \cdot 2\lambda \mathbf{w}
     $$
   
   - If $y_i \cdot f(x_i) < 1$ (within the margin or misclassified):
     $$
     \mathbf{w} = \mathbf{w} - \eta \cdot (2\lambda \mathbf{w} - y_i \cdot x_i)
     $$
     $$
     b = b - \eta \cdot (-y_i)
     $$

   where $\eta$ is the learning rate.

### Explanation of the Code

1. **Label Conversion**:
   The `fit` method converts the labels to be either -1 or 1:

   ```python
   y_ = np.where(y <= 0, -1, 1)
   ```

2. **Initialization**:
   The weights $\mathbf{w}$ and bias $b$ are initialized to zero:

   ```python
   self.w = np.zeros(n_features)
   self.b = 0
   ```

3. **Gradient Descent**:
   For each data point $x_i$, the algorithm checks if it lies within or outside the margin and updates the weights and bias accordingly:

   ```python
   condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
   if condition:
       self.w -= self.lr * (2 * self.lambda_param * self.w)
   else:
       self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
       self.b -= self.lr * y_[idx]
   ```

4. **Hinge Loss Calculation**:
   The hinge loss is calculated in the `_compute_loss` method:

   ```python
   distances = 1 - y * (np.dot(X, self.w) - self.b)
   distances = np.where(distances > 0, distances, 0)
   hinge_loss = np.mean(distances)
   return hinge_loss + self.lambda_param * np.dot(self.w, self.w)
   ```

   This method computes the distances of each point from the margin, averages the hinge loss, and adds the regularization term.

5. **Prediction**:
   The `predict` method uses the learned weights and bias to make predictions:

   ```python
   approx = np.dot(X, self.w) - self.b
   return np.sign(approx)
   ```

### Summary of Key Components and Formulas

1. **Decision Function**:
   $$
   f(x_i) = \mathbf{w} \cdot x_i - b
   $$

2. **Hinge Loss**:
   $$
   L(y_i, f(x_i)) = \max(0, 1 - y_i \cdot f(x_i))
   $$

3. **Objective Function**:
   $$
   J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^m \max(0, 1 - y_i \cdot f(x_i)) + \lambda \lVert \mathbf{w} \rVert^2
   $$

4. **Weight Update Rules**:
   - **Outside the Margin**:
     $$
     \mathbf{w} = \mathbf{w} - \eta \cdot 2\lambda \mathbf{w}
     $$

   - **Inside the Margin or Misclassified**:
     $$
     \mathbf{w} = \mathbf{w} - \eta \cdot (2\lambda \mathbf{w} - y_i \cdot x_i)
     $$
     $$
     b = b - \eta \cdot (-y_i)
     $$

### Explanation in Words

SVM aims to find a hyperplane that best separates the data points of different classes. It focuses on the data points that are closest to the decision boundary, called the support vectors. By maximizing the margin between classes, SVM improves generalization to unseen data. The inclusion of a regularization term prevents overfitting and ensures that the model does not become too complex.

The hinge loss allows SVM to penalize points that are within the margin or misclassified, ensuring that the model adjusts its decision boundary to best separate the classes.