### Theory of K-Nearest Neighbors (KNN)

**K-Nearest Neighbors (KNN)** is a simple, non-parametric, supervised learning algorithm used for both classification and regression. It works based on the idea that similar data points are located close to each other. In KNN, the predicted class or value of a data point is determined by the majority class (for classification) or average value (for regression) of its $k$-nearest neighbors.

### Key Components of KNN

1. **Distance Metric**: KNN relies on calculating the distance between data points to determine which ones are closest to each other. Common distance metrics include:
   - **Euclidean Distance**: The straight-line distance between two points in multidimensional space.
   - **Manhattan Distance**: The sum of the absolute differences between coordinates of two points.

2. **Number of Neighbors ($k$)**: The value of $k$ determines how many neighbors to consider when making a prediction.

3. **Voting Mechanism (for Classification)**: In KNN classification, the algorithm predicts the class of a data point based on the majority class among its $k$-nearest neighbors.

### Steps in the KNN Algorithm

1. **Store the Training Data**: The algorithm does not create a model during training. Instead, it memorizes the entire training dataset.

2. **Compute Distance**: For a given new data point, the algorithm computes the distance to all data points in the training dataset.

3. **Select $k$ Nearest Neighbors**: Identify the $k$-closest points based on the chosen distance metric.

4. **Vote or Average**:
   - **Classification**: The most common class among the $k$-nearest neighbors is chosen as the predicted class.
   - **Regression**: The average of the $k$-nearest neighbors is taken as the predicted value.

### Mathematical Formulas for KNN

1. **Euclidean Distance**:
   The Euclidean distance between two points $x$ and $y$ in $n$-dimensional space is given by:
   $$
   d_{\text{euclidean}}(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
   $$

2. **Manhattan Distance**:
   The Manhattan distance between two points $x$ and $y$ is given by:
   $$
   d_{\text{manhattan}}(x, y) = \sum_{i=1}^{n} |x_i - y_i|
   $$

3. **Voting for Classification**:
   Let $N_k$ denote the set of $k$-nearest neighbors of a point $x$. The predicted class $\hat{y}$ for point $x$ is given by:
   $$
   \hat{y} = \text{mode}( \{ y_i \mid x_i \in N_k \} )
   $$
   where $y_i$ is the class label of neighbor $x_i$.

4. **Averaging for Regression**:
   In the case of regression, the predicted value $\hat{y}$ is the average of the target values of the $k$-nearest neighbors:
   $$
   \hat{y} = \frac{1}{k} \sum_{x_i \in N_k} y_i
   $$

### Explanation of the Code

1. **Distance Calculation**:
   The code supports both Euclidean and Manhattan distances using the `compute_distance` method:

   ```python
   def compute_distance(self, x):
       if self.distance_metric == 'euclidean':
           return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
       elif self.distance_metric == 'manhattan':
           return np.sum(np.abs(self.X_train - x), axis=1)
       else:
           raise ValueError("Unsupported distance metric.")
   ```

   - **Euclidean Distance** is computed as the square root of the sum of squared differences.
   - **Manhattan Distance** is computed as the sum of the absolute differences.

2. **Prediction**:
   The algorithm iterates over all test samples and computes the distances to all training points. It then selects the $k$-closest points and uses them to predict the class label:

   ```python
   def predict(self, X):
       predictions = []
       for x in X:
           distances = self.compute_distance(x)
           neighbor_idxs = np.argsort(distances)[:self.k]
           neighbor_labels = self.y_train[neighbor_idxs]
           most_common = Counter(neighbor_labels).most_common(1)[0][0]
           predictions.append(most_common)
       return np.array(predictions)
   ```

   - **Distance Calculation**: For each test point $x$, the algorithm calculates the distance between $x$ and all training points.
   - **Sorting and Selecting Neighbors**: The indices of the $k$-nearest points are determined using `np.argsort`.
   - **Voting**: The class label with the highest count among the $k$-nearest neighbors is chosen as the predicted class.

### Summary of Key Components and Formulas

1. **Distance Metrics**:
   - **Euclidean Distance**:
     $$
     d_{\text{euclidean}}(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
     $$

   - **Manhattan Distance**:
     $$
     d_{\text{manhattan}}(x, y) = \sum_{i=1}^{n} |x_i - y_i|
     $$

2. **Voting (for Classification)**:
   $$
   \hat{y} = \text{mode}( \{ y_i \mid x_i \in N_k \} )
   $$

3. **Averaging (for Regression)**:
   $$
   \hat{y} = \frac{1}{k} \sum_{x_i \in N_k} y_i
   $$

4. **Algorithm Steps**:
   1. Store the training dataset.
   2. Compute distances between test points and all training points.
   3. Identify $k$-nearest neighbors.
   4. Vote for classification or average for regression.

By following these steps, KNN provides a simple yet effective way to classify or predict based on the proximity of data points in the feature space.