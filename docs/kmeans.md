## K-Means Clustering

**K-Means** is an unsupervised clustering algorithm that partitions a dataset into $k$ distinct, non-overlapping subsets (or clusters) based on similarity. It aims to minimize the sum of squared distances between each data point and the centroid of its assigned cluster.

### Key Components of K-Means

1. **Centroids**: These are the central points of each cluster. Each cluster has one centroid that represents the "average" of the points within that cluster.

2. **Labels**: Each data point is assigned to the cluster whose centroid is closest to it.

3. **Distance Metric**: The distance between each data point and the centroids is usually calculated using the **Euclidean distance** formula.

### Steps in K-Means Algorithm

1. **Initialization**:
   The initial centroids are chosen based on one of the following methods:
   - **Random Initialization**: Randomly select $k$ points from the dataset as initial centroids.
   - **K-Means++ Initialization**: A smarter way to initialize centroids by spreading them out in a way that increases the likelihood of convergence and stability.

2. **Assignment Step**:
   For each data point $x_i$, assign it to the cluster with the nearest centroid. This is calculated as:
   $$
   \text{label}(x_i) = \arg\min_{j} \left\{ \lVert x_i - c_j \rVert^2 \right\}
   $$
   where:
   - $c_j$ is the centroid of the $j$-th cluster.
   - $\lVert x_i - c_j \rVert^2$ is the squared Euclidean distance between the point $x_i$ and the centroid $c_j$.

3. **Update Step**:
   Recalculate the centroids of each cluster by taking the mean of all the points assigned to that cluster:
   $$
   c_j = \frac{1}{N_j} \sum_{x_i \in C_j} x_i
   $$
   where:
   - $N_j$ is the number of points in cluster $C_j$.
   - $x_i$ represents a point belonging to the $j$-th cluster.

4. **Convergence Check**:
   Calculate the **shift** in the centroids’ positions and check if it is below a specified tolerance ($\text{tol}$). The shift is computed as:
   $$
   \text{shift} = \lVert \mathbf{c}_{\text{old}} - \mathbf{c}_{\text{new}} \rVert
   $$
   If the shift is less than the tolerance, the algorithm stops; otherwise, it repeats the assignment and update steps.

### Mathematical Formulas for K-Means

1. **Euclidean Distance**:
   The distance between a point $x_i$ and a centroid $c_j$ is given by:
   $$
   \text{distance}(x_i, c_j) = \sqrt{\sum_{k=1}^{n} (x_{ik} - c_{jk})^2}
   $$

2. **Objective Function (Within-Cluster Sum of Squares)**:
   K-Means aims to minimize the sum of squared distances from each point to its assigned centroid:
   $$
   J = \sum_{j=1}^{k} \sum_{x_i \in C_j} \lVert x_i - c_j \rVert^2
   $$

3. **Centroid Update**:
   The new centroid for cluster $j$ is computed as:
   $$
   c_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
   $$

### Explanation of the Code

1. **Initialization**:
   The centroids are initialized using either random selection or K-Means++ initialization:

   ```python
   def initialize_centroids(self, X):
       if self.init == "random":
           indices = np.random.choice(X.shape[0], self.k, replace=False)
           return X[indices]
       elif self.init == "kmeans++":
           # K-Means++ logic
   ```

2. **Distance Calculation**:
   The code uses the Euclidean distance formula to calculate the distance between each data point and the centroids:

   ```python
   def compute_distances(self, X, centroids):
       return np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
   ```

3. **Fit Method (Training)**:
   The code iteratively assigns data points to the nearest centroid, updates the centroids based on these assignments, and checks for convergence:

   ```python
   def fit(self, X):
       self.centroids = self.initialize_centroids(X)
       for i in range(self.max_iters):
           distances = self.compute_distances(X, self.centroids)
           labels = np.argmin(distances, axis=1)
           new_centroids = np.array([
               X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 else self.centroids[j]
               for j in range(self.k)
           ])
           shift = np.linalg.norm(self.centroids - new_centroids)
           if shift < self.tol:
               break
           self.centroids = new_centroids
       self.labels = labels
   ```

   - **Convergence Check**: The code calculates the shift and compares it to the tolerance to determine if the algorithm should stop.

4. **Prediction**:
   The code assigns new data points to the nearest cluster based on the learned centroids:

   ```python
   def predict(self, X):
       distances = self.compute_distances(X, self.centroids)
       return np.argmin(distances, axis=1)
   ```