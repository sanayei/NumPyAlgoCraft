### Theory of Principal Component Analysis (PCA)

**Principal Component Analysis (PCA)** is an unsupervised dimensionality reduction technique used to reduce the number of features while retaining most of the variance present in the dataset. PCA achieves this by finding a set of new orthogonal axes, known as *principal components*, which are directions in the feature space that capture the maximum variance.

### Key Components of PCA

1. **Mean-Centering**: Before performing PCA, the dataset is centered by subtracting the mean of each feature.
2. **Covariance Matrix**: PCA analyzes the covariance between features to determine how much they vary together.
3. **Eigenvectors and Eigenvalues**: The eigenvectors of the covariance matrix represent the directions of maximum variance, while the corresponding eigenvalues represent the magnitude of this variance.
4. **Principal Components**: The eigenvectors with the largest eigenvalues are chosen as the principal components, representing the directions with the most significant variance.

### Steps in PCA Algorithm

1. **Standardize the Data**: Subtract the mean of each feature to center the dataset around the origin.
2. **Compute the Covariance Matrix**: Calculate the covariance matrix to understand the relationships between the features.
3. **Calculate Eigenvalues and Eigenvectors**: Compute the eigenvalues and eigenvectors of the covariance matrix to find the directions of maximum variance.
4. **Sort and Select Principal Components**: Sort the eigenvalues and select the eigenvectors corresponding to the top $k$ eigenvalues.
5. **Project the Data**: Transform the original dataset to the new feature space defined by the selected principal components.

### Mathematical Formulas for PCA

1. **Standardization** (Mean-Centering):
   Let $X$ be the data matrix with $m$ samples and $n$ features. The mean-centered data matrix $X_{\text{centered}}$ is given by:
   $$
   X_{\text{centered}} = X - \text{mean}(X)
   $$

2. **Covariance Matrix**:
   The covariance matrix $\Sigma$ is computed as:
   $$
   \Sigma = \frac{1}{m-1} X_{\text{centered}}^T X_{\text{centered}}
   $$
   where:
   - $X_{\text{centered}}$ is the mean-centered data matrix.

3. **Eigenvalues and Eigenvectors**:
   The eigenvalues $\lambda_i$ and eigenvectors $v_i$ of the covariance matrix $\Sigma$ are obtained by solving:
   $$
   \Sigma v_i = \lambda_i v_i
   $$

4. **Sorting Eigenvectors**:
   Sort the eigenvectors $v_i$ based on their corresponding eigenvalues $\lambda_i$ in descending order. The eigenvectors with the largest eigenvalues represent the principal components.

5. **Principal Component Selection**:
   Select the top $k$ eigenvectors, corresponding to the $k$ largest eigenvalues, to form the principal components matrix $W$:
   $$
   W = [v_1, v_2, \ldots, v_k]
   $$

6. **Projection**:
   Project the original data $X$ onto the principal components to obtain the transformed data matrix $Z$:
   $$
   Z = X_{\text{centered}} W
   $$

### Explanation of the Code

1. **Standardization (Mean-Centering)**:
   The `fit` method starts by centering the data:

   ```python
   self.mean = np.mean(X, axis=0)
   X_centered = X - self.mean
   ```

2. **Compute Covariance Matrix**:
   The covariance matrix is computed using:

   ```python
   covariance_matrix = np.cov(X_centered, rowvar=False)
   ```

3. **Calculate Eigenvalues and Eigenvectors**:
   The code uses the numpy function `np.linalg.eigh` to compute eigenvalues and eigenvectors of the covariance matrix:

   ```python
   eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
   ```

   Here, `np.linalg.eigh` is used because the covariance matrix is symmetric.

4. **Sort and Select Principal Components**:
   The eigenvectors are sorted in descending order of their corresponding eigenvalues, and the top `n_components` are selected:

   ```python
   sorted_idx = np.argsort(eigenvalues)[::-1]
   sorted_eigenvectors = eigenvectors[:, sorted_idx]
   sorted_eigenvalues = eigenvalues[sorted_idx]
   self.components = sorted_eigenvectors[:, : self.n_components]
   self.explained_variance_ = sorted_eigenvalues[: self.n_components]
   ```

5. **Project the Data**:
   The `transform` method projects the data onto the selected principal components:

   ```python
   X_centered = X - self.mean
   return X_centered.dot(self.components)
   ```

### Summary of Key Components and Formulas

1. **Mean-Centering**:
   $$
   X_{\text{centered}} = X - \text{mean}(X)
   $$

2. **Covariance Matrix**:
   $$
   \Sigma = \frac{1}{m-1} X_{\text{centered}}^T X_{\text{centered}}
   $$

3. **Eigenvalues and Eigenvectors**:
   $$
   \Sigma v_i = \lambda_i v_i
   $$

4. **Principal Component Selection**:
   The eigenvectors corresponding to the largest eigenvalues are selected as the principal components.

5. **Projection**:
   $$
   Z = X_{\text{centered}} W
   $$

### Explanation in Words

PCA transforms the dataset into a new coordinate system where the greatest variance in the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on. By retaining only the top principal components, PCA reduces the dimensionality of the data while preserving as much variability as possible.

This makes PCA an essential technique for tasks like visualization, noise reduction, and improving the performance of machine learning models by reducing the number of features.