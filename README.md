# PyCraft

**PyCraft** is a comprehensive repository dedicated to implementing essential data science and machine learning algorithms from scratch using Python and NumPy. Whether you're a beginner aiming to deepen your understanding of algorithmic foundations or an experienced practitioner seeking to refine your coding skills, this repository serves as a resource for mastering the core concepts that drive the data science field.

## 🛠️ **Features**

- **Pure Python & NumPy Implementations**: All algorithms are built from the ground up without relying on high-level libraries, ensuring a thorough grasp of underlying mechanics.
- **Class-Based Structure**: Each algorithm is encapsulated within its own class, promoting modularity, readability, and ease of maintenance.
- **Comprehensive Coverage**: Includes a wide range of algorithms spanning supervised and unsupervised learning, optimization, clustering, and more.
- **Usage Examples**: Demonstrative scripts and notebooks showcasing how to utilize each algorithm with synthetic and real datasets.
- **Documentation**: Detailed comments and docstrings elucidate the functionality and implementation details of each class and method.
- **Testing Suite**: Unit tests to verify the correctness and robustness of algorithm implementations.

## 📚 **Included Algorithms**

1. **Linear Regression**
2. **Logistic Regression**
3. **K-Nearest Neighbors (K-NN)**
4. **Decision Trees**
5. **K-Means Clustering**
6. **Principal Component Analysis (PCA)**
7. **Naive Bayes Classifier**
8. **Support Vector Machines (SVM)**
9. **Gradient Descent Optimizer & Variants**
10. **Collaborative Filtering**
11. **Hidden Markov Models (HMM)**
12. **Apriori Algorithm**
13. **Hierarchical Clustering**
14. **Basic Neural Networks**

## 🚀 **Getting Started**

### **Prerequisites**

- Python 3.6 or higher
- NumPy
- scikit-learn to compare results also generating synthetic datasets in examples
- (Optional) Matplotlib for visualization

### **Installation**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/sanayei/PyCraft.git
   cd PyCraft
   ```

2. **Install Dependencies**

   It's recommended to use a virtual environment:

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not provided, install the necessary packages manually:*

   ```bash
   pip install numpy matplotlib scikit-learn
   ```

### **Usage**

Each algorithm is implemented as a separate class within the `src/` directory. Here's a quick example of how to use the **Linear Regression** class:

```python
import numpy as np
from src.linear_regression import LinearRegression
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=20)
y = y.reshape(-1, 1)

# Initialize and train the model
lr = LinearRegression(learning_rate=0.01, n_iterations=1000)
lr.fit(X, y)

# Make predictions
predictions = lr.predict(X)

# Plot the results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, predictions, color='red', label='Linear Regression')
plt.legend()
plt.show()
```

*Similar usage patterns apply to other algorithms. Refer to the `examples/` directory for more detailed scripts and notebooks.*

## 📂 **Repository Structure**

```
NumPyAlgoCraft/
├── src/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── kmeans.py
│   ├── pca.py
│   ├── naive_bayes.py
│   ├── svm.py
│   ├── gradient_descent.py
│   ├── collaborative_filtering.py
│   ├── hmm.py
│   ├── apriori.py
│   ├── hierarchical_clustering.py
│   └── neural_network.py
├── examples/
│   ├── linear_regression_example.py
│   ├── logistic_regression_example.py
│   └── ... (additional example scripts)
├── tests/
│   ├── test_linear_regression.py
│   ├── test_logistic_regression.py
│   └── ... (additional test scripts)
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```

## 🤝 **Contributing**

Contributions are welcome! Whether it's improving existing implementations, adding new algorithms, or enhancing documentation, your efforts help make this repository a valuable learning tool for the community.

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## 📜 **License**

This project is licensed under the [MIT License](LICENSE).

## 📫 **Contact**

Amir Sanayei  
Ph.D. in Operations Research and Industrial Engineering  
Machine Learning & AI Practitioner  
[LinkedIn](https://www.linkedin.com/in/asanayei/) | [Email](mailto:asanayei@gmail.com)

---

Embark on a journey to demystify the algorithms that power the data-driven world. **PyCraft** empowers you to build, understand, and innovate with foundational data science techniques, all through the elegance and efficiency of NumPy and Python.

Happy Coding! 🚀