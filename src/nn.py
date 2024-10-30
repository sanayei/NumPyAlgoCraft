import numpy as np
# ----------------------------
#   Basic Neural Networks
# ----------------------------


class NeuralNetwork:
    def __init__(
        self, layers, activation="relu", output_activation="sigmoid", learning_rate=0.01
    ):
        self.layers = layers  # List containing the number of units in each layer
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation
        self.weights = []
        self.biases = []
        self.initialize_parameters()

    def initialize_parameters(self):
        for i in range(len(self.layers) - 1):
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(
                2.0 / self.layers[i]
            )
            bias = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, A):
        return A * (1 - A)

    def forward(self, X):
        self.Z = []
        self.A = [X]
        for i in range(len(self.weights)):
            Z = np.dot(self.A[-1], self.weights[i]) + self.biases[i]
            self.Z.append(Z)
            if i == len(self.weights) - 1:
                if self.output_activation == "sigmoid":
                    A = self.sigmoid(Z)
                else:
                    A = Z  # Linear output
            else:
                if self.activation == "relu":
                    A = self.relu(Z)
                else:
                    A = Z  # Default to linear
            self.A.append(A)
        return self.A[-1]

    def compute_loss(self, Y_pred, Y):
        m = Y.shape[0]
        if self.output_activation == "sigmoid":
            loss = -(
                Y * np.log(Y_pred + 1e-15) + (1 - Y) * np.log(1 - Y_pred + 1e-15)
            ).mean()
        else:
            loss = np.mean((Y - Y_pred) ** 2)
        return loss

    def backward(self, Y):
        m = Y.shape[0]
        self.dW = [np.zeros_like(w) for w in self.weights]
        self.dB = [np.zeros_like(b) for b in self.biases]
        # Output layer error
        if self.output_activation == "sigmoid":
            delta = self.A[-1] - Y  # Derivative of loss w.r.t sigmoid activation
        else:
            delta = 2 * (self.A[-1] - Y)  # Derivative of MSE w.r.t linear activation
        # Backpropagate
        for i in reversed(range(len(self.weights))):
            self.dW[i] = np.dot(self.A[i].T, delta) / m
            self.dB[i] = np.sum(delta, axis=0, keepdims=True) / m
            if i != 0:
                if self.activation == "relu":
                    delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(
                        self.Z[i - 1]
                    )
                else:
                    delta = np.dot(delta, self.weights[i].T)
        return

    def update_parameters(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.dW[i]
            self.biases[i] -= self.learning_rate * self.dB[i]

    def fit(self, X, Y, epochs=1000, verbose=True):
        for epoch in range(epochs):
            Y_pred = self.forward(X)
            loss = self.compute_loss(Y_pred, Y)
            self.backward(Y)
            self.update_parameters()
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss {loss}")

    def predict(self, X, threshold=0.5):
        Y_pred = self.forward(X)
        if self.output_activation == "sigmoid":
            return (Y_pred >= threshold).astype(int)
        return Y_pred
