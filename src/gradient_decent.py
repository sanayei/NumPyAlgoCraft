import numpy as np
# ----------------------------
#   Gradient Descent and Its Variants
# ----------------------------


class GradientDescentOptimizer:
    def __init__(
        self,
        learning_rate=0.01,
        n_iterations=1000,
        variant="standard",
        beta=0.9,
        epsilon=1e-8,
    ):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.variant = variant
        self.beta = beta  # For momentum-based methods
        self.epsilon = epsilon
        self.velocity = None
        self.s = None  # For Adam optimizer
        self.r = None

    def optimize(self, initial_theta, gradient_func):
        theta = initial_theta.copy()
        if self.variant == "momentum":
            self.velocity = np.zeros_like(theta)
        elif self.variant == "adam":
            self.velocity = np.zeros_like(theta)
            self.s = np.zeros_like(theta)
            self.r = np.zeros_like(theta)
            beta1 = self.beta
            beta2 = 0.999
            for t in range(1, self.n_iterations + 1):
                grad = gradient_func(theta)
                self.velocity = beta1 * self.velocity + (1 - beta1) * grad
                self.s = beta2 * self.s + (1 - beta2) * (grad**2)
                v_corrected = self.velocity / (1 - beta1**t)
                s_corrected = self.s / (1 - beta2**t)
                theta -= self.lr * v_corrected / (np.sqrt(s_corrected) + self.epsilon)
            return theta
        for i in range(self.n_iterations):
            grad = gradient_func(theta)
            if self.variant == "momentum":
                self.velocity = self.beta * self.velocity + (1 - self.beta) * grad
                theta -= self.lr * self.velocity
            else:  # Standard gradient descent
                theta -= self.lr * grad
        return theta
