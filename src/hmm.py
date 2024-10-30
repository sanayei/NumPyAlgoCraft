import numpy as np
# ----------------------------
#   Hidden Markov Models (HMM)
# ----------------------------


class HiddenMarkovModel:
    def __init__(self, n_states, n_observations):
        self.n_states = n_states
        self.n_observations = n_observations
        # Initialize transition, emission, and initial probabilities
        self.A = np.random.rand(n_states, n_states)
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.B = np.random.rand(n_states, n_observations)
        self.B /= self.B.sum(axis=1, keepdims=True)
        self.pi = np.random.rand(n_states)
        self.pi /= self.pi.sum()

    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.n_states))
        alpha[0] = self.pi * self.B[:, O[0]]
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t - 1] * self.A[:, j]) * self.B[j, O[t]]
        return alpha

    def backward(self, O):
        T = len(O)
        beta = np.zeros((T, self.n_states))
        beta[T - 1] = 1
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, O[t + 1]] * beta[t + 1])
        return beta

    def baum_welch(self, sequences, n_iter=10):
        for iteration in range(n_iter):
            A_num = np.zeros((self.n_states, self.n_states))
            A_den = np.zeros(self.n_states)
            B_num = np.zeros((self.n_states, self.n_observations))
            B_den = np.zeros(self.n_states)
            pi_new = np.zeros(self.n_states)

            for O in sequences:
                alpha = self.forward(O)
                beta = self.backward(O)
                xi = np.zeros((len(O) - 1, self.n_states, self.n_states))
                for t in range(len(O) - 1):
                    denominator = np.sum(
                        alpha[t] * self.A * self.B[:, O[t + 1]] * beta[t + 1], axis=1
                    ).sum()
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            xi[t, i, j] = (
                                alpha[t, i]
                                * self.A[i, j]
                                * self.B[j, O[t + 1]]
                                * beta[t + 1, j]
                            )
                    xi[t] /= denominator
                gamma = np.sum(xi, axis=2)
                gamma = np.vstack([gamma, np.sum(xi[-1], axis=1)])

                pi_new += gamma[0]
                A_num += np.sum(xi, axis=0)
                A_den += np.sum(gamma[:-1], axis=0)

                for t in range(len(O)):
                    B_num[:, O[t]] += gamma[t]
                B_den += np.sum(gamma, axis=0)

            self.pi = pi_new / len(sequences)
            self.A = A_num / A_den[:, None]
            self.B = B_num / B_den[:, None]
            print(f"Iteration {iteration+1} completed.")

    def viterbi(self, O):
        T = len(O)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        delta[0] = self.pi * self.B[:, O[0]]
        psi[0] = 0

        for t in range(1, T):
            for j in range(self.n_states):
                seq_probs = delta[t - 1] * self.A[:, j]
                psi[t, j] = np.argmax(seq_probs)
                delta[t, j] = np.max(seq_probs) * self.B[j, O[t]]

        path = np.zeros(T, dtype=int)
        path[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path
