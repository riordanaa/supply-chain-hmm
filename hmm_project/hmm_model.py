"""
HMM model for supply chain disruption detection.

Training: Supervised MLE (counting from labeled ground-truth data).
Inference: Uses hmmlearn's CategoricalHMM with frozen MLE parameters.

The key advantage over unsupervised Baum-Welch: since we have ground-truth
labels from the simulation, we can directly count transitions and emissions
for guaranteed optimal parameter estimates.
"""

import numpy as np
from hmmlearn.hmm import CategoricalHMM

from config import (
    STEADY, DISRUPTION, RECOVERY, STATE_NAMES,
    N_OBS, OBS_NAMES
)

N_STATES = 3  # Steady, Disruption, Recovery


class SupervisedHMM:
    """
    Hidden Markov Model trained via supervised Maximum Likelihood Estimation.

    Parameters are computed by direct counting from labeled sequences,
    then loaded into hmmlearn.CategoricalHMM for inference (Forward, Viterbi).
    """

    def __init__(self, n_states=N_STATES, n_obs=N_OBS, laplace_alpha=1.0):
        self.n_states = n_states
        self.n_obs = n_obs
        self.laplace_alpha = laplace_alpha  # Laplace smoothing constant

        # Learned parameters
        self.pi = None       # Initial state distribution (n_states,)
        self.A = None        # Transition matrix (n_states, n_states)
        self.B = None        # Emission matrix (n_states, n_obs)

        # hmmlearn model for inference
        self._model = None

    def train(self, state_sequences, obs_sequences):
        """
        Train HMM parameters via supervised MLE (direct counting).

        Args:
            state_sequences: list of numpy arrays, each with integer state labels
            obs_sequences: list of numpy arrays, each with integer observation indices
        """
        N = self.n_states
        M = self.n_obs
        alpha = self.laplace_alpha

        # --- Initial state distribution ---
        pi_counts = np.zeros(N) + alpha
        for seq in state_sequences:
            pi_counts[seq[0]] += 1
        self.pi = pi_counts / pi_counts.sum()

        # --- Transition matrix ---
        A_counts = np.zeros((N, N)) + alpha
        for seq in state_sequences:
            for t in range(len(seq) - 1):
                A_counts[seq[t], seq[t + 1]] += 1
        # Normalize rows
        self.A = A_counts / A_counts.sum(axis=1, keepdims=True)

        # --- Emission matrix ---
        B_counts = np.zeros((N, M)) + alpha
        for state_seq, obs_seq in zip(state_sequences, obs_sequences):
            for t in range(len(state_seq)):
                B_counts[state_seq[t], obs_seq[t]] += 1
        # Normalize rows
        self.B = B_counts / B_counts.sum(axis=1, keepdims=True)

        # --- Build hmmlearn model with frozen parameters ---
        self._build_hmmlearn_model()

        return self

    def _build_hmmlearn_model(self):
        """Create an hmmlearn CategoricalHMM with our MLE parameters frozen."""
        model = CategoricalHMM(n_components=self.n_states, n_features=self.n_obs)

        # Set parameters
        model.startprob_ = self.pi
        model.transmat_ = self.A
        model.emissionprob_ = self.B

        self._model = model

    def forward_probabilities(self, obs_sequence):
        """
        Compute filtered state probabilities P(S_t | o_1:t) using the Forward algorithm.

        Args:
            obs_sequence: numpy array of integer observation indices, shape (T,)

        Returns:
            filtered: numpy array of shape (T, n_states) with P(S_t=i | o_1:t)
        """
        obs_sequence = np.asarray(obs_sequence, dtype=int)
        T = len(obs_sequence)
        N = self.n_states

        # Forward algorithm with scaling (manual implementation for clarity)
        alpha = np.zeros((T, N))
        filtered = np.zeros((T, N))

        # Initialization
        alpha[0, :] = self.pi * self.B[:, obs_sequence[0]]
        c0 = alpha[0, :].sum()
        if c0 > 0:
            alpha[0, :] /= c0
        filtered[0, :] = alpha[0, :]

        # Induction
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.sum(alpha[t - 1, :] * self.A[:, j]) * self.B[j, obs_sequence[t]]
            ct = alpha[t, :].sum()
            if ct > 0:
                alpha[t, :] /= ct
            filtered[t, :] = alpha[t, :]

        return filtered

    def viterbi(self, obs_sequence):
        """
        Find the most likely state sequence using the Viterbi algorithm.

        Args:
            obs_sequence: numpy array of integer observation indices, shape (T,)

        Returns:
            path: numpy array of integer state labels, shape (T,)
            log_prob: log probability of the best path
        """
        obs_col = np.asarray(obs_sequence, dtype=int).reshape(-1, 1)
        log_prob, path = self._model.decode(obs_col, algorithm="viterbi")
        return path, log_prob

    def predict_proba(self, obs_sequence):
        """
        Compute posterior state probabilities using hmmlearn's score_samples.
        This uses the forward-backward algorithm (smoothed, not filtered).

        For real-time detection, use forward_probabilities() instead.
        """
        obs_col = np.asarray(obs_sequence, dtype=int).reshape(-1, 1)
        log_prob, posteriors = self._model.score_samples(obs_col)
        return posteriors

    def print_parameters(self):
        """Print trained parameters in a clean, readable format."""
        print("=" * 60)
        print("TRAINED HMM PARAMETERS")
        print("=" * 60)

        print("\nInitial State Distribution (pi):")
        for i in range(self.n_states):
            print(f"  P(S_0 = {STATE_NAMES[i]:12s}) = {self.pi[i]:.4f}")

        print(f"\nTransition Matrix A ({self.n_states}x{self.n_states}):")
        print(f"{'From / To':>14s}", end="")
        for j in range(self.n_states):
            print(f"  {STATE_NAMES[j]:>12s}", end="")
        print()
        for i in range(self.n_states):
            print(f"{STATE_NAMES[i]:>14s}", end="")
            for j in range(self.n_states):
                print(f"  {self.A[i, j]:12.4f}", end="")
            print()

        print(f"\nEmission Matrix B ({self.n_states}x{self.n_obs}):")
        print(f"{'State / Obs':>14s}", end="")
        for o in range(self.n_obs):
            print(f"  obs={o}", end="")
        print()
        for i in range(self.n_states):
            print(f"{STATE_NAMES[i]:>14s}", end="")
            for o in range(self.n_obs):
                print(f"  {self.B[i, o]:.4f}", end="")
            print()

        print(f"\nObservation key:")
        for o in range(self.n_obs):
            print(f"  obs={o}: {OBS_NAMES[o]}")
