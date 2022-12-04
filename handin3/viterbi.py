import numpy as np


def viterbi(obs: np.ndarray, pi: np.ndarray, T: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Viterbi algorithm for finding the most probable state sequence
    given an observation sequence and HMM model parameters.

    Args:
        obs: Observation sequence [np.ndarray(int,float) of shape (N,)]
        pi: Initial state distribution [np.ndarray(int,float) of shape (K,)]
        T: Transition matrix [np.ndarray(int,float) of shape (K,K)]
        E: Emission matrix [np.ndarray(int,float) of shape (K,X)]

    Returns:
        list of most probable state sequence [np.ndarray(int) of shape (N,)]
    """
    # Check observation sequence format
    assert ((obs.dtype == float) or (obs.dtype == int)
            ), "Observation sequence must be a numpy array of floats or ints"

    # Check that the probabilities are valid, i.e. properly normalized.
    assert np.allclose(
        pi.sum(), 1), "Initial state probabilities must sum to 1"
    assert np.allclose(
        T.sum(axis=1), 1), "Transition probabilities must sum to 1"
    assert np.allclose(
        E.sum(axis=1), 1), "Emission probabilities must sum to 1"

    # Number of states and observations
    K, N = pi.shape[0], len(obs)

    # Forward pass
    omega = np.zeros((K, N))
    omega[:, 0] = pi * E[:, obs[0]]
    for n in range(1, N):
        omega[:, n] = np.max(omega[:, n-1] * T,
                             axis=0) * E[:, obs[n]]

    # Backward pass
    states = np.zeros(N, dtype=int)
    states[-1] = np.argmax(omega[:, -1])
    for n in range(N-2, -1, -1):
        states[n] = np.argmax(omega[:, n] * T[:, states[n+1]])

    return states
