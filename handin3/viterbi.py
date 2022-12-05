import numpy as np


def log_viterbi(obs: np.ndarray, pi: np.ndarray, T: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Viterbi algorithm for finding the most probable state sequence
    given an observation sequence and HMM model parameters.
    Uses log transformation.

    Args:
        obs: Observation sequence [np.ndarray(int,float) of shape (N,)]
        pi: Initial state distribution [np.ndarray(int,float) of shape (K,)]
        T: Transition matrix [np.ndarray(int,float) of shape (K,K)]
        E: Emission matrix [np.ndarray(int,float) of shape (K,X)]

    Returns:
        list of most probable state sequence [np.ndarray(int) of shape (N,)]
    """
    # Check observation sequence format
    assert isinstance(
        obs, np.ndarray), "Observation sequence must be a numpy array"
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
    omega[:, 0] = np.log(pi) + np.log(E[:, obs[0]])
    # Forward pass
    for n in range(1, N):
        for j in range(K):
            omega[j, n] = np.log(E[j, obs[n]]) + \
                np.max(omega[:, n-1] + np.log(T[j, :]))

    # Backward pass
    z = np.zeros(N, dtype=int)
    z[N-1] = np.argmax(omega[:, -1])
    for n in range(N-2, -1, -1):
        z[n] = np.argmax(np.log(E[z[n+1], obs[n+1]]) +
                         omega[:, n] + np.log(T[:, z[n+1]]))

    return z
