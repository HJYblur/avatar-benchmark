"""
Calculate the Soft-DTW distance between two time series.
"""

import numpy as np


def soft_dtw(X, Y, gamma=1.0):
    """
    Compute the Soft-DTW distance between two time series X and Y.

    Args:
        X: numpy array of shape (n, d) representing the first time series
        Y: numpy array of shape (m, d) representing the second time series
        gamma: smoothing parameter

    Returns:
        Soft-DTW distance between X and Y
    """
    n, d = X.shape
    m, _ = Y.shape

    D = np.zeros((n + 1, m + 1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(X[i - 1] - Y[j - 1]) ** 2
            D[i, j] = (
                cost
                + min(
                    D[i - 1, j],  # insertion
                    D[i, j - 1],  # deletion
                    D[i - 1, j - 1],  # match
                )
                + gamma
                * np.log(
                    np.exp((D[i - 1, j] - D[i, j]) / gamma)
                    + np.exp((D[i, j - 1] - D[i, j]) / gamma)
                    + np.exp((D[i - 1, j - 1] - D[i, j]) / gamma)
                )
            )

    return D[n, m]
