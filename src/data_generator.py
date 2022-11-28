"""Module to generate pseudo data."""

import numpy as np


def generate_data(N=20, D=50, ld=10, sigma=5, density=0.2):
    """Generate data matrix X and observations Y."""
    np.random.seed(1)
    beta_star = np.random.randn(D)
    idxs = np.random.choice(range(D), int((1 - density) * D), replace=False)
    for idx in idxs:
        beta_star[idx] = 0
    X = np.random.randn(N, D)
    y = X.dot(beta_star) + np.random.normal(0, sigma, size=N)
    Q = 0.5 * np.identity(N)
    p = y
    A = np.vstack((X.T, -X.T))
    b = ld * np.ones(2 * D)
    return X, y, beta_star, Q, p, A, b
