import numpy as np


def gradient(n_features, n_samples, y, X, w, b, subgrad, lambda1, lambda2):
    for i in range(n_features):
        for n in range(n_samples):
            subgrad[i] += (-y[n] * X[n][i]) if y[n] * (np.dot(X[n], w) + b) < 1 else 0
        subgrad[i] += lambda1 * (-1 if w[i] < 0 else 1) + 2 * lambda2 * w[i]

    return subgrad
