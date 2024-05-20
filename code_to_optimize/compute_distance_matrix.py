import numpy as np


def compute_distance_matrix(Xs, D_current=None, i=None):
    if i is None:
        D_current = np.zeros((Xs.shape[0], Xs.shape[0]))
        for k in range(len(Xs)):
            for l in range(len(Xs)):
                D_current[k, l] = np.linalg.norm(Xs[k] - Xs[l], 2)
    else:
        for k in range(len(Xs)):
            D_current[k, i] = D_current[i, k] = np.linalg.norm(Xs[k] - Xs[i], 2)
    return D_current
