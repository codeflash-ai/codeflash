import numpy as np


def compute_mds_error(D_goal, D_current, i=None):
    if i is None:
        return sum(sum(np.power(D_goal - D_current, 2)))
    elif i < 0:
        return np.array(
            [sum(np.power(D_goal[k] - D_current[k], 2)) for k in range(len(D_goal))]
        )
    else:
        return sum(np.power(D_goal[i] - D_current[i], 2))
