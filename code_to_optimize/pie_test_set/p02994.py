def problem_p02994(input_data):
    import numpy as np

    n, l = [int(x) for x in input_data.split()]

    x = np.arange(l, l + n)

    y = np.abs(x)

    x = np.delete(x, np.argmin(y))

    return np.sum(x)
