def problem_p02970(input_data):
    import numpy as np

    N, D = list(map(int, input_data.split()))

    return int(np.ceil(N / (2 * D + 1)))
