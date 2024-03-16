def problem_p02946(input_data):
    K = list(map(int, input_data.split()))

    import numpy as np

    tmp = []

    for i in range(K[1] - K[0] + 1, K[1] + K[0]):

        if abs(i) <= 1000000:

            return i
