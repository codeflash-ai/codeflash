def problem_p02945(input_data):
    import numpy as np

    a, b = list(map(int, input_data.split()))

    ans = [a + b, a - b, a * b]

    return np.amax(ans)
