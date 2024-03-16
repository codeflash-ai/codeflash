def problem_p02731(input_data):
    import numpy as np

    L = np.array(int(eval(input_data)), dtype="float128")

    ans = (L / 3) ** 3

    return ans
