def problem_p03644(input_data):
    import numpy as np

    N = int(eval(input_data))

    return 2 ** int(np.log2(N))
