def problem_p02970():
    import numpy as np

    N, D = list(map(int, input().split()))

    print((int(np.ceil(N / (2 * D + 1)))))


problem_p02970()
