def problem_p03720():
    import numpy as np

    N, M = [int(x) for x in input().split()]

    L = [[int(y) for y in input().split()] for _ in range(M)]

    L = sum(L, [])

    for i in np.arange(1, N + 1):

        print((L.count(i)))


problem_p03720()
