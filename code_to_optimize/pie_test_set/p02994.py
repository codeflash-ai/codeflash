def problem_p02994():
    import numpy as np

    n, l = [int(x) for x in input().split()]

    x = np.arange(l, l + n)

    y = np.abs(x)

    x = np.delete(x, np.argmin(y))

    print((np.sum(x)))


problem_p02994()
