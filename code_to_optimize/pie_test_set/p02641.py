def problem_p02641():
    import numpy as np

    X, N = list(map(int, input().split()))

    if N == 0:

        print(X)

    else:

        l = list(map(int, input().split()))

        app = [x for x in range(X - int(N / 2) - 1, X + int(N / 2) + 1) if x not in l]

        print((app[np.argmin([abs(x - X) for x in app])]))


problem_p02641()
