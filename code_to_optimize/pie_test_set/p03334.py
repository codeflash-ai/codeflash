def problem_p03334():
    from math import sqrt

    from itertools import product

    import numpy as np

    def helper(N, D):

        if D % 2 == 1:

            grid = np.zeros((N, N), dtype=bool)

            grid[::2, :] = True

            grid[:, 1::2] = ~grid[:, 1::2]

        elif D % 4 == 2:

            grid = np.zeros((N, N), dtype=bool)

            grid[::2, :] = True

        else:

            M = (N - 1) // 2 + 1

            D //= 4

            g = helper(M, D)

            grid = np.zeros((2 * M, 2 * M), dtype=bool)

            grid[::2, ::2] = g

            grid[1::2, ::2] = g

            grid[::2, 1::2] = g

            grid[1::2, 1::2] = g

            grid = grid[:N, :N]

        return grid

    N, D1, D2 = list(map(int, input().split()))

    M = N * 2

    g1 = helper(N * 2, D1)

    g2 = helper(N * 2, D2)

    ans = g1 & g2

    for x, y in np.transpose(np.nonzero(ans))[: N * N]:

        print((x, y))


problem_p03334()
