def problem_p02648():
    import sys

    import numpy as np

    from numba import njit

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    # ノード、重さ→価値

    @njit("(i4,i4[:],i4[:])", cache=True)
    def precompute(N, V, W):

        U = 10**5 + 10

        dp = np.zeros((1024, U), np.int32)

        for v in range(1, min(1024, N + 1)):

            p = v // 2

            dp[v] = dp[p]

            dp[v, W[v] :] = np.maximum(dp[v, W[v] :], dp[p, : -W[v]] + V[v])

            for i in range(1, U):

                dp[v][i] = max(dp[v][i], dp[v][i - 1])

        return dp

    @njit("(i4,i4[:],i4[:],i4[:],i4[:])", cache=True)
    def main(N, V, W, v, L):

        dp = precompute(N, V, W)

        values = np.empty(1024, np.int32)

        weights = np.empty(1024, np.int32)

        def solve(i, lim, values, weights, dp):

            if i < 1024:

                return dp[i][lim]

            values[0] = 0

            weights[0] = 0

            p = 0

            while i >= 1024:

                for j in range(1 << p):

                    values[j + (1 << p)] = values[j] + V[i]

                    weights[j + (1 << p)] = weights[j] + W[i]

                p += 1

                i >>= 1

            best = 0

            for n in range(1 << p):

                if weights[n] > lim:

                    continue

                x = dp[i][lim - weights[n]] + values[n]

                best = max(best, x)

            return best

        Q = len(v)

        for i in range(Q):

            x = solve(v[i], L[i], values, weights, dp)

            print(x)

    N = int(readline())

    stdin = np.array(read().split(), np.int32)

    V = np.zeros(N + 1, np.int32)

    W = np.zeros(N + 1, np.int32)

    V[1:] = stdin[: N + N : 2]

    W[1:] = stdin[1 : N + N : 2]

    query = stdin[N + N + 1 :]

    v, L = query[::2], query[1::2]

    main(N, V, W, v, L)


problem_p02648()
