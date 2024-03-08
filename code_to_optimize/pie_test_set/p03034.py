def problem_p03034():
    import sys

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    import numpy as np

    N = int(readline())

    S = np.array(read().split(), np.int64)

    def F(d):

        # 公差を固定

        L = S[:-d:d].cumsum()

        R = S[::-1][:-d:d].cumsum()

        if (N - 1) % d == 0:

            n = (N - 1) // d

            x = np.arange(1, n + 1)

            bl = x + x <= n + 1

            L = L[bl]
            R = R[bl]

        return (L + R).max()

    def G(n):

        # 項数n+1を固定

        D = (N - 1 - n) // n

        L = np.zeros(D + 1, np.int64)

        R = np.zeros(D + 1, np.int64)

        for i in range(1, n + 1):

            L += S[: i * (D + 1) : i]

            R += S[::-1][: i * (D + 1) : i]

        ok = np.ones(D + 1, np.bool)

        overlap = np.arange(D + 1) * (n + n) >= N - 1

        overlap[1:] &= (N - 1) % np.arange(1, D + 1) == 0

        x = L + R

        x[overlap] = 0

        return x.max()

    L = int((N - 1) ** 0.5 + 10)

    L = min(N - 1, L)

    x = max(F(n) for n in range(1, L + 1))

    y = max(G(n) for n in range(1, L + 1))

    answer = max(x, y)

    print(answer)


problem_p03034()
