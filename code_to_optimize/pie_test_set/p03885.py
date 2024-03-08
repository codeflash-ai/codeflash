def problem_p03885():
    import sys

    readline = sys.stdin.readline

    readlines = sys.stdin.readlines

    sys.setrecursionlimit(10**7)

    import numpy as np

    MOD = 10**9 + 7

    N = int(readline())

    C = np.array([line.split() for line in readlines()], np.int8)

    def rank(A):

        if (A == 0).all():

            return 0

        i = np.nonzero(A[:, 0])[0]

        if len(i) == 0:

            return rank(A[:, 1:])

        i = i[0]

        temp = A[i].copy()

        A[i] = A[0]

        A[0] = temp

        A[1:] ^= A[1:, 0][:, None] * A[0][None, :]

        return 1 + rank(A[1:, 1:])

    r = rank(C)

    pow2 = np.ones(301, dtype=np.int64)

    for n in range(1, 301):

        pow2[n] = pow2[n - 1] * 2 % MOD

    # N次元空間から、M本のベクトルを選んで、D次元部分空間を生成する方法の個数

    dp = np.zeros((301, 301, 301), dtype=np.int64)

    dp[:, 0, 0] = 1

    for M in range(1, 301):

        dp[:, M, :M] += dp[:, M - 1, :M] * pow2[:M] % MOD

        dp[:, M, 1 : M + 1] += dp[:, M - 1, 0:M] * (pow2[:, None] - pow2[None, 0:M]) % MOD

        dp[:, M, :] %= MOD

    # C=ABのrankがrとなる方法の総数

    x = 0

    for n in range(r, N + 1):

        x += dp[N, N, n] * dp[N, n, r] % MOD * pow(2, N * (N - n), MOD) % MOD

    x %= MOD

    answer = x * pow(int(dp[N, N, r]), MOD - 2, MOD) % MOD

    print(answer)


problem_p03885()
