def problem_p03092():
    import numpy as np

    N, A, B, *P = list(map(int, open(0).read().split()))

    dp = np.full(N + 1, 10**18, np.int64)

    dp[0] = 0

    for p in P:

        dp[p] = min(dp[:p])

        dp[p + 1 :] += B

        dp[:p] += A

    print((min(dp)))


problem_p03092()
