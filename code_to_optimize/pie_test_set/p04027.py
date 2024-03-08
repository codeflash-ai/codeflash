def problem_p04027():
    import sys

    input = sys.stdin.readline

    import numpy as np

    MOD = 10**9 + 7

    N, C = list(map(int, input().split()))

    A = [int(x) for x in input().split()]

    B = [int(x) for x in input().split()]

    # (i,j) に j^i を入れる

    kth_pow = np.ones((C + 1, 401), dtype=np.int64)

    rng = np.arange(401, dtype=np.int64)

    for i in range(1, C + 1):

        kth_pow[i] = kth_pow[i - 1] * rng % MOD

    kth_pow_cum = kth_pow.cumsum(axis=1) % MOD

    dp = np.zeros((C + 1), dtype=np.int64)  # これまで配った個数、合計点

    dp[0] = 1

    for a, b in zip(A, B):

        arr = kth_pow_cum[:, b] - kth_pow_cum[:, a - 1]

        prev = dp

        dp = np.zeros(C + 1, dtype=np.int64)

        for n in range(C + 1):

            dp[n:] += arr[n] * prev[: C + 1 - n] % MOD

        dp %= MOD

    answer = dp[C]

    print(answer)


problem_p04027()
