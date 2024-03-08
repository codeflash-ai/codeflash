def problem_p03375():
    import sys

    input = sys.stdin.readline

    import numpy as np

    N, MOD = list(map(int, input().split()))

    """
    
    余事象を調べる。包除の原理を使う。
    
    A[n] = （1,2,...,n）が1杯以下、他は何でも良い
    
    B[n,l] : (1,2,...,n) をlグループに分ける方法の個数
    
    A[n]
    
    ・0杯のグループあり
    
    ・なし
    
    """

    B = np.zeros((N + 1, N + 1), dtype=np.int64)

    B[0, 0] = 1

    for n in range(1, N + 1):

        # 1番を単独で使う

        B[n, 1:] = B[n - 1, :-1]

        # 1番をどこかに混ぜてもらう

        B[n, 1:] += B[n - 1, 1:] * np.arange(1, N + 1) % MOD

        B[n] %= MOD

    # 2^{kl}

    pow_2 = np.ones((N + 1, N + 1), dtype=np.int64)

    for n in range(1, N + 1):

        pow_2[1, n] = 2 * pow_2[1, n - 1] % MOD

    for n in range(2, N + 1):

        pow_2[n] = pow_2[n - 1] * pow_2[1] % MOD

    A = np.zeros(N + 1, dtype=np.int64)

    for n in range(N + 1):

        A[n] = (
            pow(2, pow(2, N - n, MOD - 1), MOD)
            * B[n, 1:]
            % MOD
            * (pow_2[N - n, 1:] + pow_2[N - n, :-1] * np.arange(1, N + 1) % MOD)
            % MOD
        ).sum() % MOD

    comb = np.zeros((N + 1, N + 1), dtype=np.int64)

    comb[:, 0] = 1

    for n in range(1, N + 1):

        comb[n, 1:] = (comb[n - 1, 1:] + comb[n - 1, :-1]) % MOD

    A[::2] *= -1

    A *= comb[N]

    A %= MOD

    answer = pow(2, pow(2, N, MOD - 1), MOD) - A.sum()

    answer %= MOD

    print(answer)


problem_p03375()
