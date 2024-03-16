def problem_p04004(input_data):
    import sys

    input = sys.stdin.readline

    import numpy as np

    MOD = 10**9 + 7

    N, M, K = list(map(int, input_data.split()))

    def cumprod(arr):

        L = len(arr)
        Lsq = int(L**0.5 + 1)

        arr = np.resize(arr, Lsq**2).reshape(Lsq, Lsq)

        for n in range(1, Lsq):

            arr[:, n] *= arr[:, n - 1]
            arr[:, n] %= MOD

        for n in range(1, Lsq):

            arr[n] *= arr[n - 1, -1]
            arr[n] %= MOD

        return arr.ravel()[:L]

    U = 10**6

    x = np.full(U, 2, dtype=np.int64)
    x[0] = 1

    pow2 = cumprod(x)

    x = np.full(U, 3, dtype=np.int64)
    x[0] = 1

    pow3 = cumprod(x)

    x = np.full(U, pow(2, MOD - 2, MOD), dtype=np.int64)
    x[0] = 1

    pow2_inv = cumprod(x)

    x = np.full(U, pow(3, MOD - 2, MOD), dtype=np.int64)
    x[0] = 1

    pow3_inv = cumprod(x)

    x = np.arange(U, dtype=np.int64)
    x[0] = 1

    fact = cumprod(x)

    x = np.arange(U, 0, -1, dtype=np.int64)
    x[0] = pow(int(fact[-1]), MOD - 2, MOD)

    fact_inv = cumprod(x)[::-1]

    L = N + M

    A = np.zeros(N + M, dtype=np.int64)

    A[1:L] = (-1) * pow2[0 : L - 1] * pow3_inv[0 : L - 1] % MOD

    A[1:L] *= fact[K + 1 : K + L] * fact_inv[K] % MOD * fact_inv[1:L] % MOD
    A %= MOD

    A[1:L] *= pow3_inv[K + 1]
    A %= MOD

    A[0] = 3 * (1 - pow3_inv[K + 1]) % MOD * pow2_inv[1] % MOD

    np.cumsum(A, out=A)
    A %= MOD

    A *= pow3[:L]
    A %= MOD

    A *= pow2_inv[:L]
    A %= MOD

    comb = fact[N - 1 : N + M] * fact_inv[: M + 1] % MOD * fact_inv[N - 1] % MOD

    answer = (comb * pow3[K + M : K - 1 : -1] % MOD * A[N - 1 : N + M] % MOD).sum() % MOD

    return answer
