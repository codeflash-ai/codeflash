def problem_p02632():
    import sys

    import numpy as np

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    MOD = 10**9 + 7

    def main(K, N, MOD):

        def fact_table(N, MOD):

            inv = np.empty(N, np.int64)

            inv[0] = 0

            inv[1] = 1

            for n in range(2, N):

                q, r = divmod(MOD, n)

                inv[n] = inv[r] * (-q) % MOD

            fact = np.empty(N, np.int64)

            fact[0] = 1

            for n in range(1, N):

                fact[n] = n * fact[n - 1] % MOD

            fact_inv = np.empty(N, np.int64)

            fact_inv[0] = 1

            for n in range(1, N):

                fact_inv[n] = fact_inv[n - 1] * inv[n] % MOD

            return fact, fact_inv, inv

        fact, fact_inv, inv = fact_table(3_000_000, MOD)

        f = np.zeros(K + 1, np.int64)

        f = fact[N - 1 : K + N] * fact_inv[N - 1] % MOD * fact_inv[0 : K + 1] % MOD

        x = 1

        for i in range(1, K + 1):

            x = (x * 25) % MOD

            f[i] = f[i] * x % MOD

        for i in range(1, K + 1):

            f[i] += f[i - 1] * 26

            f[i] %= MOD

        return f[-1]

    if sys.argv[-1] == "ONLINE_JUDGE":

        import numba

        from numba.pycc import CC

        i8 = numba.from_dtype(np.int64)

        signature = (i8, i8, i8)

        cc = CC("my_module")

        cc.export("main", signature)(main)

        cc.compile()

    from my_module import main

    K = int(readline())

    N = len(read().rstrip())

    print((main(K, N, MOD)))


problem_p02632()
