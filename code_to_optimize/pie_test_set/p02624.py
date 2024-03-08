def problem_p02624():
    import sys

    import numpy as np

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    def prime_table(N):

        is_prime = np.zeros(N, np.int64)

        is_prime[2:3] = 1

        is_prime[3::2] = 1

        for p in range(3, N, 2):

            if p * p >= N:

                break

            if is_prime[p]:

                is_prime[p * p :: p + p] = 0

        return is_prime, np.where(is_prime)[0]

    def main(N, primes):

        div = np.ones(N + 1, dtype=np.int64)

        for p in primes:

            for i in range(N // p + 1):

                div[p * i] += div[i]

        div *= np.arange(N + 1)

        return div.sum()

    if sys.argv[-1] == "ONLINE_JUDGE":

        import numba

        from numba.pycc import CC

        i8 = numba.int64

        cc = CC("my_module")

        def cc_export(f, signature):

            cc.export(f.__name__, signature)(f)

            return numba.njit(f)

        main = cc_export(main, (i8, i8[:]))

        cc.compile()

    from my_module import main

    N = int(read())

    is_prime, primes = prime_table(N + 1)

    print((main(N, primes)))


problem_p02624()
