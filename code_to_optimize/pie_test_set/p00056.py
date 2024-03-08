def problem_p00056():
    import sys

    def sieve(m):

        N = list(range(1, m + 2, 2))

        r = int(m**0.5)

        h = len(N)

        N[0] = 0

        for i in range(h):

            x = N[i]

            if x > r:
                break

            if x and i + x < h:
                N[i + x : h : x] = [0] * ((h - 1 - i - x) / x + 1)

        return N

    def f0056(n):

        x = 0

        if n < 4:
            x = 0

        elif n == 4:
            x = 1

        elif n % 2 == 1:

            if SIEVES[(n - 2) / 2] != 0:

                x = 1

        else:

            a = n / 2

            for e in PRIMES:

                if e > a:
                    break

                if SIEVES[(n - e) / 2] != 0:

                    x += 1

        return x

    SIEVES = sieve(50000)

    PRIMES = [_f for _f in SIEVES if _f]

    for n in sys.stdin:

        n = int(n)

        if n == 0:
            break

        print(f0056(n))


problem_p00056()
