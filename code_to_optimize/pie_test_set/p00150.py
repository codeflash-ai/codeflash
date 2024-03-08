def problem_p00150():
    import bisect as bs

    def prime(m):

        N = list(range(1, m + 1, 2))

        N[0] = 2

        for i in range(1, int(m**0.5) + 1):

            x = N[i]

            if x:
                N[i + x :: x] = [0] * len(N[i + x :: x])

        return [_f for _f in N if _f]

    P = prime(10000)

    x = [a for a, b in zip(P[1:], P[:-1]) if a - b == 2]

    while 1:

        n = eval(input())

        if n == 0:
            break

        a = x[bs.bisect_right(x, n) - 1]

        print(a - 2, a)


problem_p00150()
