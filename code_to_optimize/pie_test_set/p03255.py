def problem_p03255():
    I = lambda: list(map(int, input().split()))

    n, p = I()

    s = [0]

    for i in I():

        s += [s[-1] + i]

    for t in range(1, n + 1):

        m = 5 * s[n] + t * p

        c = n - t * 2

        while c > 0:

            m += 2 * s[c]

            c -= t

        a = m if t == 1 else min(a, m)

    print((a + n * p))


problem_p03255()
