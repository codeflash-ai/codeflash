def problem_p02669():
    a = 2, 3, 5

    def f(n):

        m = memo.get(n)

        if m is None:

            s = {(0, 0, 0)}.union(*({(n // p, p, c), (-(-n // p), p, c)} for p, c in zip(a, b)))

            m = min(f(x) + abs(x * p - n) * d + c for x, p, c in s)

            memo[n] = m

        return m

    t = int(eval(input()))

    for _ in range(t):

        n, *b, d = list(map(int, input().split()))

        memo = {0: 0, 1: d}

        print((f(n)))


problem_p02669()
