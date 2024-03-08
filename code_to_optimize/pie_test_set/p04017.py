def problem_p04017():
    import sys

    input = sys.stdin.readline

    import bisect

    n = int(eval(input()))

    X = list(map(int, input().split()))

    l = int(eval(input()))

    U = 17

    dp = [[0] * n for _ in range(U + 1)]

    for i, x in enumerate(X):

        t = bisect.bisect_left(X, x + l)

        dp[0][i] = bisect.bisect_right(X, x + l) - 1

    for k in range(U):

        for i in range(n):

            dp[k + 1][i] = dp[k][dp[k][i]]

    def test(x, a, b):

        for i in range(U, -1, -1):

            if x >> i & 1:

                a = dp[i][a]

        return a >= b

    def solve(a, b):

        if a > b:

            a, b = b, a

        ng = 0

        ok = n - 1

        while ok - ng > 1:

            mid = (ng + ok) // 2

            if test(mid, a, b):

                ok = mid

            else:

                ng = mid

        print(ok)

    q = int(eval(input()))

    for _ in range(q):

        a, b = list(map(int, input().split()))

        a -= 1

        b -= 1

        solve(a, b)


problem_p04017()
