def problem_p03681():
    # 差が1か0のみ隣り合わない 適当に毎回割る再帰

    n, m = list(map(int, input().split()))

    if abs(n - m) > 1:

        print((0))

    elif n == m:

        ans = 1

        while n > 0:

            ans *= n

            n -= 1

            ans %= 10**9 + 7

        ans = (ans * ans * 2) % (10**9 + 7)

        print(ans)

    else:

        ans = max(n, m)

        t = 1

        x = min(n, m)

        while x > 0:

            t *= x

            x -= 1

            t %= 10**9 + 7

        ans = (ans * t * t) % (10**9 + 7)

        print(ans)


problem_p03681()
