def problem_p03681(input_data):
    # 差が1か0のみ隣り合わない 適当に毎回割る再帰

    n, m = list(map(int, input_data.split()))

    if abs(n - m) > 1:

        return 0

    elif n == m:

        ans = 1

        while n > 0:

            ans *= n

            n -= 1

            ans %= 10**9 + 7

        ans = (ans * ans * 2) % (10**9 + 7)

        return ans

    else:

        ans = max(n, m)

        t = 1

        x = min(n, m)

        while x > 0:

            t *= x

            x -= 1

            t %= 10**9 + 7

        ans = (ans * t * t) % (10**9 + 7)

        return ans
