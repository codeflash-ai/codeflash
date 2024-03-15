def problem_p02992(input_data):
    def f(n, k):

        lim = int((n + 0.1) ** 0.5) + 1

        ws = []

        s = 0

        for i in range(1, lim):

            w = n // i - n // (i + 1)

            ws.append(w)

            s += w

        ws += [1] * (n - s)

        m = len(ws)

        dp0 = ws[::-1]

        dp1 = [0] * m

        for _ in range(k - 1):

            s = 0

            for j in range(m):

                s += dp0[j]

                dp1[m - j - 1] = (s * ws[j]) % md

            dp0 = dp1[:]

        return sum(dp0) % md

    md = 10**9 + 7

    n, k = list(map(int, input_data.split()))

    f(n, k)
