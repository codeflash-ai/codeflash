def problem_p02960():
    S = eval(input())

    N = len(S)

    dp = [[0] * (13) for _ in range(N + 1)]

    dp[0][0] = 1

    mod = 10**9 + 7

    for n in range(N):

        s = S[n]

        if s != "?":

            s = int(s) * pow(10, (N - n - 1), 13) % 13

            for i in range(13):

                idx = (s + i) % 13

                dp[n + 1][idx] = dp[n][i] % mod

        else:

            for i in range(10):

                s = i * pow(10, (N - n - 1), 13) % 13

                for j in range(13):

                    idx = (s + j) % 13

                    dp[n + 1][idx] += dp[n][j] % mod

    print((dp[-1][5] % mod))


problem_p02960()
