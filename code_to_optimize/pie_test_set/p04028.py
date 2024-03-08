def problem_p04028():
    n = int(eval(input()))

    s = len(eval(input()))

    mod = 10**9 + 7

    dp = [[0] * (n + 1) for _ in range(n + 1)]

    dp[0][0] = 1

    for i in range(1, n + 1):

        dp[i][0] = (dp[i - 1][0] + dp[i - 1][1]) % mod

        for j in range(1, n):

            dp[i][j] = (dp[i - 1][j - 1] * 2 + dp[i - 1][j + 1]) % mod

        dp[i][n] = dp[i - 1][n - 1] * 2 % mod

    s2 = pow(2, s, mod)

    rev = pow(s2, mod - 2, mod)

    print((dp[n][s] * rev % mod))


problem_p04028()
