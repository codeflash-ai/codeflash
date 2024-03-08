def problem_p03172():
    n, k = list(map(int, input().split()))

    a = list(map(int, input().split()))

    mod = 10**9 + 7

    dp = [[0] * (k + 1) for _ in range(n + 1)]

    dp[0][0] = 1

    for i in range(n):

        dp[i + 1][0] = dp[i][0]

        for j in range(1, k + 1):

            dp[i + 1][j] = (dp[i + 1][j - 1] + dp[i][j]) % mod

        for j in range(k, a[i], -1):

            dp[i + 1][j] = (dp[i + 1][j] - dp[i + 1][j - a[i] - 1]) % mod

    print((dp[n][-1]))


problem_p03172()
