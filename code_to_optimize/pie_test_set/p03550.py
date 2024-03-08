def problem_p03550():
    N, Z, W = list(map(int, input().split()))

    a = list(map(int, input().split()))

    INF = 10**19

    dp = [[0] * 2 for i in range(N + 1)]

    for i in range(N - 1, -1, -1):

        dp[i][0] = -INF

        if i:
            Y = a[i - 1]

        else:
            Y = W

        dp[i][0] = max(dp[i][0], abs(Y - a[N - 1]))

        for j in range(i + 1, N):
            dp[i][0] = max(dp[i][0], dp[j][1])

        dp[i][1] = INF

        if i:
            X = a[i - 1]

        else:
            X = Z

        dp[i][1] = min(dp[i][1], abs(X - a[N - 1]))

        for j in range(i + 1, N):
            dp[i][1] = min(dp[i][1], dp[j][0])

    print((dp[0][0]))


problem_p03550()
