def problem_p02316():
    def solve_knapsack(s, n, v, w):

        dp = [[0 for x in range(s + 1)] for x in range(n + 1)]

        for i in range(n):

            for j in range(1, s + 1):

                if w[i] > j:

                    dp[i + 1][j] = dp[i][j]

                else:

                    dp[i + 1][j] = max(
                        dp[i][j], max(dp[i + 1][j - w[i]] + v[i], dp[i][j - w[i]] + v[i])
                    )

        return dp[n][s]

    n, s = list(map(int, input().split()))

    v = [0] * n

    w = [0] * n

    for i in range(n):

        v[i], w[i] = list(map(int, input().split()))

    print((solve_knapsack(s, n, v, w)))


problem_p02316()
